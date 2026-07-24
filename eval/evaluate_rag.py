"""
Évaluation automatique du système RAG Puls-Events avec Ragas.

Architecture :
- système évalué : chatbot RAG maison (FAISS + Mistral + LangChain) ;
- système évaluateur : Ragas, avec API legacy compatible avec la version
  actuellement installée du projet.
"""

import os
import sys

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, faithfulness

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Ajout de la racine du projet au PYTHONPATH.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.chatbot import PulsEventsChatbot  # noqa: E402

load_dotenv()


def load_test_set(csv_path: str) -> pd.DataFrame:
    """
    Charge et valide le jeu de test annoté.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier de test introuvable : {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_columns = {"question", "reference"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Le fichier de test doit contenir les colonnes : {required_columns}"
        )

    if df.empty:
        raise ValueError("Le jeu de test est vide.")

    return df


def build_eval_dataset(
    bot: PulsEventsChatbot,
    df_test: pd.DataFrame,
    max_samples: int = 5,
) -> Dataset:
    """
    Interroge le chatbot et construit un Dataset HuggingFace compatible Ragas.
    """
    rows = []

    for _, row in df_test.head(max_samples).iterrows():
        question = str(row["question"]).strip()
        reference = str(row["reference"]).strip()

        result = bot.ask(question)

        contexts = []
        for source in result["sources"]:
            if hasattr(source, "page_content"):
                contexts.append(str(source.page_content))
            elif isinstance(source, dict):
                title = str(source.get("title", ""))
                city = str(source.get("city", ""))
                contexts.append(f"{title} - {city}".strip(" -"))
            else:
                contexts.append(str(source))

        rows.append(
            {
                "question": question,
                "answer": str(result["answer"]),
                "contexts": contexts,
                "ground_truth": reference,
                "reference": reference,
            }
        )

    return Dataset.from_list(rows)


def build_ragas_evaluator():
    """
    Initialise le LLM et les embeddings utilisés par Ragas.

    Les wrappers LangChain sont employés car ils sont compatibles avec
    evaluate() et les métriques legacy de la version actuellement installée.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    eval_model = os.getenv("RAGAS_EVAL_MODEL", "open-mistral-nemo")
    embed_model = os.getenv("RAGAS_EMBED_MODEL", "mistral-embed")

    if not api_key:
        raise ValueError("La variable d'environnement MISTRAL_API_KEY est manquante.")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=eval_model,
        temperature=0,
        max_tokens=1024,
    )

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=embed_model,
        check_embedding_ctx_length=False,
    )

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return evaluator_llm, evaluator_embeddings


def save_results(scores_df: pd.DataFrame, output_path: str) -> None:
    """
    Enregistre les résultats détaillés dans un CSV UTF-8 avec BOM.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def classify_response(row: pd.Series) -> str:
    """
    Classe une réponse en tenant compte de la fidélité, de la récupération
    et de l'absence éventuelle de contexte.
    """
    faith = pd.to_numeric(row.get("faithfulness", 0), errors="coerce")
    precision = pd.to_numeric(row.get("context_precision", 0), errors="coerce")

    faith = 0.0 if pd.isna(faith) else float(faith)
    precision = 0.0 if pd.isna(precision) else float(precision)

    contexts = row.get("retrieved_contexts", [])
    no_context = contexts is None or len(contexts) == 0

    # Une absence de contexte ne peut pas être "correcte" si la référence
    # attend explicitement un événement ou une réponse positive.
    reference = str(row.get("reference", "")).lower()
    expected_event = any(
        term in reference
        for term in ["doit retrouver", "doit proposer", "oui."]
    )

    if no_context and expected_event:
        return "incorrecte"

    if faith >= 0.80 and precision >= 0.30:
        return "correcte"

    if faith >= 0.50:
        return "partiellement correcte"

    return "incorrecte"


def add_classification(
    scores_df: pd.DataFrame,
    eval_dataset: Dataset,
) -> pd.DataFrame:
    """
    Ajoute les informations lisibles et la classification aux scores Ragas.
    """
    dataset_df = pd.DataFrame(eval_dataset)

    for column in ["question", "answer", "reference", "ground_truth"]:
        if column in dataset_df.columns:
            scores_df[column] = dataset_df[column]

    scores_df["classification"] = scores_df.apply(classify_response, axis=1)

    return scores_df


def print_classification_summary(scores_df: pd.DataFrame) -> None:
    """
    Affiche le nombre et le pourcentage de réponses par catégorie.
    """
    print("\n📋 Résumé de classification :")

    counts = scores_df["classification"].value_counts(dropna=False)
    total = len(scores_df)

    for label in ["correcte", "partiellement correcte", "incorrecte"]:
        count = counts.get(label, 0)
        percentage = (count / total * 100) if total else 0
        print(f"- {label} : {count}/{total} ({percentage:.1f}%)")


def main() -> None:
    """
    Exécute l'évaluation Ragas de bout en bout.
    """
    test_csv_path = "eval/test_set.csv"
    output_csv_path = "eval/results/ragas_scores.csv"

    print("📥 Chargement du jeu de test...")
    df_test = load_test_set(test_csv_path)

    print("🤖 Chargement du chatbot RAG évalué...")
    bot = PulsEventsChatbot(index_path="data/faiss_index")

    print("🧱 Construction du dataset Ragas...")
    eval_dataset = build_eval_dataset(bot, df_test, max_samples=5)

    print("🧠 Initialisation de l'évaluateur Ragas...")
    evaluator_llm, evaluator_embeddings = build_ragas_evaluator()

    print("📊 Lancement de l'évaluation...")

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            context_precision,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    scores_df = result.to_pandas()
    scores_df = add_classification(scores_df, eval_dataset)

    print("💾 Sauvegarde des résultats...")
    save_results(scores_df, output_csv_path)

    print("\n✅ Évaluation terminée.")
    print(f"Résultats sauvegardés dans : {output_csv_path}")

    print("\n📈 Moyennes des métriques :")
    numeric_columns = scores_df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        print(f"- {column} : {scores_df[column].mean():.4f}")

    print_classification_summary(scores_df)


if __name__ == "__main__":
    main()