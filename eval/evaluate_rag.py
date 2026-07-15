"""
Évaluation automatique du système RAG Puls-Events avec Ragas.

Architecture :
- Le système évalué = chatbot RAG maison (FAISS + Mistral + LangChain)
- Le système évaluateur = Ragas avec un client Mistral configuré explicitement
  via une interface OpenAI-compatible

Objectifs :
1. Charger un jeu de test annoté
2. Interroger le chatbot RAG
3. Construire un dataset Ragas
4. Configurer explicitement le LLM et les embeddings d'évaluation
5. Calculer les métriques
6. Sauvegarder les résultats
"""

import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Ajout de la racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.chatbot import PulsEventsChatbot

from datasets import Dataset
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()


def load_test_set(csv_path: str) -> pd.DataFrame:
    """
    Charge le jeu de test annoté.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier de test introuvable : {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"question", "reference"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Le fichier doit contenir les colonnes {required_cols}"
        )

    if df.empty:
        raise ValueError("Le jeu de test est vide.")

    return df


def build_eval_dataset(bot: PulsEventsChatbot, df_test: pd.DataFrame, max_samples: int = 5) -> Dataset:
    """
    Construit un dataset HuggingFace compatible Ragas.
    """
    rows = []

    df_subset = df_test.head(max_samples).copy()

    for _, row in df_subset.iterrows():
        question = str(row["question"]).strip()
        reference = str(row["reference"]).strip()

        result = bot.ask(question)

        answer = result["answer"]
        contexts = [doc.page_content for doc in result["sources"]]

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference,
            }
        )

    return Dataset.from_list(rows)


def build_ragas_evaluator():
    """
    Construit explicitement le LLM et les embeddings pour Ragas
    via l'API OpenAI-compatible de Mistral.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    eval_model = os.getenv("RAGAS_EVAL_MODEL", "open-mistral-nemo")
    embed_model = os.getenv("RAGAS_EMBED_MODEL", "mistral-embed")

    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est manquante.")

    # LLM évaluateur
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=eval_model,
        temperature=0,
        max_tokens=512,
    )

    # Embeddings évaluateurs
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=embed_model,
    )

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return evaluator_llm, evaluator_embeddings


def save_results(scores_df: pd.DataFrame, output_path: str) -> None:
    """
    Sauvegarde les résultats dans un CSV.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    """
    Pipeline principal d'évaluation.
    """
    test_csv_path = "eval/test_set.csv"
    output_csv_path = "eval/results/ragas_scores.csv"

    print("📥 Chargement du jeu de test...")
    df_test = load_test_set(test_csv_path)

    print("🤖 Chargement du chatbot RAG évalué...")
    bot = PulsEventsChatbot(index_path="data/faiss_index")

    print("🧱 Construction du dataset Ragas...")
    eval_dataset = build_eval_dataset(bot, df_test, max_samples=5)

    print("🧠 Initialisation explicite de l'évaluateur Ragas...")
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

    print("💾 Sauvegarde des résultats...")
    save_results(scores_df, output_csv_path)

    print("\n✅ Évaluation terminée.")
    print(f"Résultats sauvegardés dans : {output_csv_path}")

    print("\n📈 Moyennes des métriques :")
    numeric_cols = scores_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        print(f"- {col} : {scores_df[col].mean():.4f}")


if __name__ == "__main__":
    main()