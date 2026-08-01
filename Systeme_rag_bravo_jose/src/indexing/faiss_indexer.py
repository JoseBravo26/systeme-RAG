"""
Construction et sauvegarde de l'index vectoriel FAISS
pour le projet Puls-Events.

Ce module :
1. charge les données nettoyées depuis un CSV,
2. transforme chaque ligne en Document LangChain avec métadonnées,
3. découpe les documents en chunks,
4. génère les embeddings via Mistral,
5. construit et sauvegarde l'index FAISS.
"""

import os
from typing import List

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings


load_dotenv()


def create_documents_from_csv(file_path: str) -> List[Document]:
    """
    Charge le CSV nettoyé et convertit chaque ligne en Document LangChain.

    Paramètres
    ----------
    file_path : str
        Chemin vers le fichier CSV nettoyé.

    Retour
    ------
    List[Document]
        Liste de documents LangChain prêts pour le chunking.
    """
    print(f"📖 Chargement des données depuis : {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier CSV est introuvable : {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    if df.empty:
        print("⚠️ Le CSV est vide.")
        return []

    documents = []

    for _, row in df.iterrows():
        text = str(row.get("text_to_embed", "")).strip()

        # On ignore les lignes sans texte exploitable
        if not text or text.lower() == "nan":
            continue

        metadata = {
            "uid": row.get("uid"),
            "title": row.get("title_fr", ""),
            "city": row.get("location_city", ""),
            "department": row.get("location_department", ""),
            "postal_code": row.get("location_postalcode", ""),
            "date_start": row.get("firstdate_begin", ""),
            "date_end": row.get("lastdate_end", "")
        }

        doc = Document(
            page_content=text,
            metadata=metadata
        )
        documents.append(doc)

    print(f"✅ {len(documents)} documents créés à partir du CSV.")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Découpe les documents en chunks pour améliorer la recherche sémantique.

    Paramètres
    ----------
    documents : List[Document]
        Documents LangChain à découper.
    chunk_size : int
        Taille maximale d'un chunk.
    chunk_overlap : int
        Nombre de caractères de chevauchement entre chunks.

    Retour
    ------
    List[Document]
        Liste des chunks générés.
    """
    if not documents:
        print("⚠️ Aucun document à découper.")
        return []

    print("✂️ Découpage des documents en chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ {len(chunks)} chunks générés.")
    return chunks


def build_faiss_index(
    chunks: List[Document],
    save_dir: str = "data/faiss_index"
) -> FAISS:
    """
    Construit l'index FAISS à partir des chunks et le sauvegarde sur disque.

    Paramètres
    ----------
    chunks : List[Document]
        Chunks à vectoriser et indexer.
    save_dir : str
        Dossier de sauvegarde de l'index FAISS.

    Retour
    ------
    FAISS
        Base vectorielle FAISS construite.
    """
    if not chunks:
        raise ValueError("Impossible de construire l'index : aucun chunk fourni.")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est absente du fichier .env.")

    print("🧠 Initialisation du modèle d'embeddings Mistral...")

    embeddings_model = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )

    print("🏗️ Construction de l'index FAISS...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings_model
    )

    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)

    print(f"✅ Index FAISS sauvegardé avec succès dans '{save_dir}'")

    return vector_store


def test_retriever(vector_store: FAISS, query: str, k: int = 3) -> None:
    """
    Effectue une recherche test dans l'index vectoriel.

    Paramètres
    ----------
    vector_store : FAISS
        Base vectorielle FAISS chargée ou fraîchement construite.
    query : str
        Requête utilisateur à tester.
    k : int
        Nombre de résultats à récupérer.
    """
    print(f"\n🔍 Test de recherche : {query}")

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)

    if not results:
        print("⚠️ Aucun résultat trouvé.")
        return

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Résultat {i} ---")
        print(f"Titre : {doc.metadata.get('title', 'Titre inconnu')}")
        print(f"Ville : {doc.metadata.get('city', 'Ville inconnue')}")
        print(f"Date début : {doc.metadata.get('date_start', 'Date inconnue')}")
        print(f"Extrait : {doc.page_content[:250]}...")


if __name__ == "__main__":
    # Chemins par défaut
    csv_path = "data/evenements_clean.csv"
    index_path = "data/faiss_index"

    # 1. Création des documents
    docs = create_documents_from_csv(csv_path)

    # 2. Chunking
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=150)

    # 3. Limitation optionnelle pour tests rapides / économie de crédits
    # Décommente la ligne suivante si besoin :
    #chunks = chunks[:300] #Limite à 300 chunks pour tests rapides si besoin

    # 4. Construction de l'index
    db = build_faiss_index(chunks, save_dir=index_path)

    # 5. Test rapide
    test_retriever(db, "concert de jazz à Paris")