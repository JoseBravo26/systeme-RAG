import os
import pandas as pd
from typing import List
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings

load_dotenv()

def create_documents_from_csv(file_path: str) -> List[Document]:
    """
    Charge le CSV nettoyé et convertit chaque ligne en un Document LangChain,
    en conservant les métadonnées utiles pour le RAG.
    """
    print(f"📖 Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path)
    
    documents = []
    for _, row in df.iterrows():
        # Extraction du texte à vectoriser
        text = str(row.get('text_to_embed', ''))
        if not text.strip() or text == 'nan':
            continue
            
        # Sauvegarde des métadonnées pour enrichir la réponse et filtrer plus tard si besoin
        metadata = {
            "uid": row.get('uid'),
            "title": row.get('title_fr'),
            "city": row.get('location_city'),
            "department": row.get('location_department'),
            "date_start": row.get('firstdate_begin'),
            "date_end": row.get('lastdate_end')
        }
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
        
    print(f"✅ {len(documents)} documents bruts créés.")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe les documents en morceaux (chunks) plus petits.
    Cela améliore la précision sémantique de FAISS et respecte la limite de contexte du LLM.
    """
    print("✂️ Découpage des documents (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)} chunks générés à partir des documents originaux.")
    return chunks

def build_faiss_index(chunks: List[Document], save_dir: str = "data/faiss_index"):
    """
    Génère les embeddings via Mistral, construit l'index FAISS et le sauvegarde sur le disque.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("❌ MISTRAL_API_KEY manquante dans le fichier .env")

    print(f"🧠 Initialisation du modèle Mistral Embeddings...")
    embeddings_model = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )
    
    print("🏗️ Construction de l'index vectoriel FAISS (cela peut prendre quelques minutes)...")
    # Création de la base de données FAISS
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    
    # Sauvegarde sur disque pour éviter de repayer/recalculer les embeddings à chaque lancement
    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)
    print(f"💾 Index FAISS sauvegardé avec succès dans '{save_dir}'")
    
    return vector_store

def test_retriever(vector_store, query: str):
    """
    Teste le bon fonctionnement de l'index avec une requête simple.
    """
    print(f"\n🔍 Test de recherche pour la requête : '{query}'")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    
    for i, res in enumerate(results):
        print(f"\n--- Résultat {i+1} ---")
        print(f"Titre: {res.metadata.get('title')}")
        print(f"Lieu: {res.metadata.get('city')}")
        print(f"Extrait: {res.page_content[:150]}...")

if __name__ == "__main__":
    csv_path = "data/evenements_clean.csv"
    
    # 1. Chargement
    docs = create_documents_from_csv(csv_path)
    
    # 2. Chunking
    chunks = split_documents(docs)
    
    # 3. Indexation
    # Pour un POC rapide, tu peux limiter à chunks[:500] si tu ne veux pas consommer trop de crédits Mistral
    db = build_faiss_index(chunks) 
    
    # 4. Test
    test_retriever(db, "Concert de jazz à Paris ce week-end")