"""
API REST du système RAG Puls-Events.

Endpoints :
- GET  /health   : état de l'API
- POST /ask      : interroger le chatbot
- POST /rebuild  : reconstruire l'index FAISS à partir du CSV nettoyé
"""

import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Ajout du dossier racine au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.chatbot import PulsEventsChatbot
from src.indexing.faiss_indexer import (
    create_documents_from_csv,
    split_documents,
    build_faiss_index,
)

load_dotenv()

app = FastAPI(
    title="Puls-Events RAG API",
    description="API de recommandation d'événements culturels basée sur LangChain, Mistral et FAISS.",
    version="1.1.0"
)

chatbot: Optional[PulsEventsChatbot] = None


# =========================
# Modèles Pydantic
# =========================

class ChatRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    title: str
    city: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]


class RebuildRequest(BaseModel):
    csv_path: str = "data/evenements_clean.csv"
    index_path: str = "data/faiss_index"
    max_chunks: Optional[int] = None


class RebuildResponse(BaseModel):
    status: str
    message: str
    nb_documents: int
    nb_chunks: int
    csv_path: str
    index_path: str


# =========================
# Fonctions utilitaires
# =========================

def load_chatbot(index_path: str = "data/faiss_index") -> PulsEventsChatbot:
    """
    Charge un chatbot à partir d'un index FAISS déjà présent sur disque.
    """
    return PulsEventsChatbot(index_path=index_path)


# =========================
# Démarrage
# =========================

@app.on_event("startup")
def startup_event():
    """
    Chargement initial du chatbot au démarrage du serveur.
    """
    global chatbot

    try:
        print("🚀 Chargement initial du chatbot...")
        chatbot = load_chatbot("data/faiss_index")
        print("✅ Chatbot prêt.")
    except Exception as e:
        print(f"❌ Impossible de charger le chatbot au démarrage : {e}")
        chatbot = None


# =========================
# Endpoints
# =========================

@app.get("/health")
def health_check():
    """
    Vérifie si l'API et le chatbot sont opérationnels.
    """
    return {
        "status": "ready" if chatbot is not None else "not_ready"
    }


@app.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """
    Pose une question au système RAG.
    """
    global chatbot

    if chatbot is None:
        raise HTTPException(status_code=500, detail="Le système RAG n'est pas initialisé.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        result = chatbot.ask(question)

        sources = [
            SourceDoc(
                title=doc.metadata.get("title", "Titre inconnu"),
                city=doc.metadata.get("city", "Ville inconnue")
            )
            for doc in result["sources"]
        ]

        return ChatResponse(
            question=question,
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {e}")


@app.post("/rebuild", response_model=RebuildResponse)
def rebuild_index(request: RebuildRequest):
    """
    Reconstruit complètement l'index FAISS à partir du CSV nettoyé,
    puis recharge le chatbot en mémoire.
    """
    global chatbot

    csv_path = request.csv_path
    index_path = request.index_path
    max_chunks = request.max_chunks

    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"Fichier CSV introuvable : {csv_path}"
        )

    try:
        print("🔄 Reconstruction de l'index FAISS...")
        print(f"📄 CSV source : {csv_path}")

        # 1. Création des documents
        docs = create_documents_from_csv(csv_path)
        nb_documents = len(docs)

        if nb_documents == 0:
            raise ValueError("Aucun document exploitable trouvé dans le CSV.")

        # 2. Chunking
        chunks = split_documents(docs)

        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        nb_chunks = len(chunks)

        if nb_chunks == 0:
            raise ValueError("Aucun chunk généré pour la reconstruction.")

        # 3. Reconstruction de l'index vectoriel
        build_faiss_index(chunks, save_dir=index_path)

        # 4. Rechargement du chatbot avec le nouvel index
        chatbot = load_chatbot(index_path=index_path)

        print("✅ Reconstruction terminée avec succès.")

        return RebuildResponse(
            status="success",
            message="Index FAISS reconstruit et chatbot rechargé avec succès.",
            nb_documents=nb_documents,
            nb_chunks=nb_chunks,
            csv_path=csv_path,
            index_path=index_path
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la reconstruction de l'index : {e}"
        )