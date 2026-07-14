"""
API REST pour exposer le système RAG Puls-Events.

Routes principales :
- GET  /health   : vérification de l'état de l'API
- POST /ask      : poser une question au chatbot RAG
- POST /rebuild  : reconstruire l'index FAISS à partir du CSV
"""

import os
import sys
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Permet à Python de trouver le dossier 'src' quand on lance `uvicorn api.main:app`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports de la logique métier
from src.rag.chatbot import PulsEventsChatbot
from src.indexing.faiss_indexer import (
    create_documents_from_csv,
    split_documents,
    build_faiss_index,
)

load_dotenv()

# ---------------------------
# Modèles Pydantic (schéma I/O)
# ---------------------------

class ChatRequest(BaseModel):
    """
    Modèle de requête pour l'endpoint /ask.
    """
    question: str


class SourceDoc(BaseModel):
    """
    Représentation simplifiée d’un document source renvoyé au client.
    """
    title: str
    city: str


class ChatResponse(BaseModel):
    """
    Réponse renvoyée au client : question, réponse générée, et sources.
    """
    question: str
    answer: str
    sources: List[SourceDoc]


class RebuildResponse(BaseModel):
    """
    Réponse renvoyée après reconstruction de l’index.
    """
    message: str
    nb_documents: int
    nb_chunks: int


# ---------------------------
# Initialisation FastAPI
# ---------------------------

app = FastAPI(
    title="Puls-Events RAG API",
    description="API de recommandation d'événements culturels basée sur Mistral, FAISS et LangChain.",
    version="1.0.0",
)

# Instance globale du chatbot (chargée au démarrage)
chatbot: PulsEventsChatbot | None = None


# ---------------------------
# Hooks de cycle de vie
# ---------------------------

@app.on_event("startup")
def startup_event():
    """
    Au démarrage du serveur :
    - On charge l’index FAISS déjà construit
    - On initialise le chatbot RAG
    """
    global chatbot
    try:
        print("🚀 Démarrage de l'API et chargement du système RAG...")
        chatbot = PulsEventsChatbot(index_path="data/faiss_index")
        print("✅ Chatbot initialisé avec succès.")
    except Exception as e:
        # On log l'erreur, mais on laisse l'API démarrer (le /health pourra le refléter)
        print(f"❌ Erreur lors du chargement du Chatbot : {e}")
        chatbot = None


# ---------------------------
# Routes publiques
# ---------------------------

@app.get("/health")
def health_check():
    """
    Endpoint de santé basique.
    Permet de vérifier si l’API tourne et si le chatbot est prêt.
    """
    status = "ready" if chatbot is not None else "not_ready"
    return {"status": status}


@app.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """
    Endpoint principal : pose une question au système RAG.

    - Input  : JSON {"question": "..."}
    - Output : JSON {"question": "...", "answer": "...", "sources": [...]}
    """
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Le système RAG n'est pas initialisé.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        # On interroge le chatbot (LangChain + FAISS + Mistral)
        result = chatbot.ask(question)

        # Formatage des sources pour la réponse JSON
        formatted_sources = []
        for doc in result["sources"]:
            formatted_sources.append(
                SourceDoc(
                    title=doc.metadata.get("title", "Titre inconnu"),
                    city=doc.metadata.get("city", "Ville inconnue"),
                )
            )

        return ChatResponse(
            question=question,
            answer=result["answer"],
            sources=formatted_sources,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse : {e}")


@app.post("/rebuild", response_model=RebuildResponse)
def rebuild_index():
    """
    Reconstruit l’index FAISS à partir du fichier CSV des événements.

    ⚠ À sécuriser si l’API est un jour exposée publiquement.
    Pour le POC, on le laisse accessible en local.
    """
    global chatbot

    csv_path = "data/evenements_clean.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail=f"Fichier CSV introuvable à {csv_path}")

    try:
        # 1. Création des documents à partir du CSV
        docs = create_documents_from_csv(csv_path)

        # 2. Chunking
        chunks = split_documents(docs)

        # 3. Reconstruction complète de l’index FAISS sur disque
        build_faiss_index(chunks, save_dir="data/faiss_index")

        # 4. Réinitialisation du chatbot avec le nouvel index
        chatbot = PulsEventsChatbot(index_path="data/faiss_index")

        return RebuildResponse(
            message="Index FAISS reconstruit et chatbot réinitialisé avec succès.",
            nb_documents=len(docs),
            nb_chunks=len(chunks),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la reconstruction de l'index : {e}")