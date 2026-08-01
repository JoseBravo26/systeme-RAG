"""
API REST du système RAG Puls-Events.

Endpoints :
- GET /health : vérifie l'état de l'API et du chatbot ;
- POST /ask : interroge le système RAG ;
- POST /rebuild : reconstruit l'index FAISS depuis le CSV local autorisé.

La logique métier reste dans src/rag/chatbot.py.
Cette API se limite à valider les requêtes HTTP, appeler la logique métier
et structurer les réponses.
"""

import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

# Ajout de la racine du projet au PYTHONPATH lorsque ce fichier est dans api/.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.indexing.faiss_indexer import (  # noqa: E402
    build_faiss_index,
    create_documents_from_csv,
    split_documents,
)
from src.rag.chatbot import PulsEventsChatbot  # noqa: E402

load_dotenv()

# Les chemins sont imposés côté serveur : le client ne peut pas les modifier.
CSV_PATH = os.getenv("EVENTS_CSV_PATH", "data/evenements_clean.csv")
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

# Clé facultative mais recommandée, surtout si l'API est accessible sur un réseau.
REBUILD_API_KEY = os.getenv("REBUILD_API_KEY")

chatbot: Optional[PulsEventsChatbot] = None
rebuild_lock = threading.Lock()
is_rebuilding = False


class ChatRequest(BaseModel):
    """Corps d'une requête POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1_000,
        description="Question de l'utilisateur sur les événements culturels.",
        examples=["Quels concerts de jazz sont proposés à Paris le 24 juillet 2026 ?"],
    )


class SourceDoc(BaseModel):
    """Source utilisée pour générer une réponse."""

    title: str
    city: str


class ChatResponse(BaseModel):
    """Réponse retournée par POST /ask."""

    question: str
    answer: str
    sources: List[SourceDoc]


class RebuildResponse(BaseModel):
    """Résultat de la reconstruction de l'index FAISS."""

    status: str
    message: str
    nb_documents: int
    nb_chunks: int


class HealthResponse(BaseModel):
    """État de disponibilité de l'API."""

    status: str
    rebuilding: bool


def load_chatbot() -> PulsEventsChatbot:
    """
    Charge le chatbot avec l'index FAISS local configuré côté serveur.
    """
    return PulsEventsChatbot(index_path=INDEX_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Charge le chatbot une seule fois au démarrage de l'API.
    """
    global chatbot

    try:
        print("🚀 Chargement initial du chatbot...")
        chatbot = load_chatbot()
        print("✅ Chatbot prêt.")
    except Exception as exc:
        print(f"❌ Impossible de charger le chatbot au démarrage : {exc}")
        chatbot = None

    yield

    chatbot = None


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API locale de recommandation d'événements culturels, "
        "basée sur OpenAgenda, FAISS, LangChain et Mistral."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


def check_rebuild_key(x_rebuild_api_key: Optional[str]) -> None:
    """
    Vérifie la clé dédiée à la reconstruction si elle est configurée.

    En l'absence de REBUILD_API_KEY dans .env, l'endpoint reste disponible
    uniquement pour faciliter les démonstrations locales.
    """
    if REBUILD_API_KEY and x_rebuild_api_key != REBUILD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé de reconstruction invalide ou absente.",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état de l'API",
)
def health_check() -> HealthResponse:
    """
    Indique si le chatbot est chargé et si une reconstruction est en cours.
    """
    api_status = "ready" if chatbot is not None and not is_rebuilding else "not_ready"

    return HealthResponse(
        status=api_status,
        rebuilding=is_rebuilding,
    )


@app.post(
    "/ask",
    response_model=ChatResponse,
    summary="Poser une question au chatbot RAG",
)
def ask_question(request: ChatRequest) -> ChatResponse:
    """
    Génère une réponse fondée exclusivement sur les événements récupérés
    par le système RAG.
    """
    if is_rebuilding:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base vectorielle est en cours de reconstruction.",
        )

    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le système RAG n'est pas initialisé.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La question ne peut pas être vide.",
        )

    try:
        result = chatbot.ask(question)

        sources = [
            SourceDoc(
                title=str(doc.metadata.get("title", "Titre inconnu")),
                city=str(doc.metadata.get("city", "Ville inconnue")),
            )
            for doc in result.get("sources", [])
        ]

        return ChatResponse(
            question=question,
            answer=str(result["answer"]),
            sources=sources,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne pendant la génération de la réponse.",
        ) from exc


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire l'index FAISS",
)
def rebuild_index(
    x_rebuild_api_key: Optional[str] = Header(
        default=None,
        description="Clé optionnelle de protection de la reconstruction.",
    ),
) -> RebuildResponse:
    """
    Reconstruit l'index complet à partir du CSV local autorisé, puis recharge
    le chatbot. Aucune route de fichier n'est fournie par le client.
    """
    global chatbot, is_rebuilding

    check_rebuild_key(x_rebuild_api_key)

    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Le fichier CSV configuré est introuvable sur le serveur.",
        )

    if not rebuild_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Une reconstruction est déjà en cours.",
        )

    is_rebuilding = True

    try:
        print("🔄 Reconstruction de l'index FAISS...")
        print(f"📄 CSV source : {CSV_PATH}")

        documents = create_documents_from_csv(CSV_PATH)

        if not documents:
            raise ValueError("Aucun document exploitable trouvé dans le CSV.")

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("Aucun chunk généré pour la reconstruction.")

        # Important : aucun slicing de type chunks[:300] ou chunks[:100].
        build_faiss_index(chunks, save_dir=INDEX_PATH)

        # Le chatbot est remplacé uniquement après succès de l'indexation.
        chatbot = load_chatbot()

        print("✅ Reconstruction terminée avec succès.")

        return RebuildResponse(
            status="success",
            message="Index FAISS complet reconstruit et chatbot rechargé.",
            nb_documents=len(documents),
            nb_chunks=len(chunks),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne pendant la reconstruction de l'index.",
        ) from exc

    finally:
        is_rebuilding = False
        rebuild_lock.release()