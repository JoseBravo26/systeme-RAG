"""
Chatbot RAG Puls-Events.

Ce module :
- charge l'index FAISS construit à partir des événements OpenAgenda ;
- effectue une recherche vectorielle FAISS ;
- applique des filtres explicites sur la zone, la période et le thème ;
- utilise un parcours de secours du docstore lorsque FAISS ne remonte pas
  de document conforme aux contraintes ;
- génère une réponse strictement fondée sur les documents récupérés.
"""

import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import dateutil.parser
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class PulsEventsChatbot:
    """
    Chatbot RAG pour recommander des événements culturels à partir
    d'un index vectoriel FAISS.
    """

    EMPTY_CONTEXT = (
        "Aucun événement futur correspondant à la demande n'est disponible "
        "dans la base de données."
    )

    MONTHS = {
        "janvier": 1,
        "février": 2,
        "fevrier": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "aout": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12,
        "decembre": 12,
    }

    # Les synonymes améliorent le filtrage thématique après la recherche FAISS.
    TOPIC_KEYWORDS = {
        "cinema": [
            "cinéma",
            "cinema",
            "ciné",
            "cine",
            "film",
            "films",
            "projection",
            "écran",
            "ecran",
        ],
        "jazz": [
            "jazz",
            "ciné-jazz",
            "cine-jazz",
            "jazz manouche",
            "jam jazz",
        ],
        "patrimoine": [
            "patrimoine",
            "historique",
            "histoire",
            "archives",
            "monument",
            "musée",
            "musee",
        ],
        "astronomie": [
            "astronomie",
            "éclipse",
            "eclipse",
            "soleil",
            "étoile",
            "etoile",
            "nuit des étoiles",
            "nuit des etoiles",
        ],
        "concert": [
            "concert",
            "musique",
            "musical",
            "live",
        ],
        "visite": [
            "visite",
            "visites",
            "visite guidée",
            "visite guidee",
            "coulisses",
        ],
    }

    def __init__(self, index_path: str = "data/faiss_index") -> None:
        """
        Initialise les modèles Mistral, l'index FAISS et la chaîne RAG.
        """
        self.index_path = index_path
        self._load_models_and_index()
        self._build_chain()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_models_and_index(self) -> None:
        """
        Charge le modèle génératif, les embeddings et l'index FAISS.
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        chat_model = os.getenv("MISTRAL_CHAT_MODEL", "open-mistral-nemo")
        embed_model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

        if not api_key:
            raise ValueError(
                "La variable d'environnement MISTRAL_API_KEY est manquante."
            )

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=chat_model,
            temperature=0.1,
            max_tokens=512,
        )

        # Compatible avec l'API OpenAI-compatible de Mistral.
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=embed_model,
            check_embedding_ctx_length=False,
        )

        index_dir = os.path.abspath(self.index_path)

        if not os.path.isdir(index_dir):
            raise FileNotFoundError(
                f"Répertoire d'index FAISS introuvable : {index_dir}.\n"
                "Assure-toi d'avoir exécuté le script d'indexation."
            )

        self.vector_store = FAISS.load_local(
            index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _build_chain(self) -> None:
        """
        Construit une chaîne de réponse avec garde-fous anti-hallucination.
        """
        template = """
Tu es l'assistant culturel virtuel de Puls-Events.
Aujourd'hui, nous sommes le {date_du_jour}.

Ta mission est de répondre de manière claire, chaleureuse et utile,
UNIQUEMENT à partir du contexte fourni.

RÈGLES ABSOLUES :
1. Toute information factuelle dans ta réponse doit être présente explicitement
dans le contexte : nom d'événement, ville, date, horaire et description.

2. Ne cite, n'invente, ne suggère et ne décris JAMAIS un événement absent
du contexte. N'utilise aucune connaissance externe.

3. Si le contexte indique qu'aucun événement n'a été trouvé, réponds :
"Je n'ai pas trouvé d'événement correspondant dans la base de données
pour cette demande."
Tu peux inviter l'utilisateur à élargir sa recherche à une autre date,
une autre ville ou un autre thème, sans citer d'événement précis.

4. Si des événements sont présents dans le contexte, ne propose que ceux-ci.
Respecte leurs dates et leurs villes réelles.

5. N'affirme une caractéristique pratique ou un public visé que si cette
information apparaît clairement dans le contexte.

FORMAT :
- Commence par une phrase courte et naturelle.
- Utilise une liste à puces si des événements sont disponibles.
- Mets le titre de chaque événement en gras.
- Indique la ville et la date si elles sont disponibles.
- Reste concis et ne répète pas le contexte mot pour mot.

Contexte :
{context}

Question de l'utilisateur :
{question}

Réponse :
"""

        prompt = ChatPromptTemplate.from_template(template)
        self.qa_chain = prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Analyse de la question
    # ------------------------------------------------------------------

    def _extract_constraints(self, question: str) -> Dict[str, Any]:
        """
        Extrait les contraintes de zone et de période de la question.
        """
        normalized_question = question.lower()
        today = datetime.now().date()

        city: Optional[str] = None
        department: Optional[str] = None

        if "paris" in normalized_question:
            city = "Paris"

        if (
            re.search(r"\b93\b", normalized_question)
            or "seine-saint-denis" in normalized_question
            or "seine saint denis" in normalized_question
        ):
            department = "Seine-Saint-Denis"

        start_date: date = today
        end_date: Optional[date] = None

        # Cas : "ce week-end".
        if "ce week-end" in normalized_question or "ce week end" in normalized_question:
            days_until_saturday = (5 - today.weekday()) % 7
            start_date = today + timedelta(days=days_until_saturday)
            end_date = start_date + timedelta(days=1)

        # Cas : "le 24 juillet 2026" ou "1er août 2026".
        date_pattern = (
            r"\b(?:le\s+)?(\d{1,2}|1er)\s+("
            + "|".join(self.MONTHS.keys())
            + r")\s+(20\d{2})\b"
        )
        exact_match = re.search(date_pattern, normalized_question)

        if exact_match:
            day_as_text, month_name, year_as_text = exact_match.groups()
            day = 1 if day_as_text == "1er" else int(day_as_text)
            month = self.MONTHS[month_name]
            year = int(year_as_text)

            try:
                start_date = datetime(year, month, day).date()
                end_date = start_date
            except ValueError:
                pass

        # Cas : "en août", "en septembre".
        elif True:
            for month_name, month_number in self.MONTHS.items():
                if f"en {month_name}" in normalized_question:
                    year = today.year

                    if month_number < today.month:
                        year += 1

                    start_date = datetime(year, month_number, 1).date()

                    if month_number == 12:
                        end_date = (
                            datetime(year + 1, 1, 1).date() - timedelta(days=1)
                        )
                    else:
                        end_date = (
                            datetime(year, month_number + 1, 1).date()
                            - timedelta(days=1)
                        )
                    break

        return {
            "city": city,
            "department": department,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _requested_topics(self, question: str) -> List[str]:
        """
        Détecte les thèmes explicitement demandés dans une question.
        """
        normalized_question = question.lower()
        requested_topics: List[str] = []

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(keyword in normalized_question for keyword in keywords):
                requested_topics.append(topic)

        return requested_topics

    def _document_searchable_text(self, doc: Any) -> str:
        """
        Crée un texte exploitable à partir du chunk et de ses métadonnées.
        """
        metadata = getattr(doc, "metadata", {}) or {}

        values = [
            getattr(doc, "page_content", ""),
            metadata.get("title", ""),
            metadata.get("description", ""),
            metadata.get("keywords", ""),
        ]

        return " ".join(str(value) for value in values if value).lower()

    def _matches_topic(self, doc: Any, question: str) -> bool:
        """
        Vérifie qu'un document correspond à tous les thèmes demandés.

        Exemple : une requête "cinéma et jazz" doit contenir au moins un
        mot-clé de cinéma ET un mot-clé de jazz.
        """
        requested_topics = self._requested_topics(question)

        if not requested_topics:
            return True

        document_text = self._document_searchable_text(doc)

        for topic in requested_topics:
            keywords = self.TOPIC_KEYWORDS[topic]

            if not any(keyword in document_text for keyword in keywords):
                return False

        return True

    # ------------------------------------------------------------------
    # Recherche, filtrage et contexte
    # ------------------------------------------------------------------

    def _get_all_index_documents(self) -> List[Any]:
        """
        Retourne tous les chunks stockés dans le docstore FAISS.

        Cette méthode n'est utilisée qu'en solution de repli lorsque la
        recherche vectorielle ne renvoie aucun document conforme.
        """
        docstore = getattr(self.vector_store, "docstore", None)

        if docstore is None:
            return []

        documents = getattr(docstore, "_dict", {})
        return list(documents.values())

    def _filter_documents(
        self,
        documents: List[Any],
        question: str,
        current_date: date,
        city_filter: Optional[str],
        department_filter: Optional[str],
        start_filter: Optional[date],
        end_filter: Optional[date],
    ) -> List[Any]:
        """
        Applique les règles métier aux documents candidats.

        Les filtres portent sur la ville, le département, le thème,
        le caractère futur et le chevauchement avec la période demandée.
        """
        filtered_docs: List[Any] = []
        seen_uids = set()

        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}

            date_start_raw = str(metadata.get("date_start", "")).strip()
            date_end_raw = str(metadata.get("date_end", "")).strip() or date_start_raw

            city = str(metadata.get("city", "")).strip()
            department = str(metadata.get("department", "")).strip()
            uid = metadata.get("uid")

            if not date_start_raw:
                continue

            if city_filter and city.lower() != city_filter.lower():
                continue

            if (
                department_filter
                and department_filter.lower() not in department.lower()
            ):
                continue

            if not self._matches_topic(doc, question):
                continue

            try:
                event_start = dateutil.parser.isoparse(date_start_raw).date()
                event_end = dateutil.parser.isoparse(date_end_raw).date()
            except (TypeError, ValueError, OverflowError):
                continue

            if event_end < current_date:
                continue

            # L'événement doit chevaucher la période demandée.
            if start_filter and event_end < start_filter:
                continue

            if end_filter and event_start > end_filter:
                continue

            # Un événement peut produire plusieurs chunks : on évite les doublons.
            unique_key = uid if uid is not None else (
                metadata.get("title", ""),
                date_start_raw,
                city,
            )

            if unique_key in seen_uids:
                continue

            seen_uids.add(unique_key)
            filtered_docs.append(doc)

        return filtered_docs

    def _format_docs(self, docs: List[Any]) -> str:
        """
        Formate les documents retenus pour le LLM.

        Le texte indexé est inclus afin que le modèle puisse s'appuyer sur les
        descriptions, les thèmes et les informations réellement disponibles.
        """
        if not docs:
            return self.EMPTY_CONTEXT

        formatted_docs: List[str] = []

        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}

            title = str(metadata.get("title", "Titre inconnu"))
            city = str(metadata.get("city", "Ville inconnue"))
            department = str(metadata.get("department", ""))
            date_start = str(metadata.get("date_start", ""))
            date_end = str(metadata.get("date_end", ""))

            if date_start and date_end and date_end != date_start:
                date_part = f"du {date_start} au {date_end}"
            else:
                date_part = date_start or "Date non renseignée"

            content = str(getattr(doc, "page_content", "")).strip()

            formatted_docs.append(
                f"""ÉVÉNEMENT DISPONIBLE DANS LA BASE :
Titre : {title}
Ville : {city}
Département : {department}
Date : {date_part}
Contenu indexé :
{content}"""
            )

        return "\n\n---\n\n".join(formatted_docs)

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Répond à une question utilisateur à partir des événements indexés.
        """
        current_date = datetime.now().date()
        formatted_current_date = current_date.strftime("%A %d %B %Y")

        constraints = self._extract_constraints(question)

        city_filter = constraints["city"]
        department_filter = constraints["department"]
        start_filter = constraints["start_date"]
        end_filter = constraints["end_date"]

        # Première étape : recherche sémantique vectorielle.
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 300})
        raw_docs = retriever.invoke(question)

        filtered_docs = self._filter_documents(
            documents=raw_docs,
            question=question,
            current_date=current_date,
            city_filter=city_filter,
            department_filter=department_filter,
            start_filter=start_filter,
            end_filter=end_filter,
        )

        # Seconde étape : fallback déterministe si FAISS ne contient aucun
        # candidat conforme. Utile pour les dates exactes et thèmes rares.
        if not filtered_docs:
            all_documents = self._get_all_index_documents()

            filtered_docs = self._filter_documents(
                documents=all_documents,
                question=question,
                current_date=current_date,
                city_filter=city_filter,
                department_filter=department_filter,
                start_filter=start_filter,
                end_filter=end_filter,
            )

        # Trois événements maximum sont envoyés au LLM.
        final_docs = filtered_docs[:3]
        context_text = self._format_docs(final_docs)

        answer = self.qa_chain.invoke(
            {
                "context": context_text,
                "question": question,
                "date_du_jour": formatted_current_date,
            }
        )

        return {
            "question": question,
            "answer": answer,
            "sources": final_docs,
            "context": context_text,
        }