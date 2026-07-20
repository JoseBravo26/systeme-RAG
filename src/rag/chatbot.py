import os
from datetime import datetime
import dateutil.parser
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PulsEventsChatbot:
    def __init__(self, index_path: str = "data/faiss_index"):
        """Initialise le modèle d'embedding, l'index FAISS et le LLM Mistral."""
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("❌ MISTRAL_API_KEY manquante.")

        print("🧠 Chargement des modèles et de l'index FAISS...")

        # 1. Modèle d'embedding
        self.embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

        # 2. Base vectorielle FAISS
        self.vector_store = FAISS.load_local(
            index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. LLM Mistral
        self.llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0, mistral_api_key=api_key)

        # 4. Définition du Prompt
        template = """
Tu es l'assistant culturel virtuel de Puls-Events.
Aujourd'hui, nous sommes le {date_du_jour}.

Tu dois répondre comme un vrai conseiller culturel : chaleureux, naturel, clair, utile et un peu enthousiaste.
Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. N'invente jamais d'événement, de date ou de lieu.

RÈGLES DE RÉPONSE OBLIGATOIRES :

1. CORRESPONDANCE AVEC LA PÉRIODE DEMANDÉE :
   - Si le contexte contient un ou plusieurs événements dont la vraie date correspond à la période demandée
     (par exemple « ce mois-ci », « en septembre », « cette semaine »), propose UNIQUEMENT ces événements.
   - Ne propose pas d'événements hors de cette période tant qu'il existe au moins un événement dans la période demandée.
   - Ignore toute date contradictoire dans le titre : seule la vraie date (metadata) compte.

2. ALTERNATIVES :
   - Seulement si AUCUN événement du contexte ne correspond à la période demandée, dis-le clairement.
   - Dans ce cas, propose des alternatives d'une autre période en précisant explicitement le mois et l'année.
   - Ne présente jamais un événement hors période comme s'il répondait directement à la demande initiale.

3. CONTEXTE VIDE :
   - Si le contexte est « Aucun événement futur disponible dans la base. », réponds poliment que tu n'as rien trouvé.
   - Invite éventuellement l'utilisateur à élargir sa recherche.

STYLE DE RÉPONSE :
- Commence par une courte phrase d'accueil naturelle.
- La réponse doit sembler humaine, fluide et personnalisée.
- N'utilise pas un ton télégraphique ou trop sec.
- Pour chaque événement, écris au minimum 2 phrases :
  - 1 phrase qui présente clairement l'événement.
  - 1 phrase qui donne une raison concrète d'y aller, en évoquant l'ambiance, le style, l'intérêt culturel ou l'originalité.
- Si plusieurs événements sont trouvés, termine par une courte phrase qui aide l'utilisateur à choisir.
- Si la demande porte sur des concerts, mets davantage l'accent sur l'ambiance musicale, le style et l'énergie.
- Si la demande porte sur une exposition ou une visite, mets davantage l'accent sur l'expérience culturelle et la découverte.

FORMAT DE PRÉSENTATION :
- Utilise une liste à puces (-).
- Mets le titre en gras.
- Pour chaque événement, indique toujours :
  - la ville,
  - la vraie date (jour, mois, année),
  - l’horaire si disponible,
  - une description naturelle et agréable à lire.
- Ajoute quelques émojis pertinents, sans en abuser.

LONGUEUR ATTENDUE :
- Si tu trouves des événements pertinents, la réponse doit être développée.
- Évite les réponses trop courtes ou purement factuelles.
- Ne te contente jamais de lister les événements : commente-les et rends la recommandation vivante.

Contexte :
{context}

Question de l'utilisateur : {question}

Réponse :
"""

        self.prompt = ChatPromptTemplate.from_template(template)

        # 5. Chaîne LLM simple (sans le retriever automatique !)
        self.qa_chain = self.prompt | self.llm | StrOutputParser()
        
        print("✅ Chatbot RAG prêt à l'emploi !")

    def _format_docs(self, docs):
        """Formate les documents récupérés pour les injecter dans le prompt."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask(self, question: str) -> dict:
        """
        Pose une question au chatbot, filtre les événements passés,
        et applique la règle métier : 
        - si la question demande « ce mois-ci », on privilégie strictement le mois courant
        - sinon, on prend juste le futur.
        """
        current_date = datetime.now().date()
        formatted_current_date = current_date.strftime("%A %d %B %Y")

        # 1. Récupération large (100 documents)
        large_retriever = self.vector_store.as_retriever(search_kwargs={"k": 100})
        raw_docs = large_retriever.invoke(question)

        # 2. Filtrage de base : uniquement présent et futur
        future_docs = []
        for doc in raw_docs:
            date_start_str = doc.metadata.get("date_start", "")
            if not date_start_str:
                continue

            try:
                event_date = dateutil.parser.isoparse(date_start_str).date()
                if event_date >= current_date:
                    future_docs.append(doc)
            except Exception:
                # On ignore les dates mal formées
                continue

        # 3. Règle métier « ce mois-ci »
        question_lower = question.lower()
        docs_for_context = []

        if "ce mois-ci" in question_lower:
            # On sépare les événements du mois courant et des autres mois
            same_month_docs = []
            other_month_docs = []

            for doc in future_docs:
                date_start_str = doc.metadata.get("date_start", "")
                try:
                    event_date = dateutil.parser.isoparse(date_start_str).date()
                except Exception:
                    continue

                if event_date.month == current_date.month and event_date.year == current_date.year:
                    same_month_docs.append(doc)
                else:
                    other_month_docs.append(doc)

            if same_month_docs:
                # ✅ Il y a des événements dans le mois demandé → on ne donne que ceux-là au LLM
                docs_for_context = same_month_docs[:3]
            else:
                # ❌ Aucun événement dans le mois demandé → on autorise les alternatives (autres mois)
                docs_for_context = other_month_docs[:3]
        else:
            # Cas général : pas de contrainte « ce mois-ci », on garde simplement les événements futurs
            docs_for_context = future_docs[:3]

        # 4. Construction du contexte pour le LLM
        if docs_for_context:
            context_text = self._format_docs(docs_for_context)
        else:
            context_text = "Aucun événement futur disponible dans la base."

        # 5. Appel LLM avec contexte filtré
        answer = self.qa_chain.invoke({
            "context": context_text,
            "question": question,
            "date_du_jour": formatted_current_date,
        })

        return {
            "question": question,
            "answer": answer,
            "sources": docs_for_context,
        }