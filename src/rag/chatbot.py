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
        self.llm = ChatMistralAI(model="mistral-small-latest", temperature=0.1, mistral_api_key=api_key)

           # 4. Définition du Prompt
        template = """
Tu es l'assistant culturel virtuel de Puls-Events.
Aujourd'hui, nous sommes le {date_du_jour}.

Réponds à la question de l'utilisateur UNIQUEMENT en te basant sur le contexte fourni ci-dessous. Ne l'invente jamais.

RÈGLES DE RÉPONSE OBLIGATOIRES :
1. CORRESPONDANCE PARFAITE : Si le contexte contient un événement dont la vraie date correspond à la demande, réponds avec enthousiasme en le proposant directement (sans dire qu'il n'y a pas d'événement). Ignore les dates contradictoires dans le titre de l'événement, seule la vraie date compte.
2. ALTERNATIVES : Uniquement si AUCUN événement du contexte ne correspond à la date demandée, dis-le clairement, puis propose ce que tu as comme alternative.
3. CONTEXTE VIDE : Si le contexte indique "Aucun événement futur", dis poliment que tu n'as rien à proposer.

FORMAT DE PRÉSENTATION (pour les cas 1 et 2) :
- Utilise une liste à puces (-) pour présenter les événements.
- Mets le titre en gras.
- Précise toujours la ville et la vraie date de l'événement.
- Ajoute quelques émojis pertinents (😊, 🚀, ✨).

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
        et retourne la réponse générée ainsi que les documents sources.
        """
        current_date = datetime.now().date()
        formatted_current_date = current_date.strftime("%A %d %B %Y")
        
        # 1. Récupération large (100 documents)
        large_retriever = self.vector_store.as_retriever(search_kwargs={"k": 100})
        raw_docs = large_retriever.invoke(question)
        
        # 2. Filtrage temporel strict en Python (élimine le passé)
        filtered_docs = []
        for doc in raw_docs:
            date_start_str = doc.metadata.get('date_start', '')
            if not date_start_str:
                continue
                
            try:
                event_date = dateutil.parser.isoparse(date_start_str).date()
                # On ne garde que le présent et le futur
                if event_date >= current_date:
                    filtered_docs.append(doc)
            except Exception:
                pass
                
        # 3. On sélectionne les 3 meilleurs événements FUTURS
        final_docs = filtered_docs[:3]
        
        # Si aucun document futur n'est trouvé, on passe un contexte vide
        context_text = self._format_docs(final_docs) if final_docs else "Aucun événement futur disponible dans la base."
        
        # 4. On passe manuellement nos documents filtrés à la chaîne LLM simple
        answer = self.qa_chain.invoke({
            "context": context_text, 
            "question": question,
            "date_du_jour": formatted_current_date
        })

        return {
            "question": question,
            "answer": answer,
            "sources": final_docs
        }