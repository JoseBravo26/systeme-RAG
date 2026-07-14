import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

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
            allow_dangerous_deserialization=True # Requis par LangChain pour charger un index local
        )
        
        # Le retriever récupère les 3 documents les plus pertinents
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 3. LLM Mistral pour la génération de texte (on utilise le modèle 'small' qui est un bon compromis)
        self.llm = ChatMistralAI(model="mistral-small-latest", temperature=0.1, mistral_api_key=api_key)
 
        # 4. Définition du Prompt (ajout de la variable {date_du_jour})
        template = """
Tu es l'assistant culturel virtuel de Puls-Events.
Aujourd'hui, nous sommes le {date_du_jour}.

Réponds à la question de l'utilisateur uniquement en te basant sur le contexte fourni ci-dessous.
Pour les questions concernant "aujourd'hui", "demain" ou "ce week-end", utilise la date d'aujourd'hui pour évaluer si les événements du contexte correspondent.
Si l'information n'est pas dans le contexte, ou si les dates des événements ne correspondent pas à la demande, réponds poliment que tu ne trouves pas d'événement, sans inventer d'informations.
Mentionne toujours le titre, le lieu (ville) et les dates dans ta recommandation.

Contexte :
{context}

Question de l'utilisateur : {question}

Réponse claire et enthousiaste (en français) :
"""
        # On définit les variables que le prompt attend (context, question, date_du_jour)
        self.prompt = ChatPromptTemplate.from_template(template)

        # 5. Construction de la chaîne LCEL
        # On ajoute RunnablePassthrough() pour laisser passer la date_du_jour vers le prompt
        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(self.retriever.invoke(x["question"])), 
                "question": lambda x: x["question"],
                "date_du_jour": lambda x: x["date_du_jour"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ Chatbot RAG prêt à l'emploi !")

    def _format_docs(self, docs):
        """Formate les documents récupérés pour les injecter dans le prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> dict:
        """
        Pose une question au chatbot et retourne la réponse générée ainsi que les documents sources.
        """
        # Obtenir la date actuelle formatée
        current_date = datetime.now().strftime("%A %d %B %Y")
        
        # Exécution de la chaîne en passant un dictionnaire avec la question ET la date
        answer = self.rag_chain.invoke({
            "question": question,
            "date_du_jour": current_date
        })

        # Récupération séparée des sources pour la traçabilité
        source_docs = self.retriever.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "sources": source_docs
        }

if __name__ == "__main__":
    # Test simple en ligne de commande
    bot = PulsEventsChatbot()
    print("\n" + "="*50)
    
    query = "Je cherche un concert de jazz à Paris."
    print(f"👤 Utilisateur : {query}\n")
    
    result = bot.ask(query)
    
    print(f"🤖 Puls-Events Bot :\n{result['answer']}\n")
    print("-" * 50)
    print("📚 Sources utilisées :")
    for doc in result['sources']:
        print(f"- {doc.metadata.get('title')} ({doc.metadata.get('city')})")