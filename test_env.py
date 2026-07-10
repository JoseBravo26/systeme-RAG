# test_env.py

import os
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

def test_imports():
    print("Vérification des imports en cours...")
    
    try:
        import faiss
        print("✅ FAISS importé avec succès.")
        
        from langchain_community.vectorstores import FAISS as LC_FAISS
        print("✅ LangChain FAISS importé avec succès.")
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ HuggingFaceEmbeddings importé avec succès.")
        
        from mistralai import Mistral
        print("✅ MistralClient importé avec succès.")
        
        # Vérification de la variable d'environnement
        if os.getenv("MISTRAL_API_KEY"):
            print("✅ Clé API Mistral trouvée dans l'environnement.")
        else:
            print("⚠️ Attention : MISTRAL_API_KEY n'est pas définie dans le fichier .env.")
            
    except ImportError as e:
        print(f"❌ Erreur d'importation : {e}")

if __name__ == "__main__":
    test_imports()