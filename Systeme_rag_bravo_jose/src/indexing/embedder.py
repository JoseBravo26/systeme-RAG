import os
from typing import List
from mistralai.client import Mistral
from dotenv import load_dotenv

# 1. Calculer le chemin absolu vers la racine du projet
# (__file__ = src/indexing/embedder.py -> dirname = indexing -> dirname = src -> dirname = racine)
racine_projet = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Reconstituer le chemin exact du fichier .env
chemin_env = os.path.join(racine_projet, ".env")

# 3. Forcer le chargement de CE fichier spécifique
load_dotenv(chemin_env)

def get_mistral_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convertit une liste de textes en vecteurs avec mistral-embed.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est manquante dans le fichier .env.")

    # Initialisation du client V2
    client = Mistral(api_key=api_key)

    print(f"🧠 Génération des embeddings pour {len(texts)} documents...")
    
    # Appel à l'API pour générer les vecteurs
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )

    # Extraction des vecteurs
    vectors = [data.embedding for data in response.data]
    return vectors

if __name__ == "__main__":
    import pandas as pd
    
    try:
        chemin_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "evenements_clean.csv")
        df = pd.read_csv(chemin_csv, encoding="utf-8-sig")
        
        textes = df['text_to_embed'].dropna().tolist()[:3]
        vecteurs = get_mistral_embeddings(textes)
        
        print(f"✅ Succès ! {len(vecteurs)} vecteurs générés.")
        print(f"Taille d'un vecteur : {len(vecteurs[0])} dimensions.")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")