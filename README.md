# 🎭 Puls-Events : POC RAG Assistant Culturel

Ce projet est un Proof of Concept (POC) d'un chatbot RAG pour la recommandation d'événements culturels, s'appuyant sur l'API Open Agenda (Île-de-France), Mistral et FAISS.

## 📁 Structure du projet
- `api/` : Code de l'API FastAPI
- `data/` : Fichiers de données brutes et l'index FAISS
- `src/` : Scripts de collecte, traitement et vectorisation

## ⚙️ Prérequis
- Python 3.8 ou supérieur
- Une clé API Mistral valide

## 🚀 Installation et exécution

### 1. Cloner le dépôt
\`\`\`bash
git clone <URL_DU_DEPOT>
cd puls-events-rag
\`\`\`

### 2. Créer et activer l'environnement virtuel
Sur Windows :
\`\`\`bash
python -m venv env
env\Scripts\activate
\`\`\`
Sur macOS / Linux :
\`\`\`bash
python3 -m venv env
source env/bin/activate
\`\`\`

### 3. Installer les dépendances
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Configuration des variables d'environnement
Créez un fichier `.env` à la racine du projet (ce fichier est ignoré par Git) et ajoutez votre clé API Mistral :
\`\`\`env
MISTRAL_API_KEY=votre_cle_api_ici
\`\`\`
# 1. Build de l'image Docker
docker build -t puls-events-rag .

# 2. Run de l'API avec variables d'environnement
docker run -p 8000:8000 \
  -e MISTRAL_API_KEY="votre_cle_mistral" \
  -e MISTRAL_BASE_URL="https://api.mistral.ai/v1" \
  puls-events-rag