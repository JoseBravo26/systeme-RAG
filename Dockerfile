# Dockerfile - Projet 8 système RAG Puls-Events

FROM python:3.11-slim

# 1. Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Dossier de travail dans le conteneur
WORKDIR /app

# 3. Copie des fichiers de dépendances
COPY requirements.txt /app/

# 4. Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie du code du projet
COPY . /app/

# 6. Port exposé (FastAPI / Uvicorn)
EXPOSE 8000

# 7. Commande de démarrage :
#    - reconstruction de l'index FAISS (si besoin)
#    - lancement de l'API FastAPI
CMD ["/bin/sh", "-c", "python -m src.indexing.faiss_indexer && uvicorn api.main:app --host 0.0.0.0 --port 8000"]