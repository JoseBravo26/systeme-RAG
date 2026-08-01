# Image Python légère et reproductible.
FROM python:3.11-slim

# Empêche les fichiers .pyc et affiche les logs immédiatement.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Répertoire de travail du projet dans le conteneur.
WORKDIR /app

# Installation séparée des dépendances pour tirer parti du cache Docker.
COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copie du code source, des scripts et du CSV nécessaire à l'indexation.
COPY . .

# Création des répertoires attendus par l'application.
RUN mkdir -p data/faiss_index eval/results

# Documentation du port HTTP de FastAPI.
EXPOSE 8000

# Démarrage de l'API sans le mode reload, réservé au développement local.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]