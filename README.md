# Projet 8 - Système RAG Puls-Events

API de recommandation d'événements culturels basée sur un système **RAG** (*Retrieval-Augmented Generation*) avec **FAISS**, **LangChain**, **Mistral AI** et **FastAPI**.

L'objectif du projet est de permettre à un utilisateur de poser une question en langage naturel sur des événements culturels, puis de recevoir une réponse augmentée à partir des données indexées dans une base vectorielle.

---

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation locale](#installation-locale)
- [Configuration des variables d'environnement](#configuration-des-variables-denvironnement)
- [Construction de l'index vectoriel](#construction-de-lindex-vectoriel)
- [Lancement de l'API en local](#lancement-de-lapi-en-local)
- [Documentation Swagger](#documentation-swagger)
- [Tests fonctionnels](#tests-fonctionnels)
- [Évaluation du système RAG](#évaluation-du-système-rag)
- [Exécution avec Docker](#exécution-avec-docker)
- [Démonstration soutenance](#démonstration-soutenance)
- [Structure du projet](#structure-du-projet)

---

## Vue d'ensemble

Ce projet met en œuvre un pipeline complet de recommandation d'événements :

1. ingestion et nettoyage des données événements,
2. génération d'embeddings via Mistral AI,
3. indexation vectorielle dans FAISS,
4. recherche sémantique des événements pertinents,
5. génération d'une réponse naturelle via un LLM,
6. exposition du système via une API REST FastAPI,
7. exécution locale via Docker.

Le chatbot répond à des requêtes du type :

- « Quels sont les concerts gratuits à Paris ce mois-ci ? »
- « Quels sont les événements à Paris en septembre ? »
- « Je cherche une exposition pour enfants en Seine-Saint-Denis »

---

## Architecture du projet

Le système repose sur les briques suivantes :

- **FAISS** : index vectoriel pour la recherche sémantique.
- **Mistral Embeddings** : vectorisation des événements.
- **Mistral LLM** : génération de réponses naturelles.
- **LangChain** : orchestration entre récupération contextuelle et génération.
- **FastAPI** : exposition des endpoints REST.
- **Docker** : exécution portable et locale du projet.

---

## Fonctionnalités

### Chatbot RAG

- Recherche sémantique dans les événements indexés.
- Réponses naturelles basées sur le contexte récupéré.
- Filtrage des événements passés.
- Priorisation des événements correspondant à la période demandée.
- Proposition d'alternatives si aucun événement ne correspond exactement à la période demandée.

### API REST

- `GET /health` : vérifie l'état du service.
- `POST /ask` : pose une question au chatbot.
- `POST /rebuild` : reconstruit l'index vectoriel à la demande.
- `GET /docs` : documentation Swagger générée automatiquement.

### Évaluation

- Jeu de test annoté.
- Évaluation automatique avec **Ragas**.
- Mesures de fidélité et de précision du contexte.

---

## Prérequis

Avant de lancer le projet, vérifier les éléments suivants :

- Python 3.10 ou 3.11
- Docker Desktop installé
- Une clé API Mistral valide
- Un fichier de données nettoyé dans `data/evenements_clean.csv`

---

## Installation locale

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd projet-rag
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

Sur Windows (Git Bash) :

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Configuration des variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=votre_cle_api_mistral
```

Ne jamais versionner ce fichier dans Git.

---

## Construction de l'index vectoriel

Le projet inclut un script de construction de l'index vectoriel basé sur les événements nettoyés.

### Commande recommandée

```bash
python -m src.indexing.faiss_indexer
```

Cette commande :

- charge le fichier `data/evenements_clean.csv`,
- crée les documents LangChain,
- segmente les contenus en chunks,
- génère les embeddings avec Mistral,
- construit et sauvegarde l'index FAISS dans `data/faiss_index/`.

### Résultat attendu

Après exécution, le dossier suivant doit exister :

```bash
data/faiss_index/
```

avec notamment :

- `index.faiss`
- `index.pkl`

---

## Lancement de l'API en local

L'API FastAPI peut être lancée sans Docker avec :

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Une fois démarrée, l'API est accessible à l'adresse :

```bash
http://localhost:8000
```

---

## Documentation Swagger

FastAPI génère automatiquement une documentation interactive disponible ici :

```bash
http://localhost:8000/docs
```

La spécification OpenAPI est également disponible ici :

```bash
http://localhost:8000/openapi.json
```

---

## Exemples d'utilisation de l'API

### Vérifier l'état du service

```bash
curl http://localhost:8000/health
```

Réponse attendue :

```json
{
  "status": "ready"
}
```

### Poser une question au chatbot

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les concerts gratuits à Paris ce mois-ci ?"
  }'
```

### Reconstruire l'index

```bash
curl -X POST http://localhost:8000/rebuild \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/evenements_clean.csv",
    "index_path": "data/faiss_index",
    "max_chunks": null
  }'
```

---

## Tests fonctionnels

Le projet inclut un script de test de l'API :

```bash
python api/api_test.py
```

Ce script permet de tester :

- la disponibilité de l'API,
- la génération de réponse sur `/ask`,
- la validation des entrées,
- l'accès à Swagger,
- la robustesse générale du service.

---

## Évaluation du système RAG

Le projet inclut un script d'évaluation automatique :

```bash
python eval/evaluate_rag.py
```

Il s'appuie sur un jeu de test annoté :

```bash
eval/test_set.csv
```

### Métriques utilisées

- **Faithfulness**
- **Context Precision**

Les résultats sont sauvegardés dans :

```bash
eval/results/
```

---

## Exécution avec Docker

### Construire l'image Docker

```bash
docker build -t puls-events-rag .
```

### Lancer le conteneur

```bash
docker run -p 8000:8000 --env-file .env puls-events-rag
```

### Vérifier le bon fonctionnement

Ouvrir dans le navigateur :

- Swagger UI : [http://localhost:8000/docs](http://localhost:8000/docs)
- Healthcheck : [http://localhost:8000/health](http://localhost:8000/health)

Si le port 8000 est déjà utilisé, lancer :

```bash
docker run -p 8001:8000 --env-file .env puls-events-rag
```

et utiliser ensuite :

- [http://localhost:8001/docs](http://localhost:8001/docs)
- [http://localhost:8001/health](http://localhost:8001/health)

---

## Démonstration soutenance

### Déroulé conseillé

1. Montrer la structure générale du projet.
2. Expliquer rapidement le fonctionnement du pipeline RAG.
3. Lancer l'application avec Docker.
4. Ouvrir `/docs` pour présenter les endpoints.
5. Tester `/health`.
6. Tester `/ask` avec une ou deux questions métiers.
7. Expliquer que `/rebuild` permet de reconstruire l'index si nécessaire.
8. Mentionner l'évaluation automatique avec Ragas.

### Questions de démonstration recommandées

- « Quels sont les concerts gratuits à Paris ce mois-ci ? »
- « Quels sont les événements à Paris en septembre ? »
- « Je cherche une activité culturelle gratuite à Paris »

---

## Structure du projet

```bash
.
├── api/
│   ├── api_test.py
│   └── main.py
├── data/
│   ├── evenements_clean.csv
│   └── faiss_index/
├── eval/
│   ├── evaluate_rag.py
│   ├── test_set.csv
│   └── results/
├── notebooks/
│   └── eda_events.py
├── src/
│   ├── indexing/
│   │   ├── embedder.py
│   │   └── faiss_indexer.py
│   └── rag/
│       └── chatbot.py
├── Dockerfile
├── README.md
├── README_API.md
├── requirements.txt
└── test_env.py
```

---

## Remarques finales

- Le projet est conçu comme un **POC démontrable en local**.
- L'historique conversationnel n'est pas utilisé dans cette version.
- Le endpoint `/rebuild` peut être protégé si l'application devait être exposée publiquement.
- Pour la soutenance, il est recommandé d'avoir déjà généré l'index FAISS avant de lancer la démonstration en direct.

