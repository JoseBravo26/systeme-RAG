# Projet 8 — Système RAG Puls-Events

POC d’un assistant conversationnel de recommandation d’événements culturels.  
Il s’appuie sur les données OpenAgenda, une recherche sémantique locale avec FAISS et les modèles Mistral AI pour répondre à des questions en langage naturel.

## Objectif métier

Puls-Events permet à un utilisateur de trouver plus facilement des événements culturels sans avoir à parcourir manuellement une liste d’annonces.

Exemple de question :

> « Y a-t-il un événement mêlant cinéma et jazz à Paris le 1er août 2026 ? »

Le système recherche d’abord des événements pertinents dans l’index, applique des filtres de date, de ville et de thème, puis génère une réponse fondée sur les informations récupérées.

## Fonctionnement RAG

RAG signifie *Retrieval-Augmented Generation*, ou génération augmentée par récupération.

```text
Données OpenAgenda
        |
        v
Nettoyage et préparation des événements
        |
        v
Embeddings Mistral + index vectoriel FAISS
        |
        v
Question utilisateur
        |
        v
Recherche sémantique + filtres métier
(date, ville, département, thème)
        |
        v
LLM Mistral : réponse naturelle basée sur le contexte
        |
        v
API FastAPI
```

Cette approche réduit le risque d’inventer des événements : le chatbot est invité à répondre uniquement à partir du contexte récupéré.

## Technologies utilisées

- **Python 3.11**
- **OpenAgenda** : source des données d’événements
- **Pandas** : préparation et nettoyage des données
- **Mistral AI** : embeddings et génération des réponses
- **LangChain** : orchestration du pipeline RAG
- **FAISS** : index vectoriel local et recherche sémantique
- **FastAPI** : API REST
- **Docker** : exécution portable du service
- **Pytest** : tests unitaires
- **Ragas** : évaluation de la qualité du système RAG
- **GitHub Actions** : intégration continue et test manuel du chatbot

## Fonctionnalités

- Recherche sémantique dans les événements indexés
- Réponses en langage naturel à partir des données récupérées
- Filtrage des événements passés
- Prise en compte de la ville, du département et de la période demandée
- Priorité aux événements correspondant exactement à la période recherchée
- Proposition d’alternatives seulement lorsqu’aucun événement ne correspond à la demande
- API REST documentée automatiquement avec Swagger
- Reconstruction de l’index à la demande
- Exécution locale avec Docker

## Prérequis

- Python 3.11 recommandé
- Docker Desktop, pour l’exécution conteneurisée
- Une clé API Mistral valide
- Le fichier `data/evenements_clean.csv` pour reconstruire l’index FAISS

> Le fichier source `data/evenements_clean.csv` n’est pas versionné dans Git.  
> En revanche, l’index FAISS préconstruit est fourni dans `data/faiss_index/` pour pouvoir lancer l’API sans reconstruire l’index.

## Installation locale

### Cloner le dépôt

```bash
git clone https://github.com/JoseBravo26/systeme-RAG.git
cd systeme-RAG
```

### Créer l’environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Sous macOS ou Linux :

```bash
python -m venv env
source env/bin/activate
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

## Variables d’environnement

Créer un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=votre_cle_api_mistral
MISTRAL_BASE_URL=https://api.mistral.ai/v1
MISTRAL_CHAT_MODEL=open-mistral-nemo
MISTRAL_EMBED_MODEL=mistral-embed
```

Le fichier `.env` ne doit jamais être envoyé sur GitHub.

## Construire l’index FAISS

Cette étape est nécessaire uniquement si :

- l’index n’existe pas encore ;
- le fichier `data/evenements_clean.csv` a changé ;
- vous souhaitez mettre à jour les événements indexés.

Exécuter depuis la racine du projet :

```bash
python -m src.indexing.faiss_indexer
```

Le script :

1. charge les événements nettoyés ;
2. crée des documents avec leurs métadonnées ;
3. génère les embeddings avec Mistral ;
4. construit l’index FAISS ;
5. sauvegarde l’index dans `data/faiss_index/`.

Les fichiers attendus sont :

```text
data/faiss_index/index.faiss
data/faiss_index/index.pkl
```

## Lancer l’API sans Docker

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Pour le développement, avec rechargement automatique :

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Accéder ensuite à Swagger :

```text
http://127.0.0.1:8000/docs
```

## Lancer avec Docker

### Construire l’image

```bash
docker build -t puls-events-rag:1.0 .
```

### Démarrer l’API

```bash
docker run --rm --name puls-events-api --env-file .env -p 8000:8000 puls-events-rag:1.0
```

Puis ouvrir :

- Swagger : http://127.0.0.1:8000/docs
- Health check : http://127.0.0.1:8000/health
- Spécification OpenAPI : http://127.0.0.1:8000/openapi.json

Si le port 8000 est occupé :

```bash
docker run --rm --name puls-events-api --env-file .env -p 8001:8000 puls-events-rag:1.0
```

Swagger sera alors disponible sur `http://127.0.0.1:8001/docs`.

## Endpoints API

### `GET /health`

Vérifie que l’API et le chatbot sont prêts.

```bash
curl http://127.0.0.1:8000/health
```

Réponse attendue :

```json
{
  "status": "ready",
  "rebuilding": false
}
```

### `POST /ask`

Envoie une question au chatbot.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Que se passe-t-il à Paris le 12 août 2026 autour de l’astronomie ?\"}"
```

La réponse contient :

- la question envoyée ;
- une réponse générée par le chatbot ;
- les sources utilisées, avec le titre et la ville des événements.

### `POST /rebuild`

Reconstruit l’index FAISS et recharge le chatbot.

> Cette opération peut être longue et nécessite le fichier local `data/evenements_clean.csv`.

```bash
curl -X POST http://127.0.0.1:8000/rebuild \
  -H "Content-Type: application/json" \
  -d "{\"csv_path\": \"data/evenements_clean.csv\", \"index_path\": \"data/faiss_index\", \"max_chunks\": null}"
```

## Tests

### Tests unitaires

```bash
pytest -q
```

Les tests unitaires vérifient notamment :

- le nettoyage des données ;
- l’interprétation des dates et villes ;
- le filtrage métier par thème, période et localisation.

Ils sont conçus pour ne pas appeler Mistral ni charger l’index FAISS.[file:113][file:114]

### Tests fonctionnels de l’API

Démarrer d’abord l’API, puis dans un second terminal :

```bash
python api/api_test.py
```

Le script vérifie le health check, les requêtes `/ask`, la validation des entrées et l’accès à la documentation Swagger.

## Évaluation RAG

Le script d’évaluation utilise un jeu de test annoté situé dans :

```text
eval/test_set.csv
```

Lancer l’évaluation :

```bash
python eval/evaluate_rag.py
```

Les métriques Ragas utilisées sont :

- **Faithfulness** : vérifie que la réponse reste fidèle aux informations présentes dans le contexte récupéré.
- **Context Precision** : vérifie que les documents récupérés sont pertinents pour la question.

Les résultats détaillés sont enregistrés dans :

```text
eval/results/ragas_scores.csv
```

## Scénarios de démonstration

Pour une démonstration, utiliser des questions déjà couvertes par les tests :

1. `Y a-t-il un événement mêlant cinéma et jazz à Paris le 1er août 2026 ?`
2. `Existe-t-il une visite des coulisses du Grand Rex à Paris en septembre 2026 ?`
3. `Que se passe-t-il à Paris le 12 août 2026 autour de l’astronomie ?`

Déroulé conseillé :

1. lancer le conteneur Docker avant la démonstration ;
2. ouvrir Swagger sur `/docs` ;
3. vérifier `/health` ;
4. envoyer une question sur `/ask` ;
5. montrer la réponse et les sources associées ;
6. expliquer que FAISS recherche les événements et que Mistral rédige la réponse.

## Sécurité et limites

- La clé Mistral est stockée dans `.env`, exclu de Git.
- L’index FAISS est local, mais l’application nécessite l’API Mistral pour créer les embeddings de requêtes et générer les réponses.
- L’endpoint `/rebuild` doit être protégé par authentification dans une version accessible publiquement.
- Une limite de requêtes, HTTPS, CORS restrictif et une supervision devraient être ajoutés avant une mise en production.
- Cette version ne conserve pas encore l’historique conversationnel.

## Améliorations envisagées

- Mise à jour planifiée des événements depuis OpenAgenda
- Filtres enrichis : gratuité, accessibilité, catégorie, distance et horaires
- Interface web destinée aux utilisateurs finaux
- Historique conversationnel
- Suivi de la latence, des erreurs, des coûts et des requêtes sans réponse
- Jeu de test annoté plus large et suivi des métriques dans le temps

## Structure du projet

```text
.
├── api/
│   ├── api_test.py
│   └── main.py
├── data/
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
├── eval/
│   ├── evaluate_rag.py
│   ├── test_set.csv
│   └── results/
├── src/
│   ├── indexing/
│   │   └── faiss_indexer.py
│   ├── ingestion/
│   │   └── data_pipeline.py
│   └── rag/
│       └── chatbot.py
├── tests/
│   ├── test_chatbot.py
│   └── test_data_pipeline.py
├── .github/workflows/
├── Dockerfile
├── README.md
└── requirements.txt
```

## Auteur
José Bravo
Projet réalisé dans le cadre du parcours OpenClassrooms — Projet 8 : conception et déploiement d’un système RAG.