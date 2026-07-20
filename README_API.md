# 🚀 API REST Puls-Events - Guide d'utilisation

## 📋 Vue d'ensemble

API REST pour le système RAG (Retrieval-Augmented Generation) de recommandation d'événements culturels basé sur LangChain, Mistral AI et FAISS.

## 🏗️ Architecture

```
├── main-4.py              # API FastAPI (point d'entrée)
├── src/
│   ├── rag/
│   │   └── chatbot.py     # Logique du chatbot RAG
│   └── indexing/
│       └── faiss_indexer.py  # Indexation FAISS
├── api_test.py            # Tests fonctionnels de l'API
└── evaluate_rag-2.py      # Évaluation qualité avec Ragas
```

## 🔧 Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Configurer les variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=votre_clé_api_mistral
```

### 3. Préparer les données

Assurez-vous d'avoir le fichier CSV nettoyé :

```bash
data/evenements_clean.csv
```

### 4. Construire l'index FAISS

Si l'index n'existe pas encore :

```python
from src.indexing.faiss_indexer import create_documents_from_csv, split_documents, build_faiss_index

docs = create_documents_from_csv("data/evenements_clean.csv")
chunks = split_documents(docs)
build_faiss_index(chunks, save_dir="data/faiss_index")
```

## 🚀 Démarrage de l'API

### Lancer le serveur

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Options :
- `--reload` : rechargement automatique lors de modifications du code
- `--host 0.0.0.0` : accessible depuis d'autres machines
- `--port 8000` : port d'écoute (défaut: 8000)

### Vérifier que l'API est démarrée

```bash
curl http://localhost:8000/health
```

Réponse attendue :
```json
{
  "status": "ready"
}
```

## 📚 Documentation API

### Documentation interactive Swagger

Une fois l'API lancée, accédez à :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI JSON** : http://localhost:8000/openapi.json

## 🔌 Endpoints disponibles

### 1. GET `/health`

Vérifie l'état de l'API et du chatbot.

**Exemple de requête :**
```bash
curl http://localhost:8000/health
```

**Réponse :**
```json
{
  "status": "ready"
}
```

---

### 2. POST `/ask`

Pose une question au chatbot et obtient des recommandations d'événements.

**Exemple de requête :**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les événements gratuits à Paris ce mois-ci ?"
  }'
```

**Réponse :**
```json
{
  "question": "Quels sont les événements gratuits à Paris ce mois-ci ?",
  "answer": "🎉 Voici les événements gratuits à Paris ce mois-ci:\n\n1. **Festival du Cinéma en Plein Air** 🎬\n   📍 Paris • 🗓️ 15-17 juillet 2026\n   Projection de films classiques dans les jardins publics...\n\n2. **Exposition d'Art Contemporain** 🎨\n   📍 Paris • 🗓️ 20-30 juillet 2026\n   Découvrez les œuvres d'artistes émergents...",
  "sources": [
    {
      "title": "Festival du Cinéma en Plein Air",
      "city": "Paris"
    },
    {
      "title": "Exposition d'Art Contemporain",
      "city": "Paris"
    }
  ]
}
```

**Codes de statut :**
- `200` : Réponse générée avec succès
- `400` : Question vide ou invalide
- `500` : Erreur serveur (chatbot non initialisé, erreur de génération)

---

### 3. POST `/rebuild`

Reconstruit l'index FAISS à partir du CSV nettoyé et recharge le chatbot.

⚠️ **Attention** : Cette opération peut prendre plusieurs minutes selon la taille du dataset.

**Exemple de requête :**
```bash
curl -X POST http://localhost:8000/rebuild \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/evenements_clean.csv",
    "index_path": "data/faiss_index",
    "max_chunks": null
  }'
```

**Paramètres optionnels :**
- `csv_path` : chemin vers le CSV (défaut: `data/evenements_clean.csv`)
- `index_path` : répertoire de sauvegarde de l'index (défaut: `data/faiss_index`)
- `max_chunks` : limite du nombre de chunks pour test rapide (défaut: `null` = tous)

**Réponse :**
```json
{
  "status": "success",
  "message": "Index FAISS reconstruit et chatbot rechargé avec succès.",
  "nb_documents": 150,
  "nb_chunks": 450,
  "csv_path": "data/evenements_clean.csv",
  "index_path": "data/faiss_index"
}
```

**Codes de statut :**
- `200` : Reconstruction réussie
- `404` : Fichier CSV introuvable
- `500` : Erreur pendant la reconstruction

## 🧪 Tests fonctionnels

### Lancer les tests automatiques

```bash
python api_test.py
```

**Tests inclus :**
1. ✅ Health check (`/health`)
2. ✅ Question valide (`/ask`)
3. ✅ Validation question vide (erreur 400)
4. ✅ Questions multiples (robustesse)
5. ✅ Documentation Swagger accessible
6. ⏸️ Rebuild (optionnel, commenté par défaut car long)

**Résultat attendu :**
```
📊 RÉSUMÉ DES TESTS
============================================================
✅ PASSÉ - Health Check
✅ PASSÉ - Question valide
✅ PASSÉ - Validation question vide
✅ PASSÉ - Questions multiples
✅ PASSÉ - Documentation Swagger

📈 Résultat: 5/5 tests réussis (100%)

🎉 TOUS LES TESTS ONT RÉUSSI ! L'API est opérationnelle.
```

### Tests manuels avec curl

**Test 1 : Health check**
```bash
curl http://localhost:8000/health
```

**Test 2 : Poser une question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Concerts de jazz cette semaine?"}'
```

**Test 3 : Swagger UI**
```bash
open http://localhost:8000/docs
```

## 📊 Évaluation de la qualité (Ragas)

Le système inclut une évaluation automatique de la qualité des réponses avec **Ragas**.

### Lancer l'évaluation

```bash
python evaluate_rag-2.py
```

**Métriques évaluées :**
- **Faithfulness** (fidélité) : Les réponses sont-elles fidèles au contexte récupéré ?
- **Context Precision** (précision) : Les documents récupérés sont-ils pertinents ?

**Résultats générés :**
- `eval/results/ragas_scores.csv` : scores par question
- `eval/results/full_ragas_results.csv` : résultats détaillés

### Intégration CI/CD (GitHub Actions)

Exemple de workflow `.github/workflows/test-rag.yml` :

```yaml
name: Test RAG System

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run API tests
        env:
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
        run: |
          uvicorn main:app &
          sleep 10
          python api_test.py
      
      - name: Run Ragas evaluation
        env:
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
        run: python evaluate_rag-2.py
```

## 🔒 Sécurité

### Points de vigilance appliqués ✅

1. **Protection des clés API** : Variables d'environnement avec `python-dotenv`
2. **Validation des entrées** : Questions vides rejetées (erreur 400)
3. **Gestion des erreurs** : Try-except sur toutes les opérations critiques
4. **Codes HTTP appropriés** : 400, 404, 500 selon le contexte
5. **Endpoint /rebuild protégé** : Pourrait nécessiter authentification en production

### Recommandations pour la production

Si l'API doit être exposée publiquement :

1. **Ajouter authentification** (JWT, OAuth2)
2. **Rate limiting** (ex: max 10 requêtes/minute par IP)
3. **CORS configuré** pour autoriser seulement certains domaines
4. **HTTPS obligatoire**
5. **Endpoint /rebuild protégé** par token admin

Exemple avec rate limiting :
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, chat_request: ChatRequest):
    ...
```

## ⚡ Optimisation des performances

### Implémentées ✅

1. **Chargement unique au startup** : Le chatbot est chargé une seule fois
2. **Réutilisation de l'instance** : Variable globale évite rechargement
3. **Index persisté sur disque** : Pas de reconstruction à chaque démarrage
4. **Rebuild on-demand** : Seulement quand nécessaire

### Améliorations possibles

1. **Cache des réponses fréquentes** (ex: Redis)
2. **Async/await** pour opérations I/O
3. **Pool de connexions** pour Mistral API
4. **Compression des réponses** (gzip)

## 📊 Monitoring et logs

### Logs actuels

Les logs sont affichés dans stdout :
```
🚀 Chargement initial du chatbot...
✅ Chatbot prêt.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Monitoring avancé (optionnel)

Pour un environnement de production :

```python
import logging
from prometheus_client import Counter, Histogram

# Métriques Prometheus
ask_counter = Counter('api_ask_requests_total', 'Total /ask requests')
ask_duration = Histogram('api_ask_duration_seconds', 'Time spent processing /ask')

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
```

## 🤝 Contribution

Le code est structuré de façon modulaire :
- `main-4.py` : API (FastAPI)
- `src/rag/chatbot.py` : Logique RAG
- `src/indexing/faiss_indexer.py` : Indexation FAISS
- `api_test.py` : Tests
- `evaluate_rag-2.py` : Évaluation

Chaque module peut être développé indépendamment.

## 📞 Support

En cas de problème :

1. Vérifier les logs dans le terminal
2. Tester `/health` pour vérifier l'état
3. Consulter la documentation Swagger : http://localhost:8000/docs
4. Lancer les tests : `python api_test.py`

## 📄 Licence

Ce projet est un POC (Proof of Concept) pour démonstration.
