# Rapport de tests et validation du système RAG

## Objectif

Ce document résume les tests mis en place pour valider le système RAG de recommandation d'événements culturels Puls-Events.

L'objectif est de vérifier la fiabilité du pipeline complet :
- récupération,
- nettoyage,
- vectorisation,
- interrogation de la base vectorielle,
- génération de réponses,
- exposition API,
- évaluation de qualité.

---

## Tests mis en place

### 1. Tests d'environnement

**Fichier :** `test_env.py`

Ces tests permettent de vérifier :

- que les dépendances principales sont bien installées,
- que les imports critiques fonctionnent,
- que la variable d'environnement `MISTRAL_API_KEY` est disponible,
- que l'environnement d'exécution est prêt.

### 2. Tests du pipeline de données

**Fichier :** `tests/test_data_pipeline.py`

Ces tests permettent de valider :

- le chargement des données,
- le prétraitement,
- le nettoyage,
- la cohérence générale du pipeline de données.

### 3. Tests du chatbot RAG

**Fichier :** `tests/test_chatbot.py`

Ces tests unitaires vérifient :

- que la méthode `ask()` renvoie un dictionnaire,
- que les champs `question`, `answer` et `sources` sont présents,
- que la réponse générée n'est pas vide,
- que les sources sont bien renvoyées sous forme de liste,
- que la question utilisateur est bien conservée dans la sortie.

### 4. Tests fonctionnels de l'API

**Fichier :** `api/api_test.py`

Ces tests permettent de vérifier :

- l'accessibilité de l'API,
- le bon fonctionnement de `/health`,
- le bon fonctionnement de `/ask`,
- la validation des entrées,
- l'accès à la documentation Swagger,
- la robustesse générale des endpoints.

---

## Relance automatique des tests

Les tests sont relançables automatiquement de plusieurs façons.

### En local

Tests unitaires :

```bash
pytest test_env.py tests/test_data_pipeline.py tests/test_chatbot.py
```

Tests unitaires avec couverture :

```bash
pytest test_env.py tests/test_data_pipeline.py tests/test_chatbot.py --cov --cov-config=.coveragerc --cov-report=term-missing --cov-report=xml --cov-report=html --cov-fail-under=30
```

Tests fonctionnels API :

```bash
python api/api_test.py
```

### En intégration continue

Le pipeline GitHub Actions exécute automatiquement :

- les tests unitaires,
- la couverture de code,
- la construction de l'index FAISS,
- les tests API,
- le build Docker,
- un smoke test du conteneur.

---

## Indicateurs utilisés

### 1. Couverture de tests

La couverture de code est mesurée automatiquement avec `pytest-cov`.

**Indicateur suivi :**
- couverture globale (%)

### 2. Qualité des réponses RAG

Le script `eval/evaluate_rag.py` permet d'évaluer automatiquement la qualité des réponses générées.

**Indicateurs suivis :**

- **Faithfulness** : fidélité de la réponse par rapport au contexte récupéré,
- **Context Precision** : pertinence des documents récupérés,
- **score de similarité sémantique** (si activé ultérieurement),
- **taux de réponse correcte / partiellement correcte / incorrecte** (classification manuelle ou semi-automatique possible).

### 3. Robustesse fonctionnelle

**Indicateurs observés :**

- disponibilité de l'API,
- présence d'une réponse non vide,
- présence de sources,
- fonctionnement des endpoints critiques.

---

## Résumé lisible des résultats

### Résultats attendus

- Les tests unitaires doivent réussir.
- Les tests API doivent retourner des codes HTTP valides.
- Le chatbot doit renvoyer une réponse structurée et exploitable.
- L'évaluation RAG doit produire des scores lisibles dans des fichiers CSV.

### Logs disponibles

Les résultats sont visibles :

- dans la sortie console des tests,
- dans les logs GitHub Actions,
- dans les artefacts de couverture,
- dans les résultats CSV générés par `eval/evaluate_rag.py`.

---

## Erreurs fréquentes identifiées

Les erreurs ou difficultés les plus fréquentes observées sur ce type de système sont les suivantes :

1. **Clé API Mistral absente ou invalide**
   - empêche la génération d'embeddings ou de réponses.

2. **Index FAISS absent ou non reconstruit**
   - empêche la récupération des documents.

3. **Données mal formatées ou dates invalides**
   - perturbent le filtrage temporel des événements.

4. **Réponses trop courtes ou trop génériques**
   - peuvent apparaître selon le prompt et la température du modèle.

5. **Récupération de documents peu pertinents**
   - peut dégrader la qualité finale de la réponse.

6. **Différences entre environnement local, Docker et CI**
   - peuvent provoquer des erreurs de chemin, de port ou de dépendances.

---

## Limites du système

Les principales limites identifiées sont :

- la qualité des réponses dépend directement de la qualité des données sources,
- la récupération sémantique peut manquer certains événements si la requête est ambiguë,
- le modèle génératif peut produire des formulations variables selon les paramètres,
- l'historique conversationnel n'est pas géré dans le POC,
- certaines évaluations restent partiellement qualitatives si elles reposent sur annotation humaine,
- le système dépend de services externes (API Mistral) pour fonctionner complètement.

---

## Conclusion

Le système dispose désormais :

- de tests unitaires,
- de tests fonctionnels,
- d'un pipeline relançable automatiquement,
- d'indicateurs de qualité explicites,
- d'une documentation claire des résultats, erreurs fréquentes et limites.

Cela permet de présenter une validation cohérente et argumentée du projet lors de la soutenance.