# Couverture de tests

Ce projet utilise **pytest-cov** dans GitHub Actions pour mesurer la couverture de code.

## Seuil actuel

- Couverture minimale requise : **60%**

## Workflow CI

Le pipeline GitHub Actions :

1. exécute les tests unitaires,
2. génère `coverage.xml`,
3. génère le rapport HTML `htmlcov/`,
4. échoue automatiquement si la couverture est inférieure au seuil défini.

## Commande locale

```bash
pytest test_env.py tests/test_data_pipeline.py \
  --cov=src \
  --cov=api \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-report=html
```

## Badge de couverture

Si vous utilisez Codecov ou un autre service, vous pouvez ensuite ajouter un badge dans le README principal.

Exemple Markdown :

```md
![Coverage](https://img.shields.io/badge/coverage-60%25-yellow)
```

Ou avec Codecov :

```md
[![codecov](https://codecov.io/gh/<user>/<repo>/branch/main/graph/badge.svg)](https://codecov.io/gh/<user>/<repo>)
```
