"""
Script de test fonctionnel de l'API REST Puls-Events.

Ce script effectue de vrais appels HTTP vers l'API FastAPI locale.
Il vérifie :
- GET /health ;
- POST /ask avec une question valide ;
- POST /ask avec une question vide ;
- POST /ask avec plusieurs questions ;
- disponibilité de Swagger et de la spécification OpenAPI ;
- POST /rebuild (optionnel, car long et coûteux).

Prérequis :
1. Démarrer l'API dans un autre terminal :
   uvicorn api.main:app --reload

2. Lancer ce script :
   python api/api_test.py

Note :
Le test /rebuild est volontairement désactivé dans la suite rapide. Il
reconstruit l'index FAISS complet et génère à nouveau les embeddings.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# Configuration
# =====================================================================

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 60
REBUILD_TIMEOUT_SECONDS = 1800

# Ne passe à "true" que lorsque tu souhaites réellement reconstruire
# l'index complet via l'endpoint /rebuild.
RUN_REBUILD_TEST = os.getenv("RUN_REBUILD_TEST", "false").lower() == "true"

# Cette variable doit correspondre à REBUILD_API_KEY dans .env si une
# protection est active côté API.
REBUILD_API_KEY = os.getenv("REBUILD_API_KEY")


# =====================================================================
# Utilitaires d'affichage et HTTP
# =====================================================================

def print_test_header(test_name: str) -> None:
    """Affiche un en-tête lisible pour un test."""
    print("\n" + "=" * 70)
    print(f"🧪 TEST : {test_name}")
    print("=" * 70)


def print_success(message: str) -> None:
    """Affiche un résultat positif."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Affiche un résultat négatif."""
    print(f"❌ {message}")


def safe_json(response: requests.Response) -> Any:
    """
    Retourne le JSON de la réponse lorsqu'il est disponible.

    En cas de réponse non JSON, retourne le texte HTTP pour faciliter
    le diagnostic sans faire échouer le script d'affichage.
    """
    try:
        return response.json()
    except ValueError:
        return response.text


def print_response(response: requests.Response) -> None:
    """Affiche le statut HTTP et le contenu d'une réponse."""
    print(f"Statut HTTP : {response.status_code}")
    print(f"Réponse : {safe_json(response)}")


def is_api_available() -> bool:
    """
    Vérifie que le serveur FastAPI répond avant de lancer la suite de tests.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=TIMEOUT_SECONDS,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# =====================================================================
# Tests des endpoints
# =====================================================================

def test_health_endpoint() -> bool:
    """
    Vérifie que l'endpoint GET /health répond et que le chatbot est prêt.
    """
    print_test_header("GET /health")

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=TIMEOUT_SECONDS,
        )
        print_response(response)

        if response.status_code != 200:
            print_error(
                f"Statut inattendu pour /health : {response.status_code}"
            )
            return False

        data = safe_json(response)

        if not isinstance(data, dict):
            print_error("La réponse /health n'est pas un objet JSON.")
            return False

        if data.get("status") != "ready":
            print_error(
                "L'API répond mais le chatbot n'est pas prêt. "
                f"Statut reçu : {data.get('status')}"
            )
            return False

        if data.get("rebuilding") is True:
            print_error("Une reconstruction est en cours.")
            return False

        print_success("API prête et chatbot chargé.")
        return True

    except requests.RequestException as exc:
        print_error(f"Erreur HTTP pendant le test /health : {exc}")
        return False


def test_ask_endpoint_valid() -> bool:
    """
    Vérifie que POST /ask retourne une réponse structurée pour une question
    déjà validée dans le jeu d'évaluation RAG.
    """
    print_test_header("POST /ask - Question valide")

    payload = {
        "question": (
            "Quels concerts de jazz sont proposés à Paris "
            "le 24 juillet 2026 ?"
        )
    }

    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        print_response(response)

        if response.status_code != 200:
            print_error(
                f"Statut inattendu pour /ask : {response.status_code}"
            )
            return False

        data = safe_json(response)

        if not isinstance(data, dict):
            print_error("La réponse /ask n'est pas un objet JSON.")
            return False

        required_fields = {"question", "answer", "sources"}

        if not required_fields.issubset(data.keys()):
            print_error(
                "Structure de réponse invalide. "
                f"Champs reçus : {list(data.keys())}"
            )
            return False

        if data["question"] != payload["question"]:
            print_error("La question renvoyée ne correspond pas à la requête.")
            return False

        if not isinstance(data["answer"], str) or not data["answer"].strip():
            print_error("La réponse générée est vide ou invalide.")
            return False

        if not isinstance(data["sources"], list):
            print_error("Le champ sources doit être une liste.")
            return False

        for source in data["sources"]:
            if not isinstance(source, dict):
                print_error("Une source n'est pas un objet JSON.")
                return False

            if "title" not in source or "city" not in source:
                print_error("Une source ne contient pas title et city.")
                return False

        print_success("Réponse /ask valide et correctement structurée.")
        print(f"Question : {data['question']}")
        print(f"Réponse : {data['answer'][:250]}...")
        print(f"Nombre de sources : {len(data['sources'])}")

        return True

    except requests.RequestException as exc:
        print_error(f"Erreur HTTP pendant le test /ask : {exc}")
        return False


def test_ask_endpoint_empty_question() -> bool:
    """
    Vérifie qu'une question composée uniquement d'espaces est rejetée.

    La version recommandée de l'API peut répondre 400 après strip()
    ou 422 si la validation Pydantic bloque directement la requête.
    """
    print_test_header("POST /ask - Question vide")

    payload = {"question": "   "}

    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        print_response(response)

        if response.status_code in {400, 422}:
            print_success("Question vide correctement rejetée.")
            return True

        print_error(
            "Une question vide doit être rejetée avec un statut 400 ou 422. "
            f"Statut reçu : {response.status_code}"
        )
        return False

    except requests.RequestException as exc:
        print_error(f"Erreur HTTP pendant la validation : {exc}")
        return False


def test_ask_endpoint_multiple_questions() -> bool:
    """
    Vérifie que l'API traite plusieurs requêtes consécutives sans recharger
    le chatbot ni retourner d'erreur HTTP.
    """
    print_test_header("POST /ask - Questions multiples")

    questions = [
        "Y a-t-il un événement mêlant cinéma et jazz à Paris le 1er août 2026 ?",
        "Existe-t-il une visite des coulisses du Grand Rex à Paris en septembre 2026 ?",
        "Que se passe-t-il à Paris le 12 août 2026 autour de l’astronomie ?",
    ]

    successful_questions = 0

    for index, question in enumerate(questions, start=1):
        print(f"\nQuestion {index}/{len(questions)} : {question}")

        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"question": question},
                timeout=TIMEOUT_SECONDS,
            )

            if response.status_code != 200:
                print_error(f"Erreur HTTP {response.status_code} pour cette question.")
                print(f"Détail : {safe_json(response)}")
                continue

            data = safe_json(response)

            if (
                not isinstance(data, dict)
                or not isinstance(data.get("answer"), str)
                or not data["answer"].strip()
            ):
                print_error("Réponse vide ou structure inattendue.")
                continue

            source_count = len(data.get("sources", []))
            print(
                "✓ Réponse reçue "
                f"({len(data['answer'])} caractères, {source_count} source(s))."
            )
            successful_questions += 1

        except requests.RequestException as exc:
            print_error(f"Erreur HTTP : {exc}")

    if successful_questions == len(questions):
        print_success(
            f"Toutes les questions ont été traitées : "
            f"{successful_questions}/{len(questions)}."
        )
        return True

    print_error(
        f"Questions traitées avec succès : "
        f"{successful_questions}/{len(questions)}."
    )
    return False


def test_swagger_documentation() -> bool:
    """
    Vérifie que Swagger UI, OpenAPI et les routes exigées sont accessibles.
    """
    print_test_header("Documentation Swagger et OpenAPI")

    try:
        swagger_response = requests.get(
            f"{BASE_URL}/docs",
            timeout=TIMEOUT_SECONDS,
        )
        openapi_response = requests.get(
            f"{BASE_URL}/openapi.json",
            timeout=TIMEOUT_SECONDS,
        )

        print(f"Statut /docs : {swagger_response.status_code}")
        print(f"Statut /openapi.json : {openapi_response.status_code}")

        if swagger_response.status_code != 200:
            print_error("Swagger UI n'est pas accessible.")
            return False

        if openapi_response.status_code != 200:
            print_error("La spécification OpenAPI n'est pas accessible.")
            return False

        specification = safe_json(openapi_response)

        if not isinstance(specification, dict):
            print_error("La spécification OpenAPI n'est pas un objet JSON.")
            return False

        paths = specification.get("paths", {})

        required_routes = {
            "/health": "get",
            "/ask": "post",
            "/rebuild": "post",
        }

        missing_routes = []

        for route, method in required_routes.items():
            if route not in paths or method not in paths[route]:
                missing_routes.append(f"{method.upper()} {route}")

        if missing_routes:
            print_error(
                "Routes absentes de la documentation OpenAPI : "
                + ", ".join(missing_routes)
            )
            return False

        api_info = specification.get("info", {})

        print(f"Titre API : {api_info.get('title', 'Non renseigné')}")
        print(f"Version API : {api_info.get('version', 'Non renseignée')}")
        print(f"Routes documentées : {list(paths.keys())}")

        print_success("Swagger UI et spécification OpenAPI disponibles.")
        return True

    except requests.RequestException as exc:
        print_error(f"Erreur HTTP pendant le test Swagger : {exc}")
        return False


def test_rebuild_endpoint() -> bool:
    """
    Lance une reconstruction complète via POST /rebuild.

    Ce test est optionnel car il peut durer plusieurs minutes et appelle
    le modèle d'embeddings pour tous les chunks du jeu de données.
    Aucun CSV, chemin d'index ou max_chunks n'est envoyé par le client.
    """
    print_test_header("POST /rebuild - Reconstruction complète")

    headers: Dict[str, str] = {}

    if REBUILD_API_KEY:
        headers["X-Rebuild-Api-Key"] = REBUILD_API_KEY

    print("⚠️ Reconstruction complète de l'index FAISS en cours...")
    print("Cette opération peut prendre plusieurs minutes.")

    try:
        response = requests.post(
            f"{BASE_URL}/rebuild",
            headers=headers,
            timeout=REBUILD_TIMEOUT_SECONDS,
        )
        print_response(response)

        if response.status_code != 200:
            print_error(
                f"Statut inattendu pour /rebuild : {response.status_code}"
            )
            return False

        data = safe_json(response)

        if not isinstance(data, dict):
            print_error("La réponse /rebuild n'est pas un objet JSON.")
            return False

        if data.get("status") != "success":
            print_error("Le rebuild ne retourne pas le statut success.")
            return False

        if data.get("nb_documents", 0) <= 0:
            print_error("Le rebuild indique zéro document.")
            return False

        if data.get("nb_chunks", 0) <= 0:
            print_error("Le rebuild indique zéro chunk.")
            return False

        print_success("Index FAISS complet reconstruit avec succès.")
        print(f"Documents indexés : {data['nb_documents']}")
        print(f"Chunks générés : {data['nb_chunks']}")

        return True

    except requests.RequestException as exc:
        print_error(f"Erreur HTTP pendant le rebuild : {exc}")
        return False


# =====================================================================
# Exécution de la suite de tests
# =====================================================================

TestFunction = Callable[[], bool]


def run_all_tests() -> bool:
    """
    Exécute les tests fonctionnels rapides et affiche leur bilan.

    Le test de reconstruction est ajouté uniquement si la variable
    RUN_REBUILD_TEST=true est définie dans l'environnement.
    """
    print("\n" + "🚀" * 30)
    print("DÉBUT DES TESTS FONCTIONNELS DE L'API PULS-EVENTS")
    print("🚀" * 30)

    print(f"\nURL de base : {BASE_URL}")
    print(f"Timeout standard : {TIMEOUT_SECONDS} secondes")
    print(f"Test rebuild activé : {RUN_REBUILD_TEST}")

    if not is_api_available():
        print_error(
            "L'API n'est pas joignable. Démarre-la dans un autre terminal :\n"
            "uvicorn api.main:app --reload"
        )
        return False

    tests: List[Tuple[str, TestFunction]] = [
        ("Health check", test_health_endpoint),
        ("Question valide", test_ask_endpoint_valid),
        ("Validation question vide", test_ask_endpoint_empty_question),
        ("Questions multiples", test_ask_endpoint_multiple_questions),
        ("Documentation Swagger", test_swagger_documentation),
    ]

    if RUN_REBUILD_TEST:
        tests.append(("Rebuild complet", test_rebuild_endpoint))
    else:
        print(
            "\nℹ️ Test /rebuild ignoré. Pour l'activer :\n"
            "$env:RUN_REBUILD_TEST = 'true'\n"
            "python api/api_test.py"
        )

    results: Dict[str, bool] = {}

    for test_name, test_function in tests:
        try:
            results[test_name] = test_function()

        except KeyboardInterrupt:
            print("\n\n⚠️ Tests interrompus par l'utilisateur.")
            return False

        except Exception as exc:
            print_error(
                f"Erreur inattendue pendant le test « {test_name} » : {exc}"
            )
            results[test_name] = False

    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    percentage = (passed / total * 100) if total else 0

    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)

    for test_name, success in results.items():
        status = "✅ PASSÉ" if success else "❌ ÉCHOUÉ"
        print(f"{status} - {test_name}")

    print(f"\nRésultat : {passed}/{total} tests réussis ({percentage:.0f} %).")

    if failed == 0:
        print("\n🎉 Tous les tests fonctionnels ont réussi.")
        return True

    print(f"\n⚠️ {failed} test(s) ont échoué. Consulte les logs ci-dessus.")
    return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n👋 Tests interrompus.")
        sys.exit(2)