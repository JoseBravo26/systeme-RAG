"""
Script de test fonctionnel de l'API REST Puls-Events.

Ce script vérifie que tous les endpoints de l'API fonctionnent correctement.
Utilise la bibliothèque requests pour effectuer les appels HTTP.

Usage:
    1. Démarrer l'API: uvicorn main:app --reload
    2. Lancer les tests: python api_test.py
"""

import requests
import time
from typing import Dict, Any


# =========================
# Configuration
# =========================

BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # secondes


# =========================
# Fonctions utilitaires
# =========================

def print_test_header(test_name: str):
    """Affiche un en-tête de test."""
    print("\n" + "=" * 60)
    print(f"🧪 TEST: {test_name}")
    print("=" * 60)


def print_success(message: str):
    """Affiche un message de succès."""
    print(f"✅ {message}")


def print_error(message: str):
    """Affiche un message d'erreur."""
    print(f"❌ {message}")


def print_response(response: requests.Response):
    """Affiche les détails d'une réponse HTTP."""
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")


# =========================
# Tests des endpoints
# =========================

def test_health_endpoint():
    """
    Test 1: Vérifie que l'endpoint /health fonctionne.
    """
    print_test_header("Endpoint /health")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        print_response(response)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ready":
                print_success("L'API est prête et le chatbot est chargé")
                return True
            else:
                print_error("Le chatbot n'est pas chargé")
                return False
        else:
            print_error(f"Code de statut inattendu: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur lors du test /health: {e}")
        return False


def test_ask_endpoint_valid():
    """
    Test 2: Vérifie que l'endpoint /ask fonctionne avec une question valide.
    """
    print_test_header("Endpoint /ask - Question valide")
    
    payload = {
        "question": "Quels sont les événements gratuits à Paris ce mois-ci ?"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT
        )
        print_response(response)
        
        if response.status_code == 200:
            data = response.json()
            
            # Vérification de la structure de la réponse
            if all(key in data for key in ["question", "answer", "sources"]):
                print_success("Réponse structurée correctement")
                print(f"   Question: {data['question'][:50]}...")
                print(f"   Réponse: {data['answer'][:100]}...")
                print(f"   Nombre de sources: {len(data['sources'])}")
                return True
            else:
                print_error("Structure de réponse incorrecte")
                return False
        else:
            print_error(f"Code de statut inattendu: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur lors du test /ask: {e}")
        return False


def test_ask_endpoint_empty_question():
    """
    Test 3: Vérifie que l'endpoint /ask rejette les questions vides.
    """
    print_test_header("Endpoint /ask - Question vide (validation)")
    
    payload = {
        "question": "   "  # Question vide avec espaces
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT
        )
        print_response(response)
        
        if response.status_code == 400:
            print_success("Validation correcte: question vide rejetée")
            return True
        else:
            print_error(f"La validation devrait retourner 400, reçu: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur lors du test validation: {e}")
        return False


def test_ask_endpoint_multiple_questions():
    """
    Test 4: Teste plusieurs questions différentes pour vérifier la robustesse.
    """
    print_test_header("Endpoint /ask - Questions multiples")
    
    questions = [
        "Y a-t-il des concerts de jazz cette semaine ?",
        "Je cherche une exposition pour enfants en Seine-Saint-Denis",
        "Quelles sont les activités gratuites en plein air ce weekend ?"
    ]
    
    success_count = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n   Question {i}/{len(questions)}: {question[:60]}...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"question": question},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✓ Réponse reçue ({len(data['answer'])} caractères)")
                success_count += 1
            else:
                print(f"   ✗ Erreur {response.status_code}")
                
        except Exception as e:
            print(f"   ✗ Exception: {e}")
    
    if success_count == len(questions):
        print_success(f"Toutes les questions ({success_count}/{len(questions)}) traitées avec succès")
        return True
    else:
        print_error(f"Seulement {success_count}/{len(questions)} questions traitées")
        return False


def test_rebuild_endpoint():
    """
    Test 5: Vérifie que l'endpoint /rebuild fonctionne.
    
    ATTENTION: Ce test reconstruit réellement l'index. 
    Il peut prendre plusieurs minutes selon la taille du dataset.
    """
    print_test_header("Endpoint /rebuild - Reconstruction de l'index")
    
    print("⚠️  Ce test va reconstruire l'index FAISS (opération longue)")
    print("   Appuyez sur Ctrl+C pour annuler...")
    time.sleep(3)
    
    payload = {
        "csv_path": "data/evenements_clean.csv",
        "index_path": "data/faiss_index",
        "max_chunks": 100  # Limité pour test rapide
    }
    
    try:
        print("   🔄 Reconstruction en cours...")
        response = requests.post(
            f"{BASE_URL}/rebuild",
            json=payload,
            timeout=120  # Timeout plus long pour rebuild
        )
        print_response(response)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print_success("Index reconstruit avec succès")
                print(f"   Documents: {data.get('nb_documents')}")
                print(f"   Chunks: {data.get('nb_chunks')}")
                return True
            else:
                print_error("Rebuild échoué")
                return False
        else:
            print_error(f"Code de statut inattendu: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur lors du test /rebuild: {e}")
        return False


def test_swagger_documentation():
    """
    Test 6: Vérifie que la documentation Swagger est accessible.
    """
    print_test_header("Documentation Swagger")
    
    try:
        # Test /docs (Swagger UI)
        response_docs = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        
        # Test /openapi.json (spécification OpenAPI)
        response_openapi = requests.get(f"{BASE_URL}/openapi.json", timeout=TIMEOUT)
        
        print(f"   /docs status: {response_docs.status_code}")
        print(f"   /openapi.json status: {response_openapi.status_code}")
        
        if response_docs.status_code == 200 and response_openapi.status_code == 200:
            openapi_spec = response_openapi.json()
            print(f"   API Title: {openapi_spec.get('info', {}).get('title')}")
            print(f"   API Version: {openapi_spec.get('info', {}).get('version')}")
            print(f"   Endpoints disponibles: {list(openapi_spec.get('paths', {}).keys())}")
            print_success("Documentation Swagger accessible")
            return True
        else:
            print_error("Documentation Swagger non accessible")
            return False
            
    except Exception as e:
        print_error(f"Erreur lors du test Swagger: {e}")
        return False


# =========================
# Exécution des tests
# =========================

def run_all_tests():
    """
    Execute tous les tests et affiche un résumé.
    """
    print("\n" + "🚀" * 30)
    print("DÉBUT DES TESTS FONCTIONNELS DE L'API PULS-EVENTS")
    print("🚀" * 30)
    
    print(f"\n📍 Base URL: {BASE_URL}")
    print(f"⏱️  Timeout: {TIMEOUT}s\n")
    
    # Liste des tests à exécuter
    tests = [
        ("Health Check", test_health_endpoint),
        ("Question valide", test_ask_endpoint_valid),
        ("Validation question vide", test_ask_endpoint_empty_question),
        ("Questions multiples", test_ask_endpoint_multiple_questions),
        ("Documentation Swagger", test_swagger_documentation),
        # ("Rebuild (optionnel)", test_rebuild_endpoint),  # Commenté car long
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrompus par l'utilisateur")
            break
        except Exception as e:
            print_error(f"Erreur inattendue dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for result in results.values() if result)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Résultat: {passed}/{total} tests réussis ({(passed/total)*100:.0f}%)")
    
    if failed == 0:
        print("\n🎉 TOUS LES TESTS ONT RÉUSSI ! L'API est opérationnelle.")
    else:
        print(f"\n⚠️  {failed} test(s) ont échoué. Vérifiez les logs ci-dessus.")
    
    return failed == 0


# =========================
# Point d'entrée
# =========================

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrompus. Au revoir !")
        exit(2)
