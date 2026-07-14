"""
Script de test simple pour l'API RAG Puls-Events.

À lancer après avoir démarré le serveur :
uvicorn api.main:app --reload
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print("GET /health ->", resp.status_code, resp.json())

def test_ask():
    payload = {"question": "Je cherche un concert de jazz à Paris ce week-end."}
    resp = requests.post(f"{BASE_URL}/ask", json=payload)
    print("POST /ask ->", resp.status_code)
    print(resp.json())

def test_rebuild():
    resp = requests.post(f"{BASE_URL}/rebuild")
    print("POST /rebuild ->", resp.status_code)
    print(resp.json())

if __name__ == "__main__":
    test_health()
    test_ask()
    # Décommenter si tu veux tester la reconstruction :
    # test_rebuild()