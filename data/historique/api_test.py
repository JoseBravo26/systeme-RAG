import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("GET /health ->", response.status_code, response.json())

def test_ask():
    payload = {
        "question": "Je cherche un concert de jazz à Paris ce week-end."
    }
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    print("POST /ask ->", response.status_code)
    print(response.json())

def test_rebuild():
    payload = {
        "csv_path": "data/evenements_clean.csv",
        "index_path": "data/faiss_index",
        "max_chunks": 300
    }
    response = requests.post(f"{BASE_URL}/rebuild", json=payload)
    print("POST /rebuild ->", response.status_code)
    print(response.json())

if __name__ == "__main__":
    test_health()
    test_rebuild()
    test_ask()