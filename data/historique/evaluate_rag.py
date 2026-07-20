import sys
import os
import pandas as pd

# Ajout du dossier racine au chemin de Python pour qu'il trouve le dossier 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.chatbot import PulsEventsChatbot

def run_evaluation():
    test_cases = [
        {
            "question": "Quels sont les concerts de type fanfare ou jazz prévus à Paris ?",
            "expected_keywords": ["Fanfare", "Dry Bayou", "CINÉ-JAZZ"]
        },
        {
            "question": "Y a-t-il des expositions gratuites pour enfants dans le 93 ?",
            "expected_keywords": ["gratuit", "enfant", "Seine-Saint-Denis"] 
        },
        {
            "question": "Où puis-je voir un match de rugby sur la lune ?",
            "expected_keywords": ["ne trouves pas", "désolé", "pas d'événement"] 
        }
    ]

    print("🤖 Initialisation du Chatbot pour évaluation...")
    bot = PulsEventsChatbot()
    
    results = []
    
    for i, test in enumerate(test_cases):
        print(f"\n⏳ Évaluation du scénario {i+1}...")
        response = bot.ask(test["question"])
        answer_text = response["answer"].lower()
        
        is_correct = any(kw.lower() in answer_text for kw in test["expected_keywords"])
        
        results.append({
            "question": test["question"],
            "bot_answer": response["answer"],
            "status": "✅ Correct" if is_correct else "❌ Incorrect"
        })

    print("\n" + "="*40 + " RAPPORT D'ÉVALUATION " + "="*40)
    df_results = pd.DataFrame(results)
    for _, row in df_results.iterrows():
        print(f"\n❓ Question : {row['question']}")
        print(f"💬 Réponse IA : {row['bot_answer']}")
        print(f"📊 Statut : {row['status']}")
    
    score = sum(1 for r in results if "✅" in r["status"]) / len(results) * 100
    print(f"\n🎯 Score global de précision : {score:.2f}%")

if __name__ == "__main__":
    run_evaluation()