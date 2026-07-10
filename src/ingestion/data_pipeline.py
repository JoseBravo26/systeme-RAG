import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_openagenda_data():
    print("📡 Récupération des données en cours...")
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    all_events = []
    limit = 100
    offset = 0

    while True:
        params = {
            "where": 'location_city:"Paris" AND firstdate_begin >= "2025-07-12"',
            "limit": limit,
            "offset": offset
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        records = data.get("results", [])
        
        if not records:
            break # Plus d'événements à récupérer
            
        all_events.extend(records)
        print(f"✅ {len(all_events)} événements récupérés jusqu'à présent...")
        
        if len(records) < limit:
            break # C'était la dernière page
            
        offset += limit # On passe à la page suivante

    return pd.DataFrame(all_events)

def clean_events_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Ajout des bonnes colonnes de date et de la description longue
    cols_to_keep = ['uid', 'title_fr', 'description_fr', 'longdescription_fr', 
                    'location_city', 'location_department', 'location_postalcode', 'firstdate_begin', 'lastdate_end']
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df_clean = df[existing_cols].copy()

    # On s'assure d'avoir au moins une description
    df_clean = df_clean.dropna(subset=['description_fr'])

    # Création du texte riche pour le RAG
    df_clean['text_to_embed'] = df_clean.apply(
        lambda row: f"Titre: {row.get('title_fr', '')}\n"
                    f"Lieu: {row.get('location_city', '')} ({row.get('location_department', '')})\n"
                    f"Date: du {row.get('firstdate_begin', '')} au {row.get('lastdate_end', '')}\n"
                    f"Description: {row.get('longdescription_fr', row.get('description_fr', ''))}", 
        axis=1
    )
    
    # Nettoyage basique du HTML
    df_clean['text_to_embed'] = df_clean['text_to_embed'].str.replace('<p>', '').str.replace('</p>', '').str.replace('<br>', '\n')

    print(f"✅ Nettoyage terminé : {len(df_clean)} événements conservés.")
    return df_clean

if __name__ == "__main__":
    df_events_bruts = fetch_openagenda_data()
    
    if not df_events_bruts.empty:
        # Il manquait l'appel à la fonction de nettoyage ici !
        df_clean = clean_events_data(df_events_bruts)
        
        os.makedirs("data", exist_ok=True)
        # On sauvegarde le CSV en utf-8-sig pour les accents
        df_clean.to_csv("data/evenements_clean.csv", encoding="utf-8-sig", index=False)
        print("💾 Données nettoyées et sauvegardées dans 'data/evenements_clean.csv'")