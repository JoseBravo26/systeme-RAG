import pandas as pd
from src.ingestion.data_pipeline import clean_events_data

def test_clean_events_data_drops_missing_description():
    """
    Vérifie que la fonction de nettoyage supprime bien les événements sans description
    et génère correctement la colonne 'text_to_embed'.
    """
    # Jeu de données fictif de test
    mock_data = pd.DataFrame({
        'uid': ['1', '2'],
        'title_fr': ['Concert Jazz', 'Expo Peinture'],
        'description_fr': ['Un super concert', None], # Le 2ème doit être supprimé
        'location_city': ['Paris', 'Lyon'],
        'location_department': ['Paris', 'Rhône'],
        'date_start': ['2026-08-01', '2026-09-01'],
        'date_end': ['2026-08-01', '2026-09-15']
    })
    
    df_clean = clean_events_data(mock_data)
    
    # Vérifications
    assert len(df_clean) == 1, "La ligne sans description aurait dû être supprimée."
    assert 'text_to_embed' in df_clean.columns, "La colonne text_to_embed n'a pas été créée."
    assert "Concert Jazz" in df_clean.iloc[0]['text_to_embed'], "Le texte combiné ne contient pas le titre."