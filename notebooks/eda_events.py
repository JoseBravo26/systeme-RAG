import pandas as pd
import plotly.express as px

def generer_graphiques():
    print("📊 Chargement des données locales...")
    # 1. Lecture directe depuis le fichier CSV nettoyé
    # (Ajustez le chemin selon l'endroit d'où vous exécutez le script)
    df = pd.read_csv('data/evenements_clean.csv', encoding='utf-8-sig')
    
    # 2. Préparation des données temporelles
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce')
    df_time = df.dropna(subset=['firstdate_begin']).copy()
    df_time['month_year'] = df_time['firstdate_begin'].dt.to_period('M').astype(str)
    time_counts = df_time.groupby('month_year').size().reset_index(name='count').sort_values('month_year')

    # 3. Génération du graphique temporel
    fig_time = px.line(time_counts, x='month_year', y='count', markers=True)
    fig_time.update_traces(fill='tozeroy', fillcolor='rgba(0,100,200,0.1)')
    fig_time.update_layout(
        title="Évolution temporelle des événements à Paris",
        xaxis_title="Mois",
        yaxis_title="Nombre d'événements"
    )
    # Enregistrement du graphique en tant qu'image dans le dossier 'data'
    fig_time.write_image("data/distribution_temporelle.png")
    print("✅ Graphique temporel généré avec succès.")

    # 4. Préparation et génération du graphique spatial
    # Si la colonne location_postalcode existe bien dans votre CSV
    if 'location_postalcode' in df.columns:
        df['postal_code'] = df['location_postalcode'].astype(str).str.extract(r'(\d{5})')
        df_paris = df[df['postal_code'].str.startswith('75', na=False)].copy()
        df_paris['arrondissement'] = df_paris['postal_code'].str[-2:].astype(int).astype(str)
        
        geo_counts = df_paris.groupby('arrondissement').size().reset_index(name='count')
        geo_counts = geo_counts.sort_values('count', ascending=False).head(10)
        
        fig_geo = px.bar(geo_counts, x='arrondissement', y='count')
        fig_geo.update_layout(
            title="Top 10 des arrondissements les plus actifs",
            xaxis_title="Arrondissement (Paris)",
            yaxis_title="Nombre d'événements"
        )
        fig_geo.write_image("data/distribution_spatiale.png")
        print("✅ Graphique spatial généré avec succès.")
    else:
        print("⚠️ La colonne 'location_postalcode' est manquante dans le CSV.")

if __name__ == "__main__":
    generer_graphiques()