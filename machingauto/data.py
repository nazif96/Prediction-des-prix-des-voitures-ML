import pandas as pd

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Nettoyage et préparation des données
    df = df.drop('Date', axis=1)
    df['Évaluations'] = df['Évaluations'].replace('Évaluations non disponibles', 0).astype(float)
    df['Puissance_CH'] = df['Puissance'].str.extract('(\d+\.?\d*) CH').astype(float)
    df['Prix'] = df['Prix'].str.replace('€', '').str.replace(' ', '').str.replace(',', '.').astype(float)
    df['Kilométrage'] = df['Kilométrage'].str.replace('km', '').str.replace(' ', '').str.replace(',', '.').str.replace('- ', '0').astype(float)
    df['Carburant'] = df['Carburant'].replace(['- Carburant','CNG'], 'Autre')
    df['Transmission'] = df['Transmission'].replace(['- Boîte', 'Boite non disponible'], 'Autre')
    new_df = df.drop(['Nom de la Voiture', 'Version','Vendeur' ], axis=1)
    df['Puissance'] = df['Puissance'].replace(['- kW (- CH)'], 'Autre')
    df_1 = pd.get_dummies(new_df, columns=['Modèle','Carburant','Transmission'])  
    df_new1 = df_1.drop(['Carburant_Autres', 'Puissance'], axis=1)
    df_encoded= df_new1.dropna(subset=['Puissance_CH']) 
    df_encoded = df_encoded.drop('Carburant_Autre', axis=1)
    df_encoded = df_encoded.drop('Transmission_Autre', axis=1)     

    

    return df_encoded





