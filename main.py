import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('donnees.csv')

def formalize_data_for_model(df):
    print(df.head())
    df = df.drop(['index'], axis=1)
    df = df.drop(
        ['Pression au niveau mer', 'Rafales sur une période', 'Température', "Précipitations dans la dernière heure",
         "Point de rosée", "Temps présent", "Visibilité horizontale"], axis=1)
    df['Variation de pression en 3 heures'] = df['Variation de pression en 3 heures'].fillna(
        df['Variation de pression en 3 heures'].mean())
    df['Direction du vent moyen 10 mn'] = df['Direction du vent moyen 10 mn'].fillna(
        df['Direction du vent moyen 10 mn'].mean())
    df['Vitesse du vent moyen 10 mn'] = df['Vitesse du vent moyen 10 mn'].fillna(
        df['Vitesse du vent moyen 10 mn'].mean())
    df['Humidité'] = df['Humidité'].fillna(df['Humidité'].mean())
    df['Pression station'] = df['Pression station'].fillna(df['Pression station'].mean())
    df['Periode de mesure de la rafale'] = df['Periode de mesure de la rafale'].fillna(
        df['Periode de mesure de la rafale'].mean())
    df['Précipitations dans les 3 dernières heures'] = df['Précipitations dans les 3 dernières heures'].fillna(
        df['Précipitations dans les 3 dernières heures'].mean())
    df['Température (°C)'] = df['Température (°C)'].fillna(df['Température (°C)'].mean())
    df['consommation'] = df['consommation'].fillna(df['consommation'].mean())
    df['datehour'] = df['datehour'].fillna(df['datehour'].mean())
    df['datemonth'] = df['datemonth'].fillna(df['datemonth'].mean())
    df['Type de tendance barométrique'] = df['Type de tendance barométrique'].fillna(
        df['Type de tendance barométrique'].mean())
    df['Date_Heure'] = pd.to_datetime(df['Date_Heure'])
    df['year'] = df['Date_Heure'].dt.year
    df['month'] = df['Date_Heure'].dt.month
    df['day'] = df['Date_Heure'].dt.day
    df['hour'] = df['Date_Heure'].dt.hour
    df = df.drop(['Date_Heure'], axis=1)

    # Rajoute une colonne pour chaque ligne. Si mois = 1, 2, 3, alors saison = 1, si mois = 4, 5, 6, alors saison = 2, etc.
    df['saison'] = df['month'].apply(
        lambda x: 1 if x in [1, 2, 3] else 2 if x in [4, 5, 6] else 3 if x in [7, 8, 9] else 4)

    # Catégorise la colonne saison
    df['saison'] = df['saison'].astype('category')
    # Drop datemonth et datehour
    df = df.drop(['datemonth', 'datehour'], axis=1)

    # Affiche les colonnes

    # Hot one encoding sur la colonne saison
    df = pd.get_dummies(df, columns=['saison'])
    print(" Process fini : Colonnes = ", df.columns)

    return df

def getPredictionForDF(df):
  df_for_training = formalize_data_for_model(df)

  # Génère X et y. y = colonne consommation
  X = df_for_training.drop(['consommation'], axis=1)
  y = df_for_training['consommation']
  # Créer un décalage pour y, pour prédire la consommation de la journée suivante
  y = y.shift(-1)
  y = y.dropna()
  X = X.shift(1)
  X = X.dropna()
  print("Y shape : ", y.shape)
  print("X shape : ", X.shape)
  # Normalise X et y
  from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X = sc_X.fit_transform(X)

  # Divise le dataset en training set et test set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # XGBoost
  from xgboost import XGBRegressor
  model = XGBRegressor()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  # Affiche les résultats
  print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
  print('Variance score: %.2f' % r2_score(y_test, y_pred))
  # Affiche les résultats sur le training set
  print("Valeurs prédites : ", y_pred)

  return df


predictions = getPredictionForDF(df)
predictions.plot(x='Date_Heure', y=['consommation', 'consommation_pred'], figsize=(15, 5))
plt.show()
