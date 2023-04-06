from datetime import timedelta, datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('donnees.csv')


def formalize_data_for_model(df):
    print('Lancement du processus de formalisation des données')
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
    print('DF COLUMNS FOR FORMATTING : ', df.columns)
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
    # Si il manque saison_1, saison_2, saison_3, ou saison_4, alors on les rajoute
    if 'saison_1' not in df.columns:
        df['saison_1'] = 0
    if 'saison_2' not in df.columns:
        df['saison_2'] = 0
    if 'saison_3' not in df.columns:
        df['saison_3'] = 0
    if 'saison_4' not in df.columns:
        df['saison_4'] = 0
    return df


def getPredictionForDF(df, shiftStep, filename):
    df_for_training = formalize_data_for_model(df)
    # Génère X et y. y = colonne consommation
    X = df_for_training.drop(['consommation'], axis=1)
    y = df_for_training['consommation']
    # Créer un décalage pour y, pour prédire la consommation de la journée suivante
    y = y.shift(-shiftStep)
    y = y.dropna()
    X = X.shift(shiftStep)
    X = X.dropna()
    print("Y shape : ", y.shape)
    print("X shape : ", X.shape)
    # Normalise X et y
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    # Divise le dataset en training set et test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
    # Sauvegarde le modèle
    import pickle
    pickle.dump(model, open(filename, 'wb'))

    return df


def getPredictionForDFUsingRealModel(df, model_filename, shiftStep):
    df_for_training = formalize_data_for_model(df)
    print('Formattage des données finis.')
    import pickle
    loaded_model = pickle.load(open(model_filename, 'rb'))

    # Génère X et y. y = colonne consommation
    X = df_for_training.drop(['consommation'], axis=1)
    y = df_for_training['consommation']
    print("Y shape : ", y.shape)
    print("X shape : ", X.shape)
    # Normalise X et y
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    y_pred = loaded_model.predict(X)

    print("Valeurs prédites : ", y_pred)
    print(y_pred.shape)
    #Add as many lines at the beginning of the y_pred as ShiftStep, as nan. This is because of the shift
    for i in range(shiftStep):
        y_pred = np.insert(y_pred, 0, np.nan)

    print(y_pred.shape)
    print(y_pred)
    #Drop Nan values
    y_pred = y_pred[~np.isnan(y_pred)]

    return y_pred

#Créer le modele
#dummy = getPredictionForDF(df, 112, "twoWeeksModel.sav")
#dummy = getPredictionForDF(df, 56, "oneWeekModel.sav")
#dummy = getPredictionForDF(df, 8, "oneDayModel.sav")

#predictions = getPredictionForDFUsingRealModel(df, 'oneDayModel.sav', 8)
# Add the predictions to the dataframe. Due to the shift, the first value is NaN
#df['consommation_pred'] = predictions
# Plot the predictions
#df.plot(x='Date_Heure', y=['consommation', 'consommation_pred'], figsize=(15, 5))

def getPredictForAllDaysInNextWeek():
    # Get the prediction for the next day
    df = pd.read_csv('donnees.csv')
    # Keep only the last 56 entries
    df = df.tail(56)

    predictions = getPredictionForDFUsingRealModel(df, 'oneWeekModel.sav', 56)
    # Clone le dataframe
    df2 = df.copy()
    # Set consommation to nan
    df2['consommation'] = np.nan
    # Ajoute 7 jours à la date
    df2['Date_Heure'] = df2['Date_Heure'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + timedelta(days=7))
    # Ajoute 3 heures à la date
    df2['Date_Heure'] = df2['Date_Heure'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') + timedelta(hours=4))
    # Ajoute les prédictions
    df2['consommation_pred'] = predictions
    # Concatène les deux dataframes
    df = pd.concat([df, df2])
    print("Predictions pour la semaine prochaine : ", predictions)
    # Print la date et la consommation_pred
    for index, row in df.iterrows():
        print(row['Date_Heure'], " : ", row['consommation_pred'])
    return df

def getPredictForAllDaysInNextWeekTest():
    # Get the prediction for the next day
    print('salut')
    df = pd.read_csv('donnees.csv')
    # Keep only the last 56 entries
    # Il y a x lignes pour une seule journée. Rassemble moi toutes les données pour la meme journée en utilisant la moyenne
    df = df.groupby(df['Date_Heure'].str[:10]).mean()
    print(" NEW DF : ", df)
    # Ne garde que les 7 derniers jours
    df = df.tail(7)
    print(df)
    predictions = getPredictionForDFUsingRealModel(df, 'oneDayModel.sav', 8)
    # Clone le dataframe
    print('predictions : ', predictions)
    #df2 = df.copy()
    # Set consommation to nan
    #df2['consommation'] = np.nan
    # Ajoute 7 jours à la date
    #df2['Date_Heure'] = df2['Date_Heure'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + timedelta(days=7))
    # Ajoute 3 heures à la date
    #df2['Date_Heure'] = df2['Date_Heure'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') + timedelta(hours=4))
    # Ajoute les prédictions
    #df2['consommation_pred'] = predictions
    # Concatène les deux dataframes
    #df = pd.concat([df, df2])





    # Plot the predictions
    #df.plot(x='Date_Heure', y=['consommation', 'consommation_pred'], figsize=(15, 5))
    #plt.show()
    #print("Predictions pour la semaine prochaine : ", predictions)
    ## Print la date et la consommation_pred
    #for index, row in df.iterrows():
    #    print(row['Date_Heure'], " : ", row['consommation_pred'])


getPredictForAllDaysInNextWeek()

def getPredictForNextDay():
# Get the prediction for the next day
    df = pd.read_csv('donnees.csv')
    # Keep only the last 8 entries
    df = df.tail(8)

    predictions = getPredictionForDFUsingRealModel(df, 'oneDayModel.sav', 8)
    # Clone le dataframe
    df2 = df.copy()
    # Set consommation to nan
    df2['consommation'] = np.nan
    # Ajoute 1 jours à la date
    df2['Date_Heure'] = df2['Date_Heure'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + timedelta(days=1))
    # Ajoute les prédictions
    df2['consommation_pred'] = predictions
    # Concatène les deux dataframes
    df = pd.concat([df, df2])
    print("Predictions pour demain : ", predictions)
    # Print la date et la consommation_pred
    for index, row in df.iterrows():
        print(row['Date_Heure'], " : ", row['consommation_pred'])
    return df

print(getPredictForNextDay())