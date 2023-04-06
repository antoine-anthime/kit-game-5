import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Import getPredictForAllDaysInNextWeek() from main2.py
from main2 import getPredictForAllDaysInNextWeek
from main2 import getPredictForNextDay


dfNextDay = getPredictForNextDay()


# Affiche logo.png
st.image("logo.png")
# Ecrit "Hello World" dans le titre de la page
st.title("KIT GAME - Equipe 1")

#Ecrit "Prédictions de la consommation électrique" dans le titre de la page
st.title("Prédictions de la consommation électrique")


if st.button("Charger la prédiction pour la semaine à venir"):
    # Affiche le dataframe
    # Affiche le graphique
    df = getPredictForAllDaysInNextWeek()
    # Set Date_Heure as datetime
    df['Date_Heure'] = pd.to_datetime(df['Date_Heure'])
    # A chaque ligne ou consommation_pred n'est pas nul, on rajoute une colonne avec la valeur de consommation_pred + 10% de consommation_pred, et on l'appelle consommation_pred_sup
    df['consommation_pred_sup'] = df.apply(lambda row: row['consommation_pred'] + (row['consommation_pred'] * 0.1) if pd.notnull(row['consommation_pred']) else None, axis=1)
    # A chaque ligne ou consommation_pred n'est pas nul, on rajoute une colonne avec la valeur de consommation_pred - 10% de consommation_pred, et on l'appelle consommation_pred_inf
    df['consommation_pred_inf'] = df.apply(lambda row: row['consommation_pred'] - (row['consommation_pred'] * 0.1) if pd.notnull(row['consommation_pred']) else None, axis=1)
    #Afficher le grafe de consommation et consommation pred en fonction de la date. Consommation_pred_sup et consommation_pred_inf sont en rouge, et la zone entre les deux est en bleu
    plt.figure(figsize=(10, 5))
    #Retrouver la dernière ligne ou consommation n'est pas nul
    last_row = df[df['consommation'].notna()].iloc[-1]
    #Retrouver la premiere ligne ou consommation_pred n'est pas nul
    first_row = df[df['consommation_pred'].notna()].iloc[0]
    print("LAST ROW : ", last_row)
    last_row['consommation_pred'] = last_row['consommation']
    plt.plot(df['Date_Heure'], df['consommation'], label='consommation')
    plt.plot(df['Date_Heure'], df['consommation_pred'], label='consommation_pred')
    plt.fill_between(df['Date_Heure'], df['consommation_pred_sup'], df['consommation_pred_inf'], color='blue', alpha=0.2)
    #Ajouter en légende la zone bleue en "Zone de prédiction"
    plt.legend(['consommation', 'consommation_pred', 'Zone de prédiction'])
    plt.show()
    plt.legend()



    #plt.figure(figsize=(10, 5))
    #df.plot(x='Date_Heure', y=['consommation', 'consommation_pred'], figsize=(15, 5))

    plt.legend()

    st.pyplot(plt)
    print("Colonnes pour df : ", df.columns)
    #Affiche un nombre
    #Calcul la moyenne de la consommation et de la consommation_pred, affiche en gros
    st.write("Moyenne de la consommation : ", df['consommation'].mean())
    st.write("Moyenne de la consommation prédite pour la semaine entière : ", df['consommation_pred'].mean())
    #Create a df where only the consommation_pred is not null
    df2 = df[df['consommation_pred'].notna()]
    #Group by day and get the mean for each day, without using str as it returns an error
    #Convert the date to datetime
    df2['Date_Heure'] = pd.to_datetime(df2['Date_Heure'])
    #Group by day
    df2 = df2.groupby(df2['Date_Heure'].dt.date).mean()
    print('DF APRES FILTRE : ', df2)
    #Afficher la moyenne pour chaque jour de la semaine
    st.write("Moyenne de la consommation prédite pour J+1 : ", df2['consommation_pred'].iloc[0], " pour le ", df2.index[0])
    st.write("Moyenne de la consommation prédite pour J+2 : ", df2['consommation_pred'].iloc[1], " pour le ", df2.index[1])
    st.write("Moyenne de la consommation prédite pour J+3 : ", df2['consommation_pred'].iloc[2], " pour le ", df2.index[2])
    st.write("Moyenne de la consommation prédite pour J+4 : ", df2['consommation_pred'].iloc[3], " pour le ", df2.index[3])
    st.write("Moyenne de la consommation prédite pour J+5 : ", df2['consommation_pred'].iloc[4], " pour le ", df2.index[4])
    st.write("Moyenne de la consommation prédite pour J+6 : ", df2['consommation_pred'].iloc[5], " pour le ", df2.index[5])
    st.write("Moyenne de la consommation prédite pour J+7 : ", df2['consommation_pred'].iloc[6], " pour le ", df2.index[6])

    #Télécharger la dataframe en csv
    st.download_button(
        label="Télécharger les données brutes de prédiction au format CSV",
        data=df.to_csv(),
        file_name='prediction.csv',
        mime='text/csv'
    )
    #Afficher le dataframe
    st.dataframe(df)

# Fais un bouton "Charger la prédiction pour demain"
if st.button("Charger la prédiction pour demain"):
    # Affiche le dataframe
    dfNextDay.plot(x='Date_Heure', y=['consommation', 'consommation_pred'], figsize=(15, 5))
    # Affiche le graphique
    st.pyplot(plt)
    st.dataframe(dfNextDay)

    #Calcul la moyenne de la consommation et de la consommation_pred, affiche en gros
    st.write("Moyenne de la consommation pour la journée d'hier: ", dfNextDay['consommation'].mean())
    st.write("Moyenne de la consommation prédite pour demain : ", dfNextDay['consommation_pred'].mean())

    #Télécharger la dataframe en csv
    st.download_button(
        label="Télécharger les données brutes de prédiction au format CSV",
        data=dfNextDay.to_csv(),
        file_name='prediction.csv',
        mime='text/csv'
    )

    #Afficher le dataframe
    st.dataframe(dfNextDay)


