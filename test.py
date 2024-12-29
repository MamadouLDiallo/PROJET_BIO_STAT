import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Fonction d'importation des données
@st.cache_data(persist=True)
def load_data():
    data = pd.read_excel("Donnnées.xlsx")  # Assurez-vous que le fichier existe dans le répertoire
    return data

# Transformation des variables
def transform_variables(df):
    df_transformed = df.copy()
    binary_columns = [
        "Hypertension Arterielle", 
        "Diabete", 
        "Cardiopathie", 
        "hémiplégie",
        "Paralysie faciale", 
        "Aphasie", 
        "Hémiparésie", 
        "Engagement Cerebral", 
        "Inondation Ventriculaire"
    ]
    for col in binary_columns:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].apply(lambda x: 1 if x == "OUI" else 0)

    if "SEXE" in df_transformed.columns:
        df_transformed["SEXE"] = df_transformed["SEXE"].apply(lambda x: 1 if x == "Homme" else 0)

    if "Evolution" in df_transformed.columns:
        df_transformed["Evolution"] = df_transformed["Evolution"].apply(lambda x: 1 if x == "Deces" else 0)

    if "Traitement" in df_transformed.columns:
        df_transformed["Traitement"] = df_transformed["Traitement"].apply(lambda x: 1 if x == "Thrombolyse" else 0)

    return df_transformed

# Fonction pour la séparation train/test
@st.cache_data(persist=True)
def split(df_transformed):
    y = df_transformed["Evolution"]
    X = df_transformed.drop("Evolution", axis=1)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=1) 
    return X_train, X_test, y_train, y_test
   
# Fonction principale
def main():
    st.title("Etude pronostique de Décès aprés le traitement")

    # Charger les données
    df = load_data()
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Base de données : échantillon de 10 observations")
        st.write(df.sample(10))

    # Transformation des variables
    df_transformed = transform_variables(df)

    # Séparation des données
    X_train, X_test, y_train, y_test = split(df_transformed)

    #Paramètres de recherche par grille
    st.subheader("Voici les meilleurs paramettres du modèle")
    param_grid = {
        'max_depth': ["NONE",1, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    st.sidebar.subheader("Meilleurs paramètres")
    st.write(best_params)


    #Entraîner le modèle avec les meilleurs paramètres
    classifier = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )

    classifier.fit(X_train, y_train)

    # Prédictions
    y_pred = classifier.predict(X_test)

    # Calculer les métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Afficher les résultats
    st.subheader("Résultats")
    st.write(f"**Accuracy**: {accuracy:.3f}")
    st.write(f"**Précision**: {precision:.3f}")
    st.write(f"**Recall**: {recall:.3f}")

    # Graphiques de performance avec un bouton "Exécuter"
    graphes_perf = st.sidebar.multiselect(
        "Graphiques de performance",
        ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
    )
    execute = st.sidebar.button("Affichez les Graphiques")

    if execute:
        plot_perf(graphes_perf, classifier, X_test, y_test)

    # Formulaire pour les données du patient
    st.sidebar.header("Prédiction pour un Nouveau Patient")

    # Création des champs pour toutes les variables
    new_data = {}
    for column in X_train.columns:
        if column == "SEXE":
            new_data[column] = st.sidebar.selectbox(f"{column} :", ["Homme", "Femme"])
        elif column == "Traitement":
            new_data[column] = st.sidebar.selectbox(f"{column} :", ["Thrombolyse", "Chirurgie"])
        elif column in [
            "Hypertension Arterielle", "Diabete", "Cardiopathie", "hémiplégie",
            "Paralysie faciale", "Aphasie", "Hémiparésie", "Engagement Cerebral", "Inondation Ventriculaire"
        ]:
            new_data[column] = st.sidebar.radio(f"{column} :", ["OUI", "NON"])
        elif column in [
            "AGE", "Premiers Signe - Admission à l'hopital",
            "Admission à l'hopital - Prise en charge medicale", "Temps de Suivi après traitement (en jours)"
        ]:
            # Définir les plages et les valeurs par défaut pour les variables numériques
            if column == "AGE":
                new_data[column] = st.sidebar.number_input(f"{column} :", min_value=1)
            elif column == "Premiers Signe - Admission à l'hopital":
                new_data[column] = st.sidebar.number_input(f"{column} (en heures) :", min_value=1)
            elif column == "Admission à l'hopital - Prise en charge medicale":
                new_data[column] = st.sidebar.number_input(f"{column} (en heures) :", min_value=1)
            elif column == "Temps de Suivi après traitement (en jours)":
                new_data[column] = st.sidebar.number_input(f"{column} :", min_value=1)
        else:
            # Champ par défaut pour d'autres colonnes numériques
            new_data[column] = st.sidebar.number_input(f"{column} :", value=1)


    # Transformer les données pour correspondre au modèle
    new_data_transformed = {col: 1 if val in ["OUI", "Homme", "Thrombolyse"] else 0 for col, val in new_data.items() if col not in [
        "AGE", "Premiers Signe - Admission à l'hopital",
        "Admission à l'hopital - Prise en charge medicale", "Temps de Suivi après traitement (en jours)"
    ]}
    # Ajouter les variables numériques directement
    new_data_transformed.update({
        col: val for col, val in new_data.items() if col in [
            "AGE", "Premiers Signe - Admission à l'hopital",
            "Admission à l'hopital - Prise en charge medicale", "Temps de Suivi après traitement (en jours)"
        ]
    })

    # Conversion en DataFrame
    new_data_df = pd.DataFrame([new_data_transformed])

    # Assurez-vous que les colonnes de new_data_df sont dans le même ordre que celles de X_train
    new_data_df = new_data_df[X_train.columns]
    # Bouton pour afficher le résultat de la prédiction
    if st.sidebar.button("Résultat de la Prédiction"):
        prediction = classifier.predict(new_data_df)[0]
        prediction_proba = classifier.predict_proba(new_data_df)[:, 1][0]
        result = "Décédé" if prediction == 1 else "Vivant"

        st.subheader("Résultat de la Prédiction")
        st.write(f"résultat de la prédiction  : {prediction_proba:.0f}")
        st.write(f"Le modèle prédit que le patient est **{result}**.")
        


# Graphiques de performance
def plot_perf(graphes, classifier, X_test, y_test):
    if "Confusion Matrix" in graphes:
        st.subheader("Matrice de confusion")
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="viridis")
        st.pyplot(plt.gcf())

    if "ROC Curve" in graphes:
        st.subheader("Courbe ROC")
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt.gcf())

    if "Precision-Recall Curve" in graphes:
        st.subheader("Courbe Precision-Recall")
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        st.pyplot(plt.gcf())


if __name__ == "__main__":
    main()
