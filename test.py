import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

scaler = RobustScaler()

# Fonction d'importation des données
@st.cache_data(persist=True)
def load_data():
    data = pd.read_excel("Donnnées.xlsx")  # Assurez-vous que le fichier existe dans le répertoire
    return data

# Transformation des variables
def transform_variables(df):
    df_transformed = df.drop("hémiplégie", axis=1)
    binary_columns = [
        "Hypertension Arterielle",
        "Diabete",
        "Cardiopathie",
        "Paralysie faciale",
        "Aphasie",
        "Hémiparésie",
        "Engagement Cerebral",
        "Inondation Ventriculaire",
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
    st.title("Etude pronostique de Décès après le traitement")

    # Charger les données
    df = load_data()
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Base de données : échantillon de 10 observations")
        st.write(df.sample(10))

    # Transformation des variables
    df_transformed = transform_variables(df)

    # Détection des colonnes numériques
    numCols = df_transformed.select_dtypes(include=np.number).columns.tolist()

    # Séparation des données
    X_train, X_test, y_train, y_test = split(df_transformed)
    st.write(f"Colonnes disponibles dans X_train : {X_train.columns.tolist()}")
    st.write(f"Colonnes sélectionnées pour la normalisation : {numCols}")

    # Normalisation des données
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numCols] = scaler.fit_transform(X_train[numCols])
    X_test_scaled[numCols] = scaler.transform(X_test[numCols])

    # Charger le modèle pré-entraîné
    model = None
    try:
        model = joblib.load("model.pkl")
        st.success("Modèle chargé avec succès !")
    except FileNotFoundError:
        st.error("Erreur : le fichier 'best_model.pkl' est introuvable.")
        return
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return

    # Prédictions et résultats
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.subheader("Résultats")
    st.write(f"**Accuracy**: {accuracy:.3f}")
    st.write(f"**Précision**: {precision:.3f}")
    st.write(f"**Recall**: {recall:.3f}")

    # Graphiques de performance
    graphes_perf = st.sidebar.multiselect(
        "Graphiques de performance", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
    )
    if st.sidebar.button("Afficher les Graphiques"):
        plot_perf(graphes_perf, model, X_test_scaled, y_test)

    # Prédiction pour un nouveau patient
    st.sidebar.header("Prédiction pour un Nouveau Patient")
    new_data = {}
    for col in X_train.columns:
        if col in binary_columns:
            new_data[col] = st.sidebar.radio(f"{col} :", ["OUI", "NON"])
        elif col == "SEXE":
            new_data[col] = st.sidebar.selectbox(f"{col} :", ["Homme", "Femme"])
        elif col == "Traitement":
            new_data[col] = st.sidebar.selectbox(f"{col} :", ["Thrombolyse", "Chirurgie"])
        else:
            new_data[col] = st.sidebar.number_input(f"{col} :", min_value=0)

    # Transformation et prédiction
    try:
        new_data_transformed = {col: 1 if val in ["OUI", "Homme", "Thrombolyse"] else 0 for col, val in new_data.items()}
        new_data_df = pd.DataFrame([new_data_transformed])
        new_data_df[numCols] = scaler.transform(new_data_df[numCols])
        new_data_df = new_data_df.reindex(columns=X_train_scaled.columns, fill_value=0)

        if st.sidebar.button("Résultat de la Prédiction"):
            prediction = model.predict(new_data_df)[0]
            result = "Décédé" if prediction == 1 else "Vivant"
            st.subheader("Résultat de la Prédiction")
            st.write(f"Le modèle prédit que le patient est **{result}**.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# Graphiques de performance
def plot_perf(graphes, model, X_test_scaled, y_test):
    plt.clf()  # Clear any previous plot

    if "Confusion Matrix" in graphes:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="viridis")
        st.pyplot(plt.gcf())

    if "ROC Curve" in graphes:
        st.subheader("Courbe ROC")
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt.gcf())

    if "Precision-Recall Curve" in graphes:
        st.subheader("Courbe Precision-Recall")
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()
