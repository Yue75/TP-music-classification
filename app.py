import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pretraitement import nettoyer_texte

SEUIL = 0.40       # en dessous de cette probabilité, on prévient que le modèle hésite
MOTS_MINIMUM = 20  # en dessous de ce nombre de mots, on prévient que le texte est court

exemples = {
    "(aucun)": "",
    "Metal": "Darkness falls upon the burning souls, blood and fire, the demons rise from hell, "
             "death will reign eternal in the kingdom of the damned, screaming in the void",
    "Jazz": "The moon is shining, my darling, dance with me tonight, so lovely, so heavenly, "
            "your sweet romance, a thrill I can't resist, sentimental dreams of you and me",
    "Rock": "Well all right, she's a woman on the highway, jack is drivin' all night long, "
            "rock and roll is here to stay, say you will, babe",
    "Folk": "Down by the river in the little town, my darling with a ribbon in her hair, "
            "we walked past the graveyard gates on a bright morning",
    "Pop (portugais)": "Meu amor, eu sinto saudade de você, a nossa paixão não vai acabar, "
                       "quero te ver todo dia, coração apaixonado",
}

st.set_page_config(page_title="Genre musical", page_icon="🎵")


@st.cache_resource   # le modèle n'est chargé qu'une seule fois
def charger_modele():
    with open("models/metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    modele = joblib.load("models/" + metadata["fichier"])
    return modele, metadata


def mots_importants(texte, modele, genre):
    # Pour chaque mot du texte : son poids TF-IDF x son coefficient pour le genre prédit
    tfidf = modele.named_steps["tfidf"]
    classifier = modele.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    i = list(classifier.classes_).index(genre)
    contributions = tfidf.transform([texte]).toarray()[0] * classifier.coef_[i]
    feature_names = tfidf.get_feature_names_out()
    top_indices = contributions.argsort()[::-1][:8]
    return [feature_names[j] for j in top_indices if contributions[j] > 0]


modele, metadata = charger_modele()

# ---------- Barre latérale : présentation du modèle
with st.sidebar:
    st.header("À propos du modèle")
    st.write("**Modèle :**", metadata["nom_modele"])
    st.write("**Chansons d'entraînement :**", str(metadata["nb_chansons_entrainement"]))
    col1, col2 = st.columns(2)
    col1.metric("F1 macro (test)", f"{metadata['scores_test']['F1 macro']:.2f}")
    col2.metric("Accuracy (test)", f"{metadata['scores_test']['Accuracy']:.0%}")
    st.write("**Genres possibles :**", ", ".join(metadata["classes"]))
    st.caption("« Autres » regroupe : " + ", ".join(metadata["genres_regroupes_dans_autres"]) + ".")
    st.caption("Données : " + metadata["dataset"] + " (Hugging Face). Paroles en anglais, portugais, roumain ou allemand.")
    with st.expander("Comparaison des modèles testés"):
        comparaison = pd.read_csv("models/comparaison_modeles.csv")
        st.dataframe(comparaison[["Modèle", "Famille", "F1 macro", "Accuracy"]], hide_index=True)

# ---------- Page principale
st.title("🎵 Quel est le genre de cette chanson ?")
st.write("Collez les **paroles** d'une chanson, ou importez un fichier texte : "
         "le modèle devine son **genre musical** et donne la probabilité associée.")

onglet_texte, onglet_fichier = st.tabs(["✍️ Écrire / coller", "📄 Importer un fichier .txt"])

with onglet_texte:
    choix = st.selectbox("Charger un exemple (facultatif)", list(exemples))
    texte_saisi = st.text_area("Paroles", value=exemples[choix], height=220,
                               placeholder="Collez ici les paroles de la chanson...")

with onglet_fichier:
    fichier = st.file_uploader("Fichier texte contenant les paroles", type=["txt"])
    texte_fichier = ""
    if fichier is not None:
        contenu = fichier.getvalue()
        try:
            texte_fichier = contenu.decode("utf-8")
        except UnicodeDecodeError:
            texte_fichier = contenu.decode("latin-1")
        st.text_area("Aperçu du fichier", texte_fichier[:2000], height=150, disabled=True)

# Si un fichier est importé, on utilise son texte, sinon le texte saisi
texte = texte_fichier if texte_fichier.strip() else texte_saisi

if st.button("Prédire le genre", type="primary"):

    nb_mots = len(nettoyer_texte(texte).split())

    if nb_mots == 0:
        st.error("Le texte est vide. Écrivez ou collez des paroles.")
    else:
        if nb_mots < MOTS_MINIMUM:
            st.warning(f"Le texte ne contient que {nb_mots} mots : la prédiction sera peu fiable.")

        probas = modele.predict_proba([texte])[0]
        probas = pd.Series(probas, index=modele.classes_).sort_values(ascending=False)
        genre = probas.index[0]
        proba = probas.iloc[0]

        st.subheader("Résultat")
        col1, col2 = st.columns(2)
        col1.metric("Genre prédit", genre)
        col2.metric("Probabilité", f"{proba:.1%}")

        if proba < SEUIL:
            st.info(f"Le modèle hésite avec **{probas.index[1]}** ({probas.iloc[1]:.1%}). "
                    "Les paroles de Pop, Rock, Indie et Folk se ressemblent beaucoup.")

        st.write("**Probabilité de chaque genre :**")
        tableau = pd.DataFrame({"Genre": probas.index, "Probabilité": probas.values})
        graphique = alt.Chart(tableau).mark_bar().encode(
            x=alt.X("Probabilité", axis=alt.Axis(format="%")),
            y=alt.Y("Genre", sort="-x", title=None),   # du plus probable au moins probable
        )
        st.altair_chart(graphique, use_container_width=True)

        mots = mots_importants(texte, modele, genre)
        if mots:
            st.write(f"**Mots qui orientent vers {genre} :**", ", ".join(mots))

st.caption("Projet de Machine Learning (NLP). Le genre dépend aussi de la musique : "
           "les paroles seules ne suffisent pas toujours.")
