import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Sonar Detector", page_icon="⚓", layout="wide")

# Configuration de l'URL de l'API (locale dans le conteneur)
API_URL = "http://localhost:8000"

# 1. Initialisation du session_state
if "fake_data" not in st.session_state:
    st.session_state.fake_data = None
if "result" not in st.session_state:
    st.session_state.result = None
if "explanation_img" not in st.session_state:
    st.session_state.explanation_img = None

st.title("⚓ Détection de Mines sous-marines")
st.markdown("""
Analyse des signaux sonar pour différencier les **Roches** des **Mines**.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration du Signal")
    
    # Vérification de l'API
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        if response.status_code == 200:
            st.markdown("🟢 **API : Connectée**")
        else:
            st.markdown("🟠 **API : Erreur Partielle**")
    except:
        st.markdown("🔴 **API : Hors ligne**")
    
    st.divider()
    
    # Génération des données
    if st.button("Générer des données aléatoires", width='stretch'):
        st.session_state.fake_data = np.random.uniform(0, 1, 60).tolist()
        st.session_state.result = None
        st.session_state.explanation_img = None
        st.success("Données générées !")
    
    # Aperçu des données
    if st.session_state.fake_data is not None:
        st.divider()
        st.subheader(" 📊 Aperçu du signal (60 fréquences)")
        preview_df = pd.DataFrame({
            "Fréquence": range(1, 61),
            "Amplitude": st.session_state.fake_data
        })
        st.dataframe(preview_df.head(10), width='stretch')

# --- MAIN PAGE ---
if st.session_state.fake_data is not None:
    # Visualisation du signal
    st.subheader(" Visualisation du Signal Sonar")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.fake_data,
        mode='lines+markers', 
        name='Signal',
        line=dict(color='cyan', width=2),
        marker=dict(size=4, color='yellow')
    ))
    fig.update_layout(
        title="Signal Sonar - 60 fréquences",
        xaxis_title="Fréquence",
        yaxis_title="Amplitude",
        height=400,
        template="plotly_dark",
        dragmode=False,
        hovermode='x'
    )
    # Correction width
    st.plotly_chart(fig, width='stretch')
    
    if st.button("Lancer l'analyse via l'API", type="primary", width='stretch'):
        with st.spinner("Analyse en cours via l'API..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict", 
                    json={"features": st.session_state.fake_data},
                    timeout=10
                )
                
                if response.status_code == 200:
                    st.session_state.result = response.json()
                    st.success("Analyse terminée !")
                else:
                    st.error(f"Erreur API: {response.status_code}")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Affichage des résultats
    if st.session_state.result:
        st.divider()
        res = st.session_state.result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            icon = "🔴" if res["prediction"] == "M" else "🟢"
            st.metric(
                "Prédiction", 
                f"{icon} {res['prediction']}"
            )
        with col2:
            st.metric("Confiance", f"{res['confidence']*100:.2f}%")
        with col3:
            st.metric("Description", res['description'])
        
        # Graphique des probabilités
        fig_proba = go.Figure(go.Bar(
            x=list(res['probabilities'].keys()),
            y=list(res['probabilities'].values()),
            marker_color=['#ff4b4b', '#4caf50']
        ))
        fig_proba.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_proba, width='stretch')

        st.divider()
        st.subheader("🔍 Analyse de l'Importance des Fréquences")
        st.write("Quelles fréquences sont les plus déterminantes pour le modèle en général ?")
        
        if st.button("Afficher l'importance des capteurs", width='stretch'):
            with st.spinner("Récupération des données du modèle..."):
                try:
                    import_res = requests.get(f"{API_URL}/importance")
                    if import_res.status_code == 200:
                        data_imp = import_res.json()["feature_importance"]
                        
                        # Création du DataFrame pour Plotly
                        df_imp = pd.DataFrame({
                            'Fréquence': list(data_imp.keys()),
                            'Importance': list(data_imp.values())
                        }).sort_values(by='Importance', ascending=False).head(15) # Top 15 pour la clarté

                        # Graphique Plotly Interactif
                        fig_imp = go.Figure(go.Bar(
                            x=df_imp['Importance'],
                            y=df_imp['Fréquence'],
                            orientation='h',
                            marker=dict(
                                color=df_imp['Importance'],
                                colorscale='Viridis'
                            )
                        ))
                        
                        fig_imp.update_layout(
                            title="Top 15 des fréquences les plus discriminantes",
                            xaxis_title="Score d'importance",
                            yaxis_title="Capteurs (C1-C60)",
                            template="plotly_dark",
                            height=450,
                            yaxis={'categoryorder':'total ascending'}
                        )
                        
                        st.plotly_chart(fig_imp, width='stretch')
                        st.info("Ce graphique montre que certaines plages de fréquences impactent plus l'algorithme que d'autres lors de la détection.")
                    else:
                        st.error("Erreur lors de la récupération des importances.")
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")

else:
    st.info("Utilisez la barre latérale pour générer un signal.")

st.divider()
st.markdown('<div style="text-align: center; color: gray;"><small> Sonar Project AI | Random Forest Classifier</small></div>', unsafe_allow_html=True)