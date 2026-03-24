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
            st.success("API connectée")
        else:
            st.error("API non disponible")
    except:
        st.error("API inaccessible")
    
    st.divider()
    
    # Génération des données
    if st.button("Générer des données aléatoires", width='stretch'):
        st.session_state.fake_data = np.random.uniform(0, 1, 60).tolist()
        st.session_state.result = None
        st.success("Données générées !")
    
    # Aperçu des données
    if st.session_state.fake_data is not None:
        st.divider()
        st.subheader("Aperçu du signal (60 fréquences)")
        preview_df = pd.DataFrame({
            "Fréquence": range(1, 61),
            "Amplitude": st.session_state.fake_data
        })
        st.dataframe(preview_df.head(10), use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)
    
    # Bouton d'analyse
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
                    
            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API. Vérifiez que FastAPI est démarré.")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Affichage des résultats
    if st.session_state.result:
        st.divider()
        st.subheader("Résultats de l'analyse")
        
        res = st.session_state.result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_color = "🔴" if res["prediction"] == "M" else "🟢"
            st.metric(
                label="Prédiction", 
                value=f"{prediction_color} {res['prediction']} - {res['description']}"
            )
        with col2:
            st.metric("Confiance", f"{res['confidence']*100:.2f}%")
        with col3:
            st.metric("Statut", "Analyse terminée")
        
        # Graphique des probabilités
        st.subheader("Distribution des probabilités")
        proba_df = pd.DataFrame({
            'Classe': list(res['probabilities'].keys()),
            'Probabilité': list(res['probabilities'].values())
        })
        
        fig_proba = go.Figure()
        fig_proba.add_trace(go.Bar(
            x=proba_df['Classe'],
            y=proba_df['Probabilité'],
            text=[f"{p:.1%}" for p in proba_df['Probabilité']],
            textposition='auto',
            marker_color=['#ff4b4b', '#4caf50']
        ))
        fig_proba.update_layout(
            title="Probabilités de classification",
            yaxis_title="Probabilité",
            yaxis_range=[0, 1],
            height=400
        )
        st.plotly_chart(fig_proba, width='stretch')
        
else:
    st.info("Générez des données dans la barre latérale pour commencer l'analyse")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <small> Modèle RandomForest entraîné sur dataset sonar | Interface Streamlit + API FastAPI</small>
</div>
""", unsafe_allow_html=True)