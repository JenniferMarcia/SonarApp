import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Sonar Detector", page_icon="⚓", layout="wide")


st.markdown("""
<style>
    /* Bouton principal personnalisé */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 198, 255, 0.5);
    }
    /* Harmonisation des métriques */
    [data-testid="stMetricValue"] {
        color: #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

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
        mode='lines', 
        fill='tozeroy',
        fillcolor='rgba(0, 242, 254, 0.1)',
        name='Amplitude du Signal'
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        xaxis=dict(showgrid=False, title="Index des Capteurs (60)"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Amplitude"),
        hovermode="x unified"
    )
    # Correction width
    st.plotly_chart(fig, width='stretch',config={'displayModeBar': False})
    
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
        color = "#ff4b4b" if res["prediction"] == "M" else "#4caf50"
        st.markdown(f"""
            <div style="background-color: {color}22; border: 1px solid {color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
                <h2 style="color: {color}; margin: 0;">Résultat : {res['description']}</h2>
                <p style="margin: 0; opacity: 0.8;">Indice de confiance : {res['confidence']*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Probabilités")
            fig_proba = go.Figure(go.Bar(
                x=list(res['probabilities'].keys()),
                y=list(res['probabilities'].values()),
                marker_color=[ '#ff4b4b', '#4caf50'],
                text=[f"{v*100:.1f}%" for v in res['probabilities'].values()],
                textposition='auto',
            ))
            fig_proba.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_proba, width='stretch', config={'displayModeBar': False})
            
        with col2:
            st.subheader("🎯 Indicateurs Clés")
            st.metric("Confiance", f"{res['confidence']*100:.2f}%", help="Certitude du modèle")
            st.metric("Type détecté", res['prediction'], delta="Signal Stable")
        
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
st.markdown('<div style="text-align: center; color: gray;"><small> Sonar Project AI | Jennifer Marcia </small></div>', unsafe_allow_html=True)