"""
Interface Streamlit pour Agri-Assistant
Licence: MIT
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="Agri Assistant Burkina",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f8fff8;
        border-left: 4px solid #2E8B57;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-item {
        background-color: #f0f8f0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border: 1px solid #e0e8e0;
    }
    .document-card {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stButton button {
        background-color: #2E8B57;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #3CB371;
    }
</style>
""", unsafe_allow_html=True)

class AgriAssistantFrontend:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def check_health(self):
        """Vérifie si l'API est disponible - CORRIGÉ"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except Exception as e:
            return False, {"error": str(e)}
    
    def ask_question(self, question, k=3):
        """Envoie une question à l'API - CORRIGÉ"""
        try:
            response = self.session.post(
                f"{self.api_url}/ask",
                json={"question": question, "k": k},
                timeout=200
            )
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_system_info(self):
        """Récupère les informations du système - CORRIGÉ"""
        try:
            response = self.session.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except Exception as e:
            return False, {"error": str(e)}

def main():
    # Initialisation du frontend
    assistant = AgriAssistantFrontend()
    
    # En-tête principale
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Agri Assistant Burkina</h1>
        <h3>Votre assistant IA 100% Open Source pour l'agriculture burkinabè</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ À Propos")
        st.write("""
        Cet assistant utilise exclusivement des technologies **open source** 
        pour répondre à vos questions sur l'agriculture burkinabè.
        
        **Domaines couverts:**
        🌿 Karité et transformation
        🌾 Coton et filière
        🌽 Mil et sorgho
        💧 Techniques durables
        📊 Marchés agricoles
        """)
        
        # Vérification de la santé de l'API
        st.header("🔍 Statut du Système")
        health_ok, health_data = assistant.check_health()
        
        if health_ok:
            st.success("✅ API Connectée")
            if health_data:
                st.metric("Documents chargés", health_data.get("corpus_size", 0))
                st.metric("Système initialisé", "Oui" if health_data.get("initialized") else "Non")
        else:
            st.error("❌ API Non Connectée")
            st.info("Vérifiez que le serveur backend est démarré sur http://localhost:8000")
            if health_data and "error" in health_data:
                st.error(f"Détail: {health_data['error']}")
        
        # Exemples de questions
        st.header("💡 Exemples de Questions")
        example_questions = [
            "Quelles sont les étapes de transformation du karité ?",
            "Comment cultiver le mil dans les zones arides ?",
            "Quels sont les marchés pour le coton burkinabè ?",
            "Techniques d'irrigation économiques au Burkina",
            "Variétés de mil résistantes à la sécheresse",
            "Comment produire du beurre de karité de qualité ?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}"):
                st.session_state.question_input = q
        
        # Paramètres de recherche
        st.header("⚙️ Paramètres")
        k_documents = st.slider("Nombre de documents à utiliser", 1, 5, 3)
        
        # Informations techniques
        st.header("🔧 Technologies")
        st.write("""
        - **Embeddings:** sentence-transformers
        - **Base vectorielle:** FAISS
        - **Modèle de langage:** Mistral-7B
        - **Interface:** Streamlit
        - **API:** FastAPI
        """)
    
    # Zone de question principale
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "💬 Posez votre question sur l'agriculture burkinabè:",
            value=st.session_state.get('question_input', ''),
            placeholder="Ex: Comment transformer le karité en beurre ?",
            key="question_input_main"
        )
    
    with col2:
        st.write("")  # Espacement
        st.write("")
        search_button = st.button("🔍 Rechercher", use_container_width=True)
    
    # Traitement de la question
    if search_button and question:
        with st.spinner("🔍 Recherche dans nos documents agricoles..."):
            start_time = time.time()
            success, response = assistant.ask_question(question, k_documents)
            response_time = time.time() - start_time
            
            if success and response:
                # Affichage de la réponse
                st.markdown("### 📝 Réponse")
                st.markdown(f'<div class="answer-box">{response.get("answer", "Aucune réponse générée")}</div>', unsafe_allow_html=True)
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temps de réponse", f"{response_time:.2f}s")
                with col2:
                    st.metric("Documents utilisés", response.get("context_used", 0))
                with col3:
                    st.metric("Taille du corpus", response.get("corpus_size", 0))
                
                # Sources utilisées
                st.markdown("### 📚 Sources")
                sources = response.get("sources", [])
                if sources:
                    for i, source in enumerate(sources):
                        st.markdown(f'''
                        <div class="source-item">
                            <strong>Source {i+1}:</strong> {source}
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.info("Aucune source spécifique utilisée")
                
                # Documents pertinents
                st.markdown("### 🔍 Documents Pertinents")
                documents = response.get("relevant_documents", [])
                if documents:
                    for doc in documents:
                        with st.expander(f"📄 {doc.get('title', 'Sans titre')} (Score: {doc.get('score', 0):.3f})"):
                            st.write(f"**Source:** {doc.get('source', 'Non spécifié')}")
                            st.write(f"**Rang:** {doc.get('rank', 'N/A')}")
                else:
                    st.info("Aucun document pertinent trouvé")
                
                # Données brutes (pour debug)
                with st.expander("📊 Données techniques (Debug)"):
                    st.json(response)
                
            else:
                st.error("❌ Erreur lors de la recherche de réponse")
                if response and "error" in response:
                    st.error(f"Erreur détaillée: {response['error']}")
    
    # Section d'information quand aucune recherche n'est en cours
    elif not question:
        st.markdown("---")
        st.markdown("### 🎯 Comment utiliser Agri Assistant")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3>🌍 100% Open Source</h3>
                <p>Toutes les technologies utilisées sont open source et transparentes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3>🇧🇫 Contexte Local</h3>
                <p>Spécialisé sur l'agriculture burkinabè avec des données locales</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3>🔒 Souveraineté</h3>
                <p>Pas de dépendance aux services cloud propriétaires</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistiques si disponibles
        if health_ok and health_data:
            st.markdown("### 📊 Statistiques du Système")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents", health_data.get("corpus_size", 0))
            with col2:
                st.metric("Système", "Actif" if health_data.get("initialized") else "Inactif")
            with col3:
                st.metric("API", "En ligne")
            with col4:
                st.metric("Open Source", "100%")

if __name__ == "__main__":
    main()