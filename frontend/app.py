"""
Interface Streamlit pour le RAG Agriculture Burkinabè
Frontend optimisé pour CPU 8Go
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================

st.set_page_config(
    page_title="AgriIA - Assistant Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONSTANTES
# ============================================

API_URL = "http://localhost:8000"

# ============================================
# STYLES CSS
# ============================================

st.markdown("""
    <style>
    /* Style général */
    .main {
        padding: 2rem;
    }
    
    /* Boutons */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-size: 16px;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Boxes de contenu */
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .answer-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #f44336;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #ff9800;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    /* Status indicators */
    .status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .status-disconnected {
        color: #f44336;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# FONCTIONS API
# ============================================

def check_api_health():
    """Vérifie si l'API est accessible et retourne son état"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except requests.exceptions.RequestException as e:
        return False, str(e)

def initialize_system(corpus_path="data/corpus.json"):
    """Initialise le système RAG avec le corpus"""
    try:
        response = requests.post(
            f"{API_URL}/initialize",
            json={"corpus_path": corpus_path},
            timeout=240  # 4 minutes pour l'initialisation
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = f"Erreur {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", error_msg)
            except:
                error_msg = response.text
            return False, {"message": error_msg}
    except requests.exceptions.Timeout:
        return False, {"message": "Timeout: L'initialisation prend trop de temps (>120s). Vérifiez les logs du backend."}
    except requests.exceptions.RequestException as e:
        return False, {"message": f"Erreur de connexion: {str(e)}"}

def ask_question(question, k=2):
    """Pose une question au système RAG"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "k": k},
            timeout=240  # 4 minutes pour la génération
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Erreur inconnue"
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", error_detail)
            except:
                error_detail = response.text
            return False, {"answer": f"Erreur {response.status_code}: {error_detail}"}
            
    except requests.exceptions.Timeout:
        return False, {"answer": "⏱️ Timeout: La génération prend trop de temps. Le modèle est peut-être en train de se charger pour la première fois."}
    except requests.exceptions.RequestException as e:
        return False, {"answer": f"❌ Erreur de connexion: {str(e)}"}

def get_system_info():
    """Récupère les informations du système"""
    try:
        response = requests.get(f"{API_URL}/system/info", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except requests.exceptions.RequestException:
        return False, None

def get_corpus_stats():
    """Récupère les statistiques du corpus"""
    try:
        response = requests.get(f"{API_URL}/corpus/stats", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except requests.exceptions.RequestException:
        return False, None

# ============================================
# INITIALISATION SESSION STATE
# ============================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

if 'first_question' not in st.session_state:
    st.session_state.first_question = True

# ============================================
# EN-TÊTE
# ============================================

st.title("🌾 AgriIA - Assistant Agriculture Burkinabè")
st.markdown("*Système RAG intelligent pour l'agriculture au Burkina Faso*")
st.divider()

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # === SECTION 1: État de l'API ===
    st.subheader("📡 Connexion API")
    
    if st.button("🔄 Vérifier la connexion", use_container_width=True):
        with st.spinner("Vérification..."):
            api_ok, health_data = check_api_health()
            
            if api_ok:
                st.session_state.api_connected = True
                st.success("✅ API connectée")
                
                if health_data and health_data.get("system_initialized", False):
                    st.session_state.initialized = True
                    corpus_size = health_data.get('corpus_size', 0)
                    st.info(f"📚 {corpus_size} documents chargés")
                else:
                    st.session_state.initialized = False
                    st.warning("⚠️ Système non initialisé")
            else:
                st.session_state.api_connected = False
                st.error("❌ API non accessible")
                if health_data:
                    st.caption(f"Détails: {health_data}")
    
    # Affichage du statut
    status_html = ""
    if st.session_state.api_connected:
        status_html = '<p class="status-connected">🟢 Connecté</p>'
    else:
        status_html = '<p class="status-disconnected">🔴 Déconnecté</p>'
    st.markdown(status_html, unsafe_allow_html=True)
    
    st.divider()
    
    # === SECTION 2: Initialisation ===
    st.subheader("🚀 Initialisation")
    
    corpus_path = st.text_input(
        "Chemin du corpus",
        value="data/corpus.json",
        help="Chemin relatif vers le fichier corpus.json depuis le dossier backend"
    )
    
    if st.button("📂 Initialiser le système", use_container_width=True):
        if not st.session_state.api_connected:
            st.error("❌ Connectez d'abord l'API")
        else:
            with st.spinner("⏳ Initialisation en cours (30-90 secondes)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📚 Chargement du corpus...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("🔤 Chargement des embeddings...")
                progress_bar.progress(50)
                
                success, result = initialize_system(corpus_path)
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if success:
                    status = result.get("status", "unknown")
                    message = result.get("message", "")
                    corpus_size = result.get("corpus_size", 0)
                    note = result.get("note", "")
                    
                    if status == "success":
                        st.session_state.initialized = True
                        st.success(f"✅ {message}")
                        st.info(f"📚 {corpus_size} documents")
                        if note:
                            st.caption(f"💡 {note}")
                    elif status == "warning":
                        st.warning(f"⚠️ {message}")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Échec")
                    st.error(result.get("message", "Erreur inconnue"))
    
    # Statut d'initialisation
    if st.session_state.initialized:
        st.success("✅ Système prêt")
    else:
        st.warning("⚠️ Non initialisé")
    
    st.divider()
    
    # === SECTION 3: Paramètres ===
    st.subheader("🔍 Paramètres")
    
    num_docs = st.slider(
        "Documents à récupérer",
        min_value=1,
        max_value=5,
        value=2,
        help="Nombre de documents pertinents pour le contexte"
    )
    
    st.divider()
    
    # === SECTION 4: Actions ===
    st.subheader("🔧 Actions")
    
    if st.button("📋 Infos système", use_container_width=True):
        if st.session_state.api_connected:
            success, info = get_system_info()
            if success:
                st.json(info)
            else:
                st.error("❌ Impossible de récupérer les infos")
        else:
            st.error("❌ API non connectée")
    
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # === SECTION 5: Informations ===
    st.caption("💻 Optimisé pour CPU 8Go")
    st.caption("🤖 facebook/opt-350m")
    st.caption("📦 MiniLM-L12-v2")
    st.caption("⚡ Version 1.0.0")

# ============================================
# CORPS PRINCIPAL
# ============================================

st.header("💬 Posez votre question")

# Vérifications préalables
if not st.session_state.api_connected:
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ API non connectée</h4>
        <p>Veuillez d'abord vérifier la connexion à l'API.</p>
        <p><b>Étapes :</b></p>
        <ol>
            <li>Démarrez le backend : <code>python app.py</code></li>
            <li>Cliquez sur "🔄 Vérifier la connexion"</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.initialized:
    st.markdown("""
    <div class="info-box">
        <h4>📂 Système non initialisé</h4>
        <p>Le système doit être initialisé avant utilisation.</p>
        <p><b>Étapes :</b></p>
        <ol>
            <li>Vérifiez que <code>data/corpus.json</code> existe</li>
            <li>Cliquez sur "📂 Initialiser le système"</li>
            <li>Attendez 30-90 secondes</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

else:
    # === AFFICHAGE DE L'HISTORIQUE ===
    if st.session_state.chat_history:
        st.subheader("📜 Historique")
        
        for i, entry in enumerate(st.session_state.chat_history):
            question_preview = entry['question'][:60]
            is_last = (i == len(st.session_state.chat_history) - 1)
            
            with st.expander(f"💬 {question_preview}...", expanded=is_last):
                st.markdown(f"**Question :** {entry['question']}")
                
                if entry['status'] == 'success':
                    answer_html = f"""
                    <div class="answer-box">
                        <h4>🤖 Réponse</h4>
                        <p>{entry['answer']}</p>
                    </div>
                    """
                    st.markdown(answer_html, unsafe_allow_html=True)
                    
                    if entry.get('relevant_documents'):
                        st.markdown("**📚 Documents sources**")
                        for doc in entry['relevant_documents']:
                            titre = doc.get('titre', 'Sans titre')
                            score = doc.get('score', 0)
                            extrait = doc.get('extrait', '')
                            source = doc.get('source', 'Non spécifiée')
                            
                            doc_html = f"""
                            <div class="source-box">
                                <b>{titre}</b><br>
                                <small>Score: {score:.3f}</small><br>
                                <i>{extrait}</i><br>
                                <small>📎 {source}</small>
                            </div>
                            """
                            st.markdown(doc_html, unsafe_allow_html=True)
                    
                    st.caption(f"⏱️ {entry.get('timestamp', '')}")
                    
                elif entry['status'] == 'warning':
                    warning_html = f"""
                    <div class="warning-box">
                        <p>{entry['answer']}</p>
                    </div>
                    """
                    st.markdown(warning_html, unsafe_allow_html=True)
                    
                else:
                    error_html = f"""
                    <div class="error-box">
                        <p>{entry['answer']}</p>
                    </div>
                    """
                    st.markdown(error_html, unsafe_allow_html=True)
        
        st.divider()
    
    # === NOUVELLE QUESTION ===
    st.subheader("❓ Nouvelle question")
    
    # Note importante pour la première question
    if st.session_state.first_question and not st.session_state.chat_history:
        st.info("💡 **Note** : La première question peut prendre 60-90 secondes (chargement du modèle). Les suivantes seront plus rapides !")
    
    
    
    cols = st.columns(2)
    
    # Zone de saisie
    question = st.text_area(
        "Votre question :",
        value=st.session_state.get('example_question', ''),
        height=100,
        placeholder="Ex: Comment cultiver le sorgho ?",
        help="Posez une question claire sur l'agriculture"
    )
    
    if 'example_question' in st.session_state:
        del st.session_state.example_question
    
    # Boutons d'action
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ask_button = st.button("🚀 Poser la question", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Effacer", use_container_width=True)
    
    # Traitement de la question
    if ask_button and question.strip():
        # Estimation du temps
        estimated_time = "60-90 secondes" if st.session_state.first_question else "10-30 secondes"
        
        with st.spinner(f"🔍 Traitement en cours (environ {estimated_time})..."):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            progress_text.text("📚 Recherche des documents...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            progress_text.text("🤖 Génération de la réponse...")
            progress_bar.progress(40)
            
            if st.session_state.first_question:
                progress_text.text("⏳ Chargement du modèle (première fois)...")
                progress_bar.progress(60)
            
            success, response = ask_question(question, k=num_docs)
            
            progress_bar.progress(100)
            progress_text.empty()
            progress_bar.empty()
            
            if success:
                st.session_state.first_question = False
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response.get('answer', ''),
                    'status': response.get('status', 'unknown'),
                    'relevant_documents': response.get('relevant_documents', []),
                    'sources': response.get('sources', []),
                    'timestamp': response.get('timestamp', '')
                })
                st.rerun()
            else:
                error_msg = response.get('answer', 'Erreur inconnue')
                st.error(f"❌ {error_msg}")
    
    elif ask_button:
        st.warning("⚠️ Veuillez saisir une question")
    
    if clear_button:
        st.rerun()

# ============================================
# PIED DE PAGE
# ============================================

st.divider()
footer_html = """
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>
        🌾 AgriIA - Assistant Agriculture Burkinabè | 
        💻 Optimisé CPU 8Go | 
        🤖 OPT-350m
    </p>
    <p><small>Version 1.0.0 - 2025</small></p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)