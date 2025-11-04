from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import AgriRAGSystem
from fastapi.middleware.cors import CORSMiddleware
import os

# --- Initialisation de l'application FastAPI ---
app = FastAPI(title="AgriRAG API", version="1.0")

# --- Configuration CORS pour Streamlit ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement du système RAG ---
rag_system = AgriRAGSystem()

# Chemin vers le corpus - ajustez selon votre structure
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.json")

# Initialisation au démarrage
@app.on_event("startup")
async def startup_event():
    try:
        rag_system.initialize(CORPUS_PATH)
        print("✅ Système RAG initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")

# --- Modèle de requête pour les questions ---
class QuestionRequest(BaseModel):
    question: str
    k: int = 3  # Valeur par défaut

# --- Endpoint racine : simple message ---
@app.get("/")
def root():
    """
    Endpoint principal : vérifie si l'API est en ligne
    """
    return {
        "message": "✅ API AgriRAG en ligne !",
        "initialized": rag_system.is_initialized,
        "corpus_size": rag_system.corpus_size
    }

# --- Endpoint de santé --- CORRIGÉ
@app.get("/health")
def health():
    """
    Donne le statut du système RAG (utile pour Streamlit)
    """
    return {
        "status": "ok",
        "initialized": rag_system.is_initialized,
        "corpus_size": rag_system.corpus_size
    }

# --- Endpoint pour poser une question --- CORRIGÉ
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Permet de poser une question au système RAG
    """
    if not rag_system.is_initialized:
        raise HTTPException(status_code=400, detail="Le système RAG n'est pas encore initialisé.")

    try:
        response = rag_system.ask(request.question, request.k)
        return response  # Retourne directement la réponse du système RAG
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)