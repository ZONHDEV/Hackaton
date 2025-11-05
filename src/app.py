"""
Serveur API FastAPI pour AgriIA - Backend optimisé CPU 8Go
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire backend au path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import du système RAG
try:
    from rag_pipeline import AgriRAGSystem
    logger.info("✅ Module rag_pipeline importé avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur d'import rag_pipeline: {e}")
    logger.info(f"📁 Contenu du dossier backend: {os.listdir(backend_dir)}")
    sys.exit(1)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API RAG Agriculture Burkinabè",
    description="API pour le système RAG d'agriculture burkinabè avec OPT-350m",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODÈLES PYDANTIC
# ============================================

class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = 2

class InitializeRequest(BaseModel):
    corpus_path: str = "data/corpus.json"

class RAGResponse(BaseModel):
    status: str
    question: str
    answer: str
    sources: list
    relevant_documents: list
    context_used: int
    corpus_size: int
    model: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    system_initialized: bool
    model: str
    corpus_size: int
    message: Optional[str] = None

class InitializeResponse(BaseModel):
    status: str
    message: str
    corpus_size: int
    note: Optional[str] = None

# ============================================
# INSTANCE GLOBALE DU SYSTÈME RAG
# ============================================

rag_system = None

# ============================================
# EVENTS HANDLERS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage du serveur"""
    global rag_system
    
    logger.info("="*60)
    logger.info("🌱 DÉMARRAGE DU SERVEUR RAG AGRICULTURE BURKINABÈ")
    logger.info("="*60)
    logger.info(f"📁 Répertoire backend: {backend_dir}")
    logger.info(f"📌 API disponible sur: http://localhost:8000")
    logger.info(f"📖 Documentation: http://localhost:8000/docs")
    logger.info("="*60)
    
    try:
        rag_system = AgriRAGSystem()
        logger.info("✅ Système RAG créé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur création système RAG: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt du serveur"""
    logger.info("🛑 Arrêt du serveur RAG...")
    logger.info("👋 Au revoir!")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Endpoint racine - Informations générales"""
    return {
        "message": "🌾 API RAG Agriculture Burkinabè",
        "version": "1.0.0",
        "model": "facebook/opt-350m",
        "status": "running",
        "endpoints": {
            "health": "GET /health - Vérification de l'état",
            "system_info": "GET /system/info - Informations système",
            "initialize": "POST /initialize - Initialisation du corpus",
            "ask": "POST /ask - Poser une question",
            "corpus_stats": "GET /corpus/stats - Statistiques du corpus",
            "docs": "GET /docs - Documentation interactive"
        },
        "documentation": "http://localhost:8000/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état du serveur et du système RAG"""
    
    if rag_system is None:
        return HealthResponse(
            status="error",
            system_initialized=False,
            model="N/A",
            corpus_size=0,
            message="Système RAG non créé"
        )
    
    return HealthResponse(
        status="healthy" if rag_system.is_initialized else "not_initialized",
        system_initialized=rag_system.is_initialized,
        model="facebook/opt-350m",
        corpus_size=rag_system.corpus_size,
        message="Système opérationnel" if rag_system.is_initialized else "En attente d'initialisation"
    )

@app.get("/system/info")
async def system_info():
    """Informations détaillées sur le système RAG"""
    
    if rag_system is None:
        raise HTTPException(
            status_code=500, 
            detail="Système RAG non créé"
        )
    
    info = rag_system.get_system_info()
    
    return {
        "status": "success",
        "system_info": info,
        "backend_dir": str(backend_dir),
        "python_version": sys.version.split()[0]
    }

@app.post("/initialize", response_model=InitializeResponse)
async def initialize_system(request: InitializeRequest):
    """
    Initialise le système RAG avec le corpus de documents
    
    - **corpus_path**: Chemin vers le fichier corpus.json (relatif au backend)
    """
    
    if rag_system is None:
        raise HTTPException(
            status_code=500, 
            detail="Système RAG non créé"
        )
    
    try:
        logger.info("="*60)
        logger.info(f"📂 DEMANDE D'INITIALISATION")
        logger.info(f"Chemin du corpus: {request.corpus_path}")
        logger.info("="*60)
        
        # Appel de la méthode d'initialisation
        result = rag_system.initialize(request.corpus_path)
        
        # Vérification du résultat
        if result["status"] == "error":
            logger.error(f"❌ Échec de l'initialisation: {result['message']}")
            raise HTTPException(
                status_code=500, 
                detail=result["message"]
            )
        
        if result["status"] == "warning":
            logger.warning(f"⚠️ Avertissement: {result['message']}")
        else:
            logger.info(f"✅ Initialisation réussie: {result['message']}")
        
        return InitializeResponse(
            status=result["status"],
            message=result["message"],
            corpus_size=result["corpus_size"],
            note=result.get("note")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur serveur: {str(e)}"
        )

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    """
    Pose une question au système RAG
    
    - **question**: La question à poser
    - **k**: Nombre de documents pertinents à récupérer (1-5, défaut: 2)
    """
    
    if rag_system is None:
        raise HTTPException(
            status_code=500, 
            detail="Système RAG non créé"
        )
    
    # Validation des paramètres
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide"
        )
    
    if request.k < 1 or request.k > 5:
        raise HTTPException(
            status_code=400,
            detail="Le paramètre k doit être entre 1 et 5"
        )
    
    try:
        logger.info("="*60)
        logger.info(f"❓ NOUVELLE QUESTION")
        logger.info(f"Question: {request.question}")
        logger.info(f"Nombre de documents: {request.k}")
        logger.info("="*60)
        
        # Appel du système RAG
        response = rag_system.ask(request.question, request.k)
        
        # Log du résultat
        if response["status"] == "error":
            logger.error(f"❌ Erreur: {response['answer']}")
        elif response["status"] == "warning":
            logger.warning(f"⚠️ Avertissement: {response['answer']}")
        else:
            logger.info(f"✅ Réponse générée avec succès")
        
        return RAGResponse(**response)
        
    except Exception as e:
        logger.error(f"❌ Erreur endpoint ask: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur serveur: {str(e)}"
        )

@app.get("/corpus/stats")
async def corpus_stats():
    """Statistiques sur le corpus chargé"""
    
    if rag_system is None:
        raise HTTPException(
            status_code=500, 
            detail="Système RAG non créé"
        )
    
    if not rag_system.is_initialized:
        raise HTTPException(
            status_code=400, 
            detail="Système non initialisé. Veuillez d'abord initialiser le système."
        )
    
    return {
        "status": "success",
        "corpus_size": rag_system.corpus_size,
        "documents_count": len(rag_system.documents),
        "is_initialized": rag_system.is_initialized,
        "embedding_dimension": 384,
        "model": "facebook/opt-350m"
    }

@app.get("/ping")
async def ping():
    """Endpoint simple pour vérifier que le serveur répond"""
    return {"ping": "pong", "status": "ok"}

# ============================================
# GESTION DES ERREURS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Gestionnaire d'erreur 404"""
    return {
        "error": "Endpoint non trouvé",
        "path": str(request.url),
        "message": "Consultez /docs pour la liste des endpoints disponibles"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Gestionnaire d'erreur 500"""
    logger.error(f"Erreur serveur 500: {exc}")
    return {
        "error": "Erreur interne du serveur",
        "message": "Une erreur s'est produite. Consultez les logs pour plus de détails."
    }

# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 LANCEMENT DU SERVEUR API")
    print("="*60)
    print("📌 URL: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("💡 Conseil: Initialisez le système via POST /initialize")
    print("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Désactivé pour économiser la mémoire
        log_level="info",
        access_log=True
    )