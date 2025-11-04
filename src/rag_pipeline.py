"""
Pipeline RAG 100% Open Source pour l'Agriculture Burkinabè
Technologies utilisées :
- Embeddings: sentence-transformers (Apache 2.0)
- Vector DB: FAISS (MIT)
- LLM: Mistral-7B (Apache 2.0)
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Modèle d'embeddings open source multilingue
    Licence: Apache 2.0
    Source: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        logger.info(f"Chargement du modèle d'embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension des embeddings pour ce modèle
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes en vecteurs numériques"""
        return self.model.encode(texts, convert_to_tensor=False)

class VectorDatabase:
    """
    Base de données vectorielle avec FAISS
    Licence: MIT
    Source: https://github.com/facebookresearch/faiss
    """
    
    def __init__(self):
        self.index = None
        self.documents = []
        self.doc_embeddings = None
    
    def build_index(self, documents: List[Dict], embedding_model: EmbeddingModel):
        """Construit l'index FAISS à partir des documents"""
        logger.info(f"Construction de l'index avec {len(documents)} documents")
        self.documents = documents
        
        # Extraction du contenu texte
        texts = [doc["content"] for doc in documents]
        
        # Génération des embeddings
        self.doc_embeddings = embedding_model.encode(texts)
        
        # Création de l'index FAISS
        self.index = faiss.IndexFlatIP(embedding_model.dimension)
        
        # Normalisation des vecteurs pour la similarité cosinus
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings.astype('float32'))
        
        logger.info(f"Index construit avec {self.index.ntotal} vecteurs")
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """Recherche les k documents les plus similaires"""
        if self.index is None:
            raise ValueError("Index non initialisé. Appelez build_index() d'abord.")
        
        # Normalisation de la requête
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Recherche par similarité cosinus
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(scores[0][i]),
                    "rank": i + 1
                })
        
        return results

class LanguageModel:
    """
    Modèle de langage open source - Version légère pour démonstration
    En production, utiliser Mistral-7B ou Llama-2-7B
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        logger.info(f"Chargement du modèle de langage: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    def generate_answer(self, context: str, question: str, max_length: int = 300) -> str:
        """Génère une réponse contextuelle"""
        prompt = self._build_prompt(context, question)
        
        try:
            response = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            answer = generated_text.replace(prompt, "").strip()
            
            # Nettoyage de la réponse
            answer = self._clean_response(answer)
            return answer
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return "Je n'ai pas pu générer de réponse pour le moment. Veuillez reformuler votre question."
    
    def _build_prompt(self, context: str, question: str) -> str:
        """Construit le prompt pour le modèle"""
        return f"""Basé sur le contexte suivant sur l'agriculture burkinabè, réponds à la question de manière précise et utile.

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""
    
    def _clean_response(self, text: str) -> str:
        """Nettoie la réponse générée"""
        # Supprime les répétitions et coupe à la première ponctuation forte si nécessaire
        stop_phrases = ["QUESTION:", "CONTEXTE:", "RÉPONSE:", "\\n", "**"]
        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[0]
        
        return text.strip()

class AgriRAGSystem:
    """
    Système RAG complet pour l'agriculture burkinabè
    """
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDatabase()
        self.llm = LanguageModel()
        self.is_initialized = False
        self.corpus_size = 0
    
    def initialize(self, corpus_path: str = "data/corpus.json"):
        """Initialise le système RAG avec les données"""
        logger.info("Initialisation du système RAG...")
        
        try:
            # Chargement des données
            with open(corpus_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            self.corpus_size = len(documents)
            logger.info(f"Chargement de {self.corpus_size} documents")
            
            # Construction de l'index vectoriel
            self.vector_db.build_index(documents, self.embedding_model)
            self.is_initialized = True
            
            logger.info("✅ Système RAG initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            raise
    
    def ask(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Pose une question au système RAG"""
        if not self.is_initialized:
            return {
                "question": question,
                "answer": "Système non initialisé. Veuillez charger les données d'abord.",
                "sources": [],
                "relevant_documents": [],
                "error": "System not initialized"
            }
        
        try:
            logger.info(f"Traitement de la question: {question}")
            
            # Étape 1: Encodage de la question
            question_embedding = self.embedding_model.encode([question])[0]
            
            # Étape 2: Recherche de documents pertinents
            search_results = self.vector_db.search(question_embedding, k=k)
            
            # Étape 3: Préparation du contexte
            context = self._prepare_context(search_results)
            
            # Étape 4: Génération de la réponse
            answer = self.llm.generate_answer(context, question)
            
            # Étape 5: Préparation des métadonnées
            response_data = {
                "question": question,
                "answer": answer,
                "sources": [result["document"]["source"] for result in search_results],
                "relevant_documents": [
                    {
                        "title": result["document"]["title"],
                        "source": result["document"]["source"],
                        "score": result["score"],
                        "rank": result["rank"]
                    }
                    for result in search_results
                ],
                "context_used": len(search_results),
                "corpus_size": self.corpus_size
            }
            
            logger.info(f"✅ Réponse générée avec {len(search_results)} documents de contexte")
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement: {e}")
            return {
                "question": question,
                "answer": f"Erreur lors du traitement: {str(e)}",
                "sources": [],
                "relevant_documents": [],
                "error": str(e)
            }
    
    def _prepare_context(self, search_results: List[Dict]) -> str:
        """Prépare le contexte à partir des documents trouvés"""
        context_parts = []
        
        for i, result in enumerate(search_results):
            doc = result["document"]
            context_parts.append(
                f"Document {i+1} (score: {result['score']:.3f}):\n"
                f"Titre: {doc['title']}\n"
                f"Contenu: {doc['content']}\n"
                f"Source: {doc['source']}\n"
            )
        
        return "\n".join(context_parts)

# Instance globale pour l'utilisation
rag_system = AgriRAGSystem()

