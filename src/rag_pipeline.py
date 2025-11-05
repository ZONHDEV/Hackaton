"""
RAG Ultra-Léger 100% Open Source pour CPU 8Go
Optimisé pour MiniLM embeddings + DialoGPT local
"""

import json, re, logging, numpy as np
from typing import List, Dict
from datetime import datetime
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------
# Embeddings léger
# --------------------
class EmbeddingModel:
    def __init__(self, local_dir=r"J:\Hackaton\AgriIA\embeddings\paraphrase-multilingual-MiniLM-L12-v2"):
        logger.info(f"📦 Chargement embeddings depuis: {local_dir}")
        try:
            if not os.path.exists(local_dir):
                logger.error(f"❌ Dossier embeddings introuvable: {local_dir}")
                raise FileNotFoundError(f"Dossier embeddings introuvable: {local_dir}")
            
            self.model = SentenceTransformer(local_dir)
            self.dimension = 384
            logger.info("✅ Embeddings chargés avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement embeddings: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size=32) -> np.ndarray:
        try:
            return self.model.encode(
                texts, 
                convert_to_tensor=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
        except Exception as e:
            logger.error(f"❌ Erreur encodage: {e}")
            raise

# --------------------
# Base vectorielle FAISS
# --------------------
class VectorDatabase:
    def __init__(self):
        self.index = None
        self.documents = []
        self.doc_embeddings = None
    
    def build_index(self, documents: List[Dict], embedding_model: EmbeddingModel):
        try:
            self.documents = documents
            
            logger.info(f"📝 Préparation de {len(documents)} documents...")
            texts = [
                re.sub(r'\s+', ' ', doc.get("contenu", ""))[:500]
                for doc in documents
            ]
            
            logger.info("🔄 Encodage des documents...")
            self.doc_embeddings = embedding_model.encode(texts, batch_size=16)
            
            logger.info("🔨 Construction de l'index FAISS...")
            self.index = faiss.IndexFlatIP(embedding_model.dimension)
            self.index.add(self.doc_embeddings.astype('float32'))
            
            logger.info(f"✅ Index construit: {self.index.ntotal} vecteurs")
            
        except Exception as e:
            logger.error(f"❌ Erreur construction index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k=2) -> List[Dict]:
        try:
            if self.index is None:
                raise ValueError("Index non initialisé")
            
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        "document": self.documents[idx],
                        "score": float(scores[0][i]),
                        "rank": i+1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            raise


# --------------------
# LLM DialoGPT-medium - VERSION CORRIGÉE
# --------------------
class LightLanguageModel:
    def __init__(self):
        self.local_model_dir = r"J:\Hackaton\AgriIA\llm\DialoGPT-medium"
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        
        logger.info(f"📋 Configuration LLM DialoGPT-medium")
    
    def _check_local_model(self):
        """Vérifie si le modèle DialoGPT est disponible localement"""
        if not os.path.exists(self.local_model_dir):
            logger.error(f"❌ Dossier modèle introuvable: {self.local_model_dir}")
            return False
            
        required_files = ['config.json', 'pytorch_model.bin']
        for file in required_files:
            if not os.path.exists(os.path.join(self.local_model_dir, file)):
                logger.error(f"❌ Fichier manquant: {file}")
                return False
        
        logger.info(f"✅ Modèle DialoGPT trouvé et complet")
        return True
    
    def _load_model(self):
        """Charge DialoGPT en mode LOCAL UNIQUEMENT"""
        if self._is_loaded:
            return
        
        logger.info("🤖 Chargement de DialoGPT-medium...")
        
        if not self._check_local_model():
            raise FileNotFoundError(f"Modèle DialoGPT introuvable dans {self.local_model_dir}")
        
        try:
            # Tokenizer
            logger.info("🔤 Chargement du tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_dir,
                local_files_only=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ Tokenizer chargé")
            
            # Modèle
            logger.info("🤖 Chargement du modèle...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_dir,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            self._is_loaded = True
            logger.info("✅ DialoGPT-medium chargé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement DialoGPT: {e}")
            raise
    
    def generate_answer(self, context: str, question: str, max_tokens=100):
        """Génère une réponse avec DialoGPT - VERSION CORRIGÉE"""
        try:
            if not self._is_loaded:
                self._load_model()
            
            # Nettoyer et limiter le contexte
            context_clean = context[:400].strip()
            
            # PROMPT CORRIGÉ - Format conversationnel que DialoGPT comprend
            prompt = f"User: J'ai ces informations: {context_clean}\nQuestion: {question}\nBot:"
            
            logger.info(f"📝 Prompt préparé ({len(prompt)} caractères)")
            
            # Tokenization
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            
            # GÉNÉRATION CORRIGÉE
            logger.info("🔄 Génération de la réponse...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    early_stopping=True,
                    num_return_sequences=1
                )
            
            # Décodage
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"📄 Texte complet généré: {generated_text}")
            
            # EXTRACTION CORRIGÉE - Prendre tout après "Bot:"
            if "Bot:" in generated_text:
                answer = generated_text.split("Bot:")[-1].strip()
            else:
                # Fallback: prendre la fin après le prompt
                answer = generated_text.replace(prompt, "").strip()
            
            # Si toujours vide, utiliser une réponse par défaut
            if not answer or len(answer) < 5:
                answer = self._create_fallback_answer(context, question)
            else:
                # Nettoyer la réponse
                answer = self._clean_answer(answer)
            
            logger.info(f"✅ Réponse finale: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Erreur génération: {e}", exc_info=True)
            return self._create_fallback_answer(context, question)
    
    def _clean_answer(self, answer):
        """Nettoie la réponse"""
        if not answer:
            return "Je n'ai pas pu générer de réponse spécifique."
        
        # Supprimer les répétitions de "Bot:" ou "User:"
        answer = re.sub(r'(Bot:|User:).*', '', answer)
        
        # Nettoyer les caractères spéciaux
        answer = re.sub(r'[<\|\>\"\']', '', answer)
        
        # Séparer en phrases et supprimer les doublons
        sentences = [s.strip() for s in answer.split('.') if s.strip() and len(s.strip()) > 10]
        
        if sentences:
            # Prendre les 2 premières phrases uniques
            unique_sentences = []
            seen = set()
            for sent in sentences:
                key = sent.lower()[:30]  # Prendre le début pour comparaison
                if key not in seen:
                    unique_sentences.append(sent)
                    seen.add(key)
                    if len(unique_sentences) >= 2:
                        break
            
            answer = '. '.join(unique_sentences)
            if not answer.endswith('.'):
                answer += '.'
        else:
            answer = "Je ne trouve pas d'information spécifique dans les documents."
        
        # Limiter la longueur
        if len(answer) > 250:
            answer = answer[:247] + '...'
        
        return answer
    
    def _create_fallback_answer(self, context, question):
        """Réponse de secours basée sur le contexte"""
        logger.info("🔄 Utilisation de la réponse de secours...")
        
        # Extraire les phrases les plus pertinentes
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        # Filtrer les phrases trop courtes
        relevant_sentences = [s for s in sentences if len(s) > 20]
        
        if relevant_sentences:
            # Prendre les 2 premières phrases
            return f"D'après les documents: {'. '.join(relevant_sentences[:2])}."
        else:
            return "Les documents contiennent des informations sur ce sujet. Veuillez consulter les documents pertinents pour plus de détails."


# --------------------
# Système RAG léger - VERSION CORRIGÉE
# --------------------
class AgriRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.llm = None
        self.is_initialized = False
        self.corpus_size = 0
        self.documents = []
        logger.info("🌱 Système RAG créé")
    
    def initialize(self, corpus_path="data/corpus.json"):
        try:
            logger.info(f"📂 Initialisation du système RAG...")
            
            # ÉTAPE 1: Chargement du corpus
            logger.info("📚 [1/4] Chargement du corpus...")
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(f"Corpus introuvable: {corpus_path}")
                
            with open(corpus_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)

            self.documents = [
                {
                    "titre": d.get("titre", "Sans titre"),
                    "contenu": d.get("contenu", ""),
                    "sources": d.get("sources", "Source inconnue")
                }
                for d in (raw_docs if isinstance(raw_docs, list) else [raw_docs]) 
                if d.get("contenu")
            ]

            self.corpus_size = len(self.documents)
            logger.info(f"✅ {self.corpus_size} documents chargés")

            if self.corpus_size == 0:
                return {
                    "status": "warning",
                    "message": "Aucun document valide trouvé",
                    "corpus_size": 0
                }

            # ÉTAPE 2: Embeddings
            logger.info("🔤 [2/4] Chargement des embeddings...")
            self.embedding_model = EmbeddingModel()
            
            # ÉTAPE 3: Index vectoriel
            logger.info("🗂️ [3/4] Construction de l'index...")
            self.vector_db = VectorDatabase()
            self.vector_db.build_index(self.documents, self.embedding_model)
            
            # ÉTAPE 4: Configuration LLM
            logger.info("🤖 [4/4] Configuration du LLM...")
            self.llm = LightLanguageModel()

            self.is_initialized = True
            
            logger.info(f"✅ Système initialisé avec succès")
            return {
                "status": "success",
                "message": "Système RAG initialisé",
                "corpus_size": self.corpus_size
            }

        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            return {
                "status": "error",
                "message": f"Erreur: {str(e)}",
                "corpus_size": 0
            }
    
    def ask(self, question: str, k=3):
        """Question au système RAG - VERSION AMÉLIORÉE"""
        try:
            if not self.is_initialized:
                return {
                    "status": "error",
                    "answer": "Système non initialisé. Veuillez d'abord appeler initialize()."
                }
            
            logger.info(f"❓ Question: {question}")
            
            if not question or len(question.strip()) < 2:
                return {
                    "status": "error",
                    "answer": "Question trop courte."
                }
            
            # Recherche de documents
            q_emb = self.embedding_model.encode([question], batch_size=1)[0]
            results = self.vector_db.search(q_emb, k=k)
            
            logger.info(f"📄 {len(results)} documents trouvés")
            
            if not results:
                return {
                    "status": "warning",
                    "question": question,
                    "answer": "Aucun document pertinent trouvé pour cette question.",
                    "relevant_documents": [],
                    "corpus_size": self.corpus_size
                }
            
            # Construction du contexte amélioré
            context_parts = []
            for i, result in enumerate(results):
                content = result["document"].get("contenu", "")
                excerpt = content[:350] + "..." if len(content) > 350 else content
                context_parts.append(f"[Document {i+1}] {excerpt}")
            
            context = " ".join(context_parts)
            logger.info(f"📝 Contexte préparé ({len(context)} caractères)")
            
            # Génération de réponse
            answer = self.llm.generate_answer(context, question)
            
            # Préparation des résultats
            relevant_docs = [
                {
                    "titre": r["document"].get("titre", "Sans titre"),
                    "extrait": r["document"].get("contenu", "")[:200] + "...",
                    "score": round(r["score"], 3),
                    "source": r["document"].get("sources", "Source inconnue")
                }
                for r in results
            ]
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "relevant_documents": relevant_docs,
                "context_used": len(context),
                "corpus_size": self.corpus_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur dans ask(): {e}", exc_info=True)
            return {
                "status": "error",
                "question": question,
                "answer": f"Erreur lors du traitement: {str(e)}",
                "relevant_documents": [],
                "corpus_size": self.corpus_size
            }

    def get_system_info(self):
        """Retourne les informations du système"""
        return {
            "is_initialized": self.is_initialized,
            "corpus_size": self.corpus_size,
            "llm_loaded": self.llm._is_loaded if self.llm else False,
            "llm_type": "DialoGPT-medium"
        }


