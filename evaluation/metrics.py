"""
Système d'évaluation pour Agri-Assistant
Licence: MIT
"""

import json
import time
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.precision_recall_fscore_support import precision_recall_fscore_support
import logging
from rag_pipeline import rag_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Calcule les métriques d'évaluation pour le système RAG"""
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calcule la similarité cosinus entre deux textes"""
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        cosine_sim = np.dot(vectors[0], vectors[1]) / (
            np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
        )
        return cosine_sim
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
        """Calcule la précision à K documents"""
        if k == 0:
            return 0.0
        
        top_k_sources = [doc['document']['source'] for doc in retrieved_docs[:k]]
        relevant_count = sum(1 for source in top_k_sources if source in relevant_docs)
        return relevant_count / k
    
    @staticmethod
    def reciprocal_rank(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """Calcule le rang réciproque moyen"""
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc['document']['source'] in relevant_docs:
                return 1.0 / rank
        return 0.0

class AgriAssistantEvaluator:
    """Évalue les performances du système Agri-Assistant"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.metrics = EvaluationMetrics()
    
    def evaluate_retrieval(self, test_questions: List[Dict], k: int = 3) -> Dict[str, float]:
        """Évalue la composante de recherche du RAG"""
        precision_scores = []
        reciprocal_ranks = []
        response_times = []
        
        for question_data in test_questions:
            question = question_data["question"]
            expected_sources = [question_data.get("expected_source", "")]
            
            start_time = time.time()
            search_results = self.rag_system.vector_db.search(
                self.rag_system.embedding_model.encode([question])[0], 
                k=k
            )
            response_time = time.time() - start_time
            
            # Métriques de recherche
            precision = self.metrics.precision_at_k(search_results, expected_sources, k)
            rr = self.metrics.reciprocal_rank(search_results, expected_sources)
            
            precision_scores.append(precision)
            reciprocal_ranks.append(rr)
            response_times.append(response_time)
        
        return {
            "precision@k": np.mean(precision_scores),
            "mean_reciprocal_rank": np.mean(reciprocal_ranks),
            "mean_response_time": np.mean(response_times),
            "retrieval_coverage": len([p for p in precision_scores if p > 0]) / len(precision_scores)
        }
    
    def evaluate_generation(self, test_questions: List[Dict]) -> Dict[str, float]:
        """Évalue la composante de génération du RAG"""
        similarity_scores = []
        response_times = []
        
        for question_data in test_questions:
            question = question_data["question"]
            expected_answer = question_data["expected_answer"]
            
            start_time = time.time()
            response = self.rag_system.ask(question)
            response_time = time.time() - start_time
            
            # Similarité sémantique entre réponse générée et attendue
            if "answer" in response:
                similarity = self.metrics.calculate_similarity(
                    response["answer"], 
                    expected_answer
                )
                similarity_scores.append(similarity)
            
            response_times.append(response_time)
        
        return {
            "answer_similarity": np.mean(similarity_scores) if similarity_scores else 0,
            "generation_success_rate": len(similarity_scores) / len(test_questions),
            "mean_generation_time": np.mean(response_times)
        }
    
    def evaluate_end_to_end(self, test_file: str = "evaluation/questions.json") -> Dict[str, Any]:
        """Évaluation complète du système"""
        logger.info("🚀 Début de l'évaluation du système...")
        
        # Chargement des questions de test
        with open(test_file, 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        logger.info(f"📋 {len(test_questions)} questions chargées pour l'évaluation")
        
        # Vérification que le système est initialisé
        if not self.rag_system.is_initialized:
            self.rag_system.initialize()
        
        # Évaluations
        retrieval_metrics = self.evaluate_retrieval(test_questions)
        generation_metrics = self.evaluate_generation(test_questions)
        
        # Score composite
        composite_score = (
            retrieval_metrics["precision@k"] * 0.4 +
            retrieval_metrics["mean_reciprocal_rank"] * 0.2 +
            generation_metrics["answer_similarity"] * 0.4
        )
        
        results = {
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "composite_score": composite_score,
            "test_set_size": len(test_questions),
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Affichage des résultats
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Affiche les résultats de l'évaluation"""
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE L'ÉVALUATION AGRICULTURE BURKINABÈ")
        print("="*60)
        
        print(f"\n🔍 MÉTRIQUES DE RECHERCHE:")
        print(f"   Précision @3: {results['retrieval_metrics']['precision@k']:.1%}")
        print(f"   Rang Réciproque Moyen: {results['retrieval_metrics']['mean_reciprocal_rank']:.3f}")
        print(f"   Temps Réponse Moyen: {results['retrieval_metrics']['mean_response_time']:.2f}s")
        print(f"   Couverture: {results['retrieval_metrics']['retrieval_coverage']:.1%}")
        
        print(f"\n🤖 MÉTRIQUES DE GÉNÉRATION:")
        print(f"   Similarité des Réponses: {results['generation_metrics']['answer_similarity']:.1%}")
        print(f"   Taux de Succès: {results['generation_metrics']['generation_success_rate']:.1%}")
        print(f"   Temps Génération Moyen: {results['generation_metrics']['mean_generation_time']:.2f}s")
        
        print(f"\n⭐ SCORE COMPOSITE: {results['composite_score']:.1%}")
        print(f"📋 Taille du jeu de test: {results['test_set_size']} questions")
        print(f"🕐 Date d'évaluation: {results['evaluation_timestamp']}")
        print("="*60)

def run_evaluation():
    """Lance l'évaluation complète"""
    evaluator = AgriAssistantEvaluator(rag_system)
    results = evaluator.evaluate_end_to_end()
    
    # Sauvegarde des résultats
    with open('evaluation/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

if __name__ == "__main__":
    run_evaluation()