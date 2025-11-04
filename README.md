# Hackaton
projet agriIA


# 🌱 Agri Assistant Burkina

Assistant IA contextuel 100% open source pour l'agriculture burkinabè.

## 🎯 Objectif

Développer un système d'IA capable de répondre à des questions sur l'agriculture burkinabè en utilisant exclusivement des technologies open source.

## 🏗️ Architecture Technique



### Pipeline RAG

1. **Question** → Encodage en embeddings
2. **Recherche** → Similarité vectorielle avec FAISS
3. **Contexte** → Extraction des documents pertinents
4. **Génération** → Réponse contextuelle avec Mistral-7B
5. **Réponse** → Retour avec sources citées

## 🛠️ Technologies Open Source Utilisées

### Composants Principaux

| Composant | Technologie | Licence | Lien |
|-----------|-------------|---------|------|
| **Embeddings** | sentence-transformers | Apache 2.0 | [Lien](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| **Vector DB** | FAISS | MIT | [Lien](https://github.com/facebookresearch/faiss) |
| **LLM** | Mistral-7B | Apache 2.0 | [Lien](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |
| **Backend** | FastAPI | MIT | [Lien](https://fastapi.tiangolo.com) |
| **Frontend** | Streamlit | MIT | [Lien](https://streamlit.io) |

### Bibliothèques Support

- **Transformers** (Apache 2.0) - Modèles de langage
- **PyTorch** (BSD) - Calcul tensoriel
- **NumPy** (BSD) - Calcul scientifique
- **Pandas** (BSD) - Manipulation de données

## 🚀 Installation et Utilisation

### Prérequis

- Python 3.8+
- 8GB RAM minimum
- 2GB espace disque

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/agri-assistant.git
cd agri-assistant
