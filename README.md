
1. **Question** → Saisie de l'utilisateur
2. **Encodage** → Embeddings avec MiniLM multilingue
3. **Recherche** → Similarité vectorielle avec FAISS
4. **Contexte** → Extraction des documents pertinents
5. **Génération** → Réponse contextuelle avec DialoGPT-medium
6. **Réponse** → Retour avec sources citées

## 🛠️ Technologies Open Source Utilisées

### Composants Principaux

| Composant | Technologie | Spécifications | Licence |
|-----------|-------------|----------------|---------|
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | 384 dimensions, multilingue, léger | Apache 2.0 |
| **Vector DB** | **FAISS** (Facebook AI Similarity Search) | Index FlatIP, recherche rapide | MIT |
| **LLM** | **DialoGPT-medium** | 345M paramètres, optimisé dialogue | MIT |
| **Framework ML** | **PyTorch** + **Transformers** | Inférence CPU, gestion mémoire | BSD/Apache 2.0 |
| **Embeddings** | **Sentence-Transformers** | Encodage par batch, normalisation | Apache 2.0 |

### Caractéristiques Techniques

- **🖥️ Compatible CPU** : Fonctionne sur machines 8GB RAM
- **⚡ Optimisations mémoire** : 
  - Encodage par batch (16-32 documents)
  - Limitation contexte (400-600 caractères)
  - Génération contrôlée (100-150 tokens)
- **🌍 Multilingue** : Support français/langues locales
- **💾 Local uniquement** : Aucune connexion internet requise

## 📊 Performance et Optimisations

### Gestion Mémoire
```python
# Encodage optimisé
embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True)

# Génération contrôlée
outputs = model.generate(
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)
