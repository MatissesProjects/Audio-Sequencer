# Track Plan: ai-embed (AI Embedding Engine)

## Objective
Integrate the `laion-clap` model to convert audio clips into 512-dimensional semantic vectors, enabling "vibe-based" search and compatibility scoring.

## Tasks
1. [ ] **Model Integration:** Create `src/embeddings.py` to load the CLAP model locally.
2. [ ] **Embedding Generation:** Implement logic to generate vectors for 30-second audio windows.
3. [ ] **Vector Storage:** Integrate with `DataManager` to store embeddings in ChromaDB.
4. [ ] **Semantic Scoring:** Update `src/scoring.py` to include cosine similarity between vectors.
5. [ ] **Validation:** Verify that "ambient" tracks have higher similarity to each other than to "aggressive" tracks.

## Dependencies
- Requires `torch`, `laion-clap`, and `transformers`.
