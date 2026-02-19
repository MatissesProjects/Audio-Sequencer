# Track Plan: sys-data (Data Persistence & State)

## Objective
Establish a robust persistence layer for both relational metadata (SQLite) and high-dimensional vector embeddings (ChromaDB).

## Tasks
1. [x] **ChromaDB Initialization:** Setup the local vector database client and create the `audio_embeddings` collection.
2. [x] **SQLite Schema Expansion:** Add fields for transient data (e.g., beat onsets) to support seamless looping and alignment.
3. [x] **Data Access Layer:** Create a unified `DataManager` in `src/database.py` that handles both SQLite and ChromaDB operations.
4. [ ] **State Persistence:** Ensure the system can recover the last scan state and library statistics.

## Next Action (Immediate)
- Expand `src/database.py` to include ChromaDB setup.
- Add transient data extraction to `src/analysis.py`.
