# Track Plan: gui-core (UI Canvas & Browser)

## Objective
Build a native desktop interface using PyQt6 that allows users to browse their audio library, view AI-driven recommendations, and arrange tracks on a "Flow" canvas.

## Tasks
1. [x] **Main Window Scaffolding:** Create the base PyQt6 application structure.
2. [x] **Library Browser:** Implement a table/list view for tracks with:
    - Metadata display (BPM, Key).
3. [x] **Flow Canvas (Phase 1):** A simple drag-and-drop area to place a "Current Track".
4. [x] **Recommendation Sidebar:** Real-time updates showing the top 5 compatible tracks based on the multi-dimensional score.
5. [ ] **Playback Controls:** Basic transport (Play/Stop) for the selected track or the "Flow".
6. [ ] **Semantic Search:** Implement the text-to-audio filtering logic.

## Success Metrics
- User can browse their local library and see BPM/Key/Semantic scores visually.
- Dragging a track to the canvas triggers an immediate recommendation list.
