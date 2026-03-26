# Infinite Radio Stream Plan

## Overview
Create a continuous, generative "Infinite Radio" that dynamically orchestrates, renders, and streams audio endlessly without requiring an upfront render of the entire sequence.

## Approach
1. **Infinite Orchestration (`src/orchestrator.py` & new `src/infinite_orchestrator.py`)**
   - Refactor `FullMixOrchestrator` logic to generate segments in "depth" batches (e.g., 2-3 minutes at a time).
   - Maintain a running state of the sequence to ensure the next batch continues seamlessly from the end of the previous one (key compatibility, tempo matching, segment roles).

2. **Chunked Streaming Renderer (`src/renderer.py`)**
   - The current `FlowRenderer` renders the whole timeline to a single numpy array and exports to MP3. This will cause OOM for an infinite stream.
   - Introduce a `render_chunk` or `yield_chunks` method that only evaluates a specific `time_range` (e.g., `current_time` to `current_time + 10s`).
   - The renderer will look at active segments within that time window, apply ducking/sidechaining/FX dynamically, and yield the raw numpy audio block.

3. **Real-time Audio Playback Engine (`src/streaming_player.py`)**
   - Add `sounddevice` to `requirements.txt`.
   - Create a background thread that manages a thread-safe Queue of rendered audio chunks.
   - Use `sounddevice.OutputStream` with a callback that reads from the queue to ensure zero-latency, gapless playback.
   - Include a buffer management system to keep the orchestrator/renderer 30-60 seconds ahead of playback.

4. **UI Integration (`src/ui/main_window.py`)**
   - Add an "Infinite Radio" button to the main toolbar.
   - Create a dynamic "Now Playing / Up Next" panel that updates as new batches are orchestrated.
   - Tie the start/stop logic to the streaming background threads.

## Steps
1. [ ] Install and configure `sounddevice` for realtime playback.
2. [ ] Create the `StreamingPlayer` class capable of continuous array queuing.
3. [ ] Update `FlowRenderer` to support windowed chunk rendering.
4. [ ] Create `InfiniteOrchestrator` to generate track segments dynamically and indefinitely.
5. [ ] Wire the pieces together and add GUI controls.
