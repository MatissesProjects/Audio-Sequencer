import json
from typing import List, Optional, Any
from src.core.models import TrackSegment

class UndoManager:
    def __init__(self) -> None:
        self.undo_stack: List[List[str]] = []
        self.redo_stack: List[List[str]] = []

    def push_state(self, segments: List[TrackSegment]) -> None:
        state = [json.dumps(s.to_dict()) for s in segments]
        self.undo_stack.append(state)
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self, current_segments: List[TrackSegment]) -> Optional[List[str]]:
        if not self.undo_stack:
            return None
        self.redo_stack.append([json.dumps(s.to_dict()) for s in current_segments])
        return self.undo_stack.pop()

    def redo(self, current_segments: List[TrackSegment]) -> Optional[List[str]]:
        if not self.redo_stack:
            return None
        self.undo_stack.append([json.dumps(s.to_dict()) for s in current_segments])
        return self.redo_stack.pop()
