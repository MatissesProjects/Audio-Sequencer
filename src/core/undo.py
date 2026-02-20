import json

class UndoManager:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push_state(self, segments):
        state = [json.dumps(s.to_dict()) for s in segments]
        self.undo_stack.append(state)
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self, current_segments):
        if not self.undo_stack:
            return None
        self.redo_stack.append([json.dumps(s.to_dict()) for s in current_segments])
        return self.undo_stack.pop()

    def redo(self, current_segments):
        if not self.redo_stack:
            return None
        self.undo_stack.append([json.dumps(s.to_dict()) for s in current_segments])
        return self.redo_stack.pop()
