
import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import AudioSequencerApp

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_app_ui_initialization(qapp):
    """Verifies that the main application window can initialize its UI without errors."""
    window = AudioSequencerApp()
    assert window is not None
    assert window.timeline_widget is not None
    assert window.timeline_widget.lane_count == 8
    window.close()

def test_lane_management(qapp):
    """Verifies that lanes can be added and removed correctly via UI buttons."""
    window = AudioSequencerApp()
    initial_lanes = window.timeline_widget.lane_count
    
    # Simulate Add Lane click
    window.timeline_widget.add_lane()
    assert window.timeline_widget.lane_count == initial_lanes + 1
    
    # Simulate Remove Lane click
    window.timeline_widget.remove_lane()
    assert window.timeline_widget.lane_count == initial_lanes
    
    window.close()
