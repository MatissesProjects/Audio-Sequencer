from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QApplication, QMessageBox
import traceback

class DetailedErrorDialog(QDialog):
    def __init__(self, title, message, details, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        msg_layout = QHBoxLayout()
        
        icon_label = QLabel("❌")
        icon_label.setStyleSheet("font-size: 32px;")
        msg_layout.addWidget(icon_label)
        
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        msg_layout.addWidget(msg_label, stretch=1)
        
        layout.addLayout(msg_layout)
        
        l = QLabel("Technical Details:")
        l.setStyleSheet("color: #aaa;")
        layout.addWidget(l)
        
        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setText(details)
        self.details_box.setStyleSheet("background-color: #1a1a1a; color: #ff5555; font-family: Consolas, monospace; border: 1px solid #333;")
        layout.addWidget(self.details_box)
        
        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        btn_layout.addWidget(copy_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        self.setStyleSheet("QDialog { background-color: #252525; } QPushButton { background-color: #444; color: white; padding: 8px; border-radius: 4px; }")

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.details_box.toPlainText())
        QMessageBox.information(self, "Copied", "Error details copied to clipboard.")

def show_error(parent, title, message, exception):
    details = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    dialog = DetailedErrorDialog(title, message, details, parent)
    dialog.exec()
