import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QSlider, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIntValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("时间序列图片查看器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 数据存储
        self.data = None
        self.current_index = 0
        self.total_images = 0
        
        # 创建主界面
        self.init_ui()
        
    def init_ui(self):
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # 文件路径输入区域
        file_layout = QHBoxLayout()
        
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("请输入 .npy 文件路径或拖拽文件到窗口...")
        self.file_path_input.setMinimumWidth(400)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_file)
        
        load_button = QPushButton("加载数据")
        load_button.clicked.connect(self.load_data)
        
        file_layout.addWidget(QLabel("数据文件:"))
        file_layout.addWidget(self.file_path_input)
        file_layout.addWidget(browse_button)
        file_layout.addWidget(load_button)
        file_layout.addStretch()
        
        layout.addLayout(file_layout)
        
        # 图片显示区域
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 控制区域
        control_layout = QHBoxLayout()
        
        # 当前时间显示
        self.time_label = QLabel("时间: 0 / 0")
        control_layout.addWidget(self.time_label)
        
        # 左箭头按钮
        self.prev_button = QPushButton("← 上一帧")
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setEnabled(False)
        control_layout.addWidget(self.prev_button)
        
        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.slider_changed)
        control_layout.addWidget(self.slider)
        
        # 右箭头按钮
        self.next_button = QPushButton("下一帧 →")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        control_layout.addWidget(self.next_button)
        
        # 跳转到指定帧
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("跳转到帧:"))
        
        self.jump_input = QLineEdit()
        self.jump_input.setMaximumWidth(60)
        self.jump_input.setValidator(QIntValidator(0, 999999))
        self.jump_input.returnPressed.connect(self.jump_to_frame)
        
        jump_button = QPushButton("跳转")
        jump_button.clicked.connect(self.jump_to_frame)
        
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(jump_button)
        jump_layout.addStretch()
        
        control_layout.addLayout(jump_layout)
        
        layout.addLayout(control_layout)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 支持拖拽
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            if file_path.endswith('.npy'):
                self.file_path_input.setText(file_path)
                self.load_data()
            else:
                QMessageBox.warning(self, "警告", "请拖拽 .npy 文件")
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "NumPy 文件 (*.npy)")
        if file_path:
            self.file_path_input.setText(file_path)
    
    def load_data(self):
        file_path = self.file_path_input.text()
        if not file_path:
            QMessageBox.warning(self, "警告", "请输入文件路径")
            return
            
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", "文件不存在")
            return
            
        try:
            # 加载数据
            self.data = np.load(file_path)
            
            # 检查数据格式
            if len(self.data.shape) != 3:
                QMessageBox.warning(self, "警告", "数据格式错误，需要 [N, n, n] 的三维数组")
                return
                
            self.total_images = self.data.shape[0]
            self.current_index = 0
            
            # 更新界面
            self.slider.setMaximum(max(0, self.total_images - 2))
            self.update_display()
            self.update_controls()
            
            self.statusBar().showMessage(f"成功加载 {self.total_images} 张图片")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载数据失败: {str(e)}")
    
    def update_display(self):
        if self.data is None or self.total_images < 2:
            return
            
        self.figure.clear()
        
        # 创建子图
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)
        
        # 显示当前帧和下一帧
        img1 = self.data[self.current_index]
        if self.current_index + 1 < self.total_images:
            img2 = self.data[self.current_index + 1]
        else:
            img2 = np.zeros_like(img1)
        
        # 显示图片
        im1 = ax1.imshow(img1, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax1.set_title(f'Frame {self.current_index}')
        ax1.axis('off')
        
        im2 = ax2.imshow(img2, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax2.set_title(f'Frame {self.current_index + 1}')
        ax2.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # 更新标签
        self.time_label.setText(f"时间: {self.current_index} / {self.total_images - 1}")
    
    def update_controls(self):
        has_data = self.data is not None and self.total_images > 0
        self.prev_button.setEnabled(has_data and self.current_index > 0)
        self.next_button.setEnabled(has_data and self.current_index < self.total_images - 2)
        self.slider.setEnabled(has_data)
    
    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.setValue(self.current_index)
            self.update_display()
            self.update_controls()
    
    def next_frame(self):
        if self.current_index < self.total_images - 2:
            self.current_index += 1
            self.slider.setValue(self.current_index)
            self.update_display()
            self.update_controls()
    
    def slider_changed(self, value):
        if self.data is not None:
            self.current_index = value
            self.update_display()
            self.update_controls()
    
    def jump_to_frame(self):
        if self.data is None:
            return
            
        try:
            frame_num = int(self.jump_input.text())
            if 0 <= frame_num < self.total_images - 1:
                self.current_index = frame_num
                self.slider.setValue(self.current_index)
                self.update_display()
                self.update_controls()
            else:
                QMessageBox.warning(self, "警告", f"帧数必须在 0 到 {self.total_images - 2} 之间")
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的帧数")


def main():
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()