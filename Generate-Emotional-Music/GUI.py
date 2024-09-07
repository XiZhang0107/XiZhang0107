from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QDesktopWidget
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import subprocess
import os
import sys
import pygame
from PyQt5.QtGui import QPixmapCache


class SpeechRecognitionThread(QThread):
    message = pyqtSignal(str)
    emotion_detected = pyqtSignal(str)  # 新的信号，用于传递检测到的情绪
    midi_generated = pyqtSignal(str)  # 新的信号，用于通知生成了 MIDI 文件

    def run(self):
        try:
            process = subprocess.Popen(
                ["D:\\emotion-recognition-using-speech-master\\venv\\Scripts\\python.exe",
                 "D:\\emotion-recognition-using-speech-master\\emotion-recognition-using-speech-master\\test.py"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd="D:\\emotion-recognition-using-speech-master\\emotion-recognition-using-speech-master"
            )

            emotion = None  # 用于存储检测到的情绪

            # 实时读取输出
            for line in iter(process.stdout.readline, ''):
                self.message.emit(line.strip())

                # 假设情绪结果在一行输出
                if line.strip() in ["sad", "neutral", "happy"]:
                    emotion = line.strip()
                    self.emotion_detected.emit(emotion)  # 发送信号

            process.stdout.close()
            process.wait()

            # 处理情绪检测结果并进行映射
            if emotion:
                emotion_mapped = self.map_emotion(emotion)
                print(f"Mapped emotion: {emotion_mapped}")

                # 运行 EBS.py 并传递映射后的情绪
                self.run_ebs_script(emotion_mapped)
                midi_file_path = "out.mid"
                if os.path.exists(midi_file_path):
                    print("MIDI 文件已生成，路径：", midi_file_path)
                    self.midi_generated.emit(midi_file_path)
                else:
                    print("MIDI 文件未找到。")
        except subprocess.CalledProcessError as e:
            self.message.emit(f"Error: {str(e)}\n{e.output}")
        except Exception as ex:
            self.message.emit(f"Unexpected Error: {str(ex)}")

    def map_emotion(self, emotion):
        """将识别到的情绪映射到 positive, negative 或 neutral"""
        if emotion == "happy":
            return "positive"
        elif emotion == "sad":
            return "negative"
        else:
            return "neutral"

    def run_ebs_script(self, mapped_emotion):
        """运行 EBS.py 并传递映射后的情绪作为参数"""
        try:
            ebs_command = [
                "python",
                "D:\\Generate-Emotional-Music-main\\Generate-Emotional-Music-main\\GRU_EBS.py",
                "--emotion",
                mapped_emotion
            ]
            subprocess.run(ebs_command, check=True)
            print(f"EBS.py successfully executed with emotion: {mapped_emotion}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run EBS.py: {e}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和图标
        self.setWindowTitle("Speech Emotion Recognition Interface")
        self.setWindowIcon(QIcon('yinfu.webp'))

        # 设置窗口大小
        self.resize(1200, 700)

        # 调用方法将窗口居中
        self.center()

        # 创建主布局
        main_layout = QVBoxLayout()

        # 添加波形图像
        self.wave_label = QLabel(self)
        wave_pixmap = QPixmap('bo3.png')  # 替换为你生成的波形图像路径
        wave_pixmap = wave_pixmap.scaled(1000, 200, Qt.KeepAspectRatio)
        self.wave_label.setPixmap(wave_pixmap)
        self.wave_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.wave_label)

        # 添加欢迎文本，设置紧贴波形图像
        self.text_label = QLabel("Welcome to music generation System", self)
        font = QFont("Arial", 18, QFont.Bold)  # 设置字体、大小和粗体
        self.text_label.setFont(font)
        self.text_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)  # 设置对齐方式
        main_layout.addWidget(self.text_label, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # 创建一个新的布局来放置麦克风图标和按钮
        left_top_layout = QVBoxLayout()

        # 创建麦克风图标
        self.icon_label = QLabel(self)
        pixmap = QPixmap('maike.webp')  # 替换为你的麦克风图标路径
        if not pixmap.isNull():  # 检查图像是否加载成功
            scaled_pixmap = pixmap.scaled(44, 44, Qt.KeepAspectRatio)  # 调整图标大小并保持纵横比
            self.icon_label.setPixmap(scaled_pixmap)
        else:
            self.icon_label.setText("图标加载失败")
        left_top_layout.addWidget(self.icon_label, alignment=Qt.AlignLeft)

        # 创建按钮并设置固定大小
        self.button = QPushButton("Click to talk", self)
        self.button.setFixedSize(200, 40)
        left_top_layout.addWidget(self.button, alignment=Qt.AlignLeft)

        # 创建一个主布局来放置整个内容
        content_layout = QHBoxLayout()
        content_layout.addLayout(left_top_layout)
        content_layout.addStretch()  # 让剩余部分填充在右侧

        # 将内容布局添加到主布局中
        main_layout.addLayout(content_layout)

        # 创建一个新的标签用于显示结果，并将其添加到主布局
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        # 连接按钮点击事件
        self.button.clicked.connect(self.start_speech_recognition)

        # 添加音乐图片和播放按钮（初始隐藏）
        self.music_label = QLabel(self)
        self.music_label.setFixedSize(1000, 300)
        # QPixmapCache.clear()

        music_pixmap = QPixmap('cropped_output.png')  # 替换为你的音乐图标路径
        self.music_label.setPixmap(music_pixmap)
        self.music_label.setAlignment(Qt.AlignCenter)
        self.music_label.hide()  # 初始隐藏

        self.play_button = QPushButton("Click to play", self)
        self.play_button.setFixedSize(200, 40)
        self.play_button.setIcon(QIcon('play_icon.webp'))  # 替换为播放按钮图标路径
        self.play_button.hide()  # 初始隐藏
        self.play_button.clicked.connect(self.play_music)

        main_layout.addWidget(self.music_label)
        main_layout.addWidget(self.play_button)  # 位置：添加音乐图片和播放按钮

        # 初始化 pygame
        pygame.mixer.init()

        # 设置布局的对齐方式
        self.setLayout(main_layout)

        # Initialize file modification watcher
        self.setup_file_modification_watcher()

    def setup_file_modification_watcher(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_file_modification)
        self.timer.start(2000)  # 检查文件修改的时间间隔设为2000毫秒

    def check_file_modification(self):
        file_path = 'cropped_output.png'
        try:
            current_mod_time = os.path.getmtime(file_path)
            if not hasattr(self, 'last_mod_time') or self.last_mod_time < current_mod_time:
                self.last_mod_time = current_mod_time
                self.update_music_sheet()
        except FileNotFoundError:
            pass  # 文件未找到时忽略

    def update_music_sheet(self):
        music_pixmap = QPixmap('cropped_output.png')
        self.music_label.setPixmap(music_pixmap.scaled(1000, 300, Qt.KeepAspectRatio))
        self.music_label.show()

    def center(self):
        # 获取屏幕的矩形大小
        qr = self.frameGeometry()
        # 获取显示器的中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 将矩形的中心点移动到屏幕的中心
        qr.moveCenter(cp)
        # 将窗口的左上角移动到qr矩形的左上角，从而使窗口居中
        self.move(qr.topLeft())



    def closeEvent(self, event):
     file_path = 'cropped_output.png'
     if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
     event.accept()  # 关闭窗口

    def start_speech_recognition(self):
        self.thread = SpeechRecognitionThread()
        self.thread.message.connect(self.update_label)
        self.thread.emotion_detected.connect(self.update_emotion)
        self.thread.midi_generated.connect(self.show_music_controls)
        self.thread.start()

    def update_label(self, text):
        self.result_label.setText(text)

    def update_emotion(self, emotion):
        self.result_label.setText(f"Detected Emotion: {emotion}")

    def show_music_controls(self, midi_file_path):
        """当生成 MIDI 文件时显示音乐图片和播放按钮"""
        self.music_label.show()
        self.play_button.show()
        self.midi_file_path = midi_file_path

    def play_music(self):
        """播放生成的 MIDI 文件"""
        if hasattr(self, 'midi_file_path') and os.path.exists(self.midi_file_path):
            pygame.mixer.music.load(self.midi_file_path)
            pygame.mixer.music.play()
        else:
            self.result_label.setText("MIDI 文件未找到，无法播放。")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())