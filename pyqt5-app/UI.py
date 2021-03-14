from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton, QPlainTextEdit, QLabel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QSizePolicy, QFrame
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.Qt import QUrl, QDesktopServices
from PyQt5.QtCore import Qt
try:
    from model_for_app import CassavaModel # required tensorflow
except:
    pass


labels = {"Cassava Bacterial Blight (CBB)": (0, "https://en.wikipedia.org/wiki/Bacterial_blight_of_cassava"),
          "Cassava Brown Streak Disease (CBSD)": (1, "https://en.wikipedia.org/wiki/Cassava_brown_streak_virus_disease"),
          "Cassava Green Mottle (CGM)": (2, "https://en.wikipedia.org/wiki/Cassava_green_mottle_virus"),
          "Cassava Mosaic Disease (CMD)": (3, "https://en.wikipedia.org/wiki/Cassava_mosaic_virus"),
          "Healthy": (4, "https://en.wikipedia.org/wiki/Cassava")}


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.image_wh = 384
        self.setWindowTitle("Cassava Leaf")

        self.main_layout = QVBoxLayout()
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)

        self.separatorLine1 = QFrame()
        self.separatorLine1.setFrameShape(QFrame.HLine)
        self.separatorLine2 = QFrame()
        self.separatorLine2.setFrameShape(QFrame.HLine)

        self.first_part()
        self.second_part()
        self.third_part()
        
        self.setCentralWidget(self.main_widget)
        try:
            self.model = CassavaModel()
        except:
            pass


    def first_part(self):
        """
        First layout with 'help' button
        """
        self.first_layout = QHBoxLayout()
        self.first_layout.addStretch(1)
        
        self.info_button = QPushButton("Help")
        self.info_button.clicked.connect(self.help)
        self.info_button.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.first_layout.addWidget(self.info_button)

        self.main_layout.addLayout(self.first_layout)


    def second_part(self):
        """
        Second layout of uploading photo.
        """
        self.first_frame = QFrame()
        self.first_frame.setFrameShape(QFrame.StyledPanel)
        self.second_layout = QHBoxLayout()

        self.first_left_layout = QVBoxLayout()
        self.upload_text = QPlainTextEdit("address of photo")
        self.upload_text.setFixedHeight(24*2)
        self.upload_text.setFixedWidth(self.image_wh)
        self.first_left_layout.addWidget(self.upload_text)

        self.choosen_image = QLabel()
        self.choosen_image.setPixmap(QPixmap("Graphics/unknown.png").scaled(self.image_wh, self.image_wh))
        self.first_left_layout.addWidget(self.choosen_image)
        self.second_layout.addLayout(self.first_left_layout)

        self.first_right_layout = QVBoxLayout()
        self.upload_button = QPushButton("Browse")
        self.upload_button.clicked.connect(self.choose_file)
        self.upload_button.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.first_right_layout.addWidget(self.upload_button)
        self.first_right_layout.addStretch(1)
        
        self.check_button = QPushButton("Confirm")
        self.check_button.clicked.connect(self.confirm_file)
        self.check_button.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.first_right_layout.addWidget(self.check_button)
        self.second_layout.addLayout(self.first_right_layout)

        self.first_frame.setLayout(self.second_layout)
        self.main_layout.addWidget(self.first_frame)


    def third_part(self):
        """
        Third layout of showing results (dissease and link to wiki).
        """
        self.second_frame = QFrame()
        self.second_frame.setFrameShape(QFrame.StyledPanel)
        self.third_layout = QHBoxLayout()

        self.second_left_frame = QVBoxLayout()
        self.disease_name = QPlainTextEdit("name of disease")
        self.disease_name.setFixedWidth(self.image_wh)
        self.disease_name.setEnabled(False)
        self.disease_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.disease_name.setFixedHeight(24)
        self.second_left_frame.addWidget(self.disease_name)
        
        self.disease_picture = QLabel()
        self.disease_picture.setPixmap(QPixmap("Graphics/known.png").scaled(self.image_wh, self.image_wh))
        self.second_left_frame.addWidget(self.disease_picture)
        self.third_layout.addLayout(self.second_left_frame)

        self.second_right_frame = QVBoxLayout()
        self.second_right_frame.addStretch(1)
        self.disease_desc = QPushButton("about the disease")
        self.disease_desc.setEnabled(False)
        self.disease_desc.clicked.connect(self.open_wiki)
        self.disease_desc.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.second_right_frame.addWidget(self.disease_desc)
        self.third_layout.addLayout(self.second_right_frame)
        

        self.second_frame.setLayout(self.third_layout)
        self.main_layout.addWidget(self.second_frame)


    def help(self):
        """
        Opening popup window with useful informations.
        """
        msg = QMessageBox()
        msg.setWindowTitle("Help window")
        line_1 = "Press 'Browse' button and select photo of leaf."
        line_2 = "When you choosen the photo, press 'Confirm' button."
        line_3 = "In the lower part, you will see a leaf disease."
        line_4 = "You can see a book example of the disease"
        line_5 = "and read more on Wikipedia."
        lines = [line_1, line_2, line_3, line_4, line_5]
        msg.setText("\n".join(lines))
        msg.setIcon(QMessageBox.Information)
        msg.exec_()


    def resize_pixmap(self, pixmap, new_wh):
        """
        Resizing pixmap to the same width and height (new_wh).
        """
        return pixmap.scaled(new_wh, new_wh)


    def choose_file(self):
        """
        Choosing the photo with leaf from drives.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Choose the photo", "c\\",
                                              "png files (*.png *.jpg)")
        if path != "":
            self.path = path
            self.check_button.setEnabled(True)
            self.upload_text.setPlainText(self.path)
            self.choosen_image.setPixmap(QPixmap(self.path).scaled(self.image_wh, self.image_wh))


    def confirm_file(self):
        """
        Confirming the photo with leaf.
        """
        self.detected_diss, self.acc = self.model.predict(self.path)

        self.disease_name.setEnabled(True)
        self.disease_desc.setEnabled(True)
        self.disease_name.setPlainText(self.detected_diss)
        self.disease_picture.setPixmap(QPixmap(f"Graphics/{labels[self.detected_diss][0]}.png").scaled(self.image_wh, self.image_wh))


    def open_wiki(self):
        """
        Opening wiki page with dissease of leaf.
        """
        url = labels[self.detected_diss][1]
        url = QUrl(url)
        QDesktopServices.openUrl(url)
        
        
        
                
def run():
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec_()


if __name__ == "__main__":
    run()
