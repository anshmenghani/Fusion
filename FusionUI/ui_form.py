# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

# PySide6 User Interface for the Fusion Model 

import os 
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QLabel, QLineEdit,
    QProgressBar, QPushButton, QSizePolicy, QStackedWidget,
    QWidget)
import numpy as np
import sys 
sys.path.append('/Users/anshmenghani/Documents/GitHub/Fusion/src/FUSION')
import prediction
import webbrowser
    
class Ui_Fusion(object):
    def set_vars(self):
        self.inputs = None
        self.name = "Modeled Star"
        self.outputs = None
        self.error = ""

    def conversion(self, teff, lum, rad):
        try:
            try:
                float(teff)
            except ValueError:
                teff = 5778 * ((float(lum)/(float(rad)**2)) ** (1/4))
            try:
                float(lum)
            except ValueError:
                lum = (float(rad)**2) * ((float(teff)/5778)**4)
            try:
                float(rad)
            except ValueError:
                rad = (float(lum)/((float(teff)/5778)**4)) ** (1/2)
        except ValueError:
            self.error = "Enter At Least Two Valid Numbers"

        return float(teff), float(lum), float(rad)

    def min_max(self, num):
        if num > 9:
            return 9
        elif num < 0:
            return 0
        else:
            return num
        
    def get_spec_class(self, c, t):
        c = int(c)
        if c == 0:
            return "M" + str(self.min_max(int(np.floor((t - 4000) / 160))))
        elif c == 1:
            return "K" + str(self.min_max(int(np.floor((t - 5200) / 120))))
        elif c == 2:
            return "G" + str(self.min_max(int(np.floor((t - 7000) / 200))))
        elif c == 3:
            return "F" + str(self.min_max(int(np.floor((t - 12000) / 500))))
        elif c == 4:
            return "A" + str(self.min_max(int(np.floor((t - 20000) / 800))))
        elif c == 5:
            return "B" + str(self.min_max(int(np.floor((t - 34000) / 1400))))
        elif c == 6:
            return "O" + str(self.min_max(int(np.floor((t - 100000) / 7600))))
        
    def get_lum_class(self, l):
        l = int(l)
        if l == 0:
            return "D"
        elif l == 6:
            return "V"
        elif l == 5:
            return "IV"
        elif l == 4:
            return "III"
        elif l == 3:
            return "II"
        elif l == 2:
            return "Ib"
        elif l == 1:
            return "Ia"
        
    def get_star_type(self, t):
        t = int(t)
        if t == 0:
            return "Brown Dwarf"
        elif t == 1:
            return "Red Dwarf"
        elif t == 2:
            return "White Dwarf"
        elif t == 3:
            return "Main Sequence"
        elif t == 4:
            return "Supergiant"
        elif t == 5:
            return "Hypergiant"
   
    def run_fusion_from_inputs(self):
        self.progressBar.setValue(5)
        teff = self.lineEdit.text()
        lum = self.lineEdit_2.text()
        rad = self.lineEdit_3.text()
        self.name = self.lineEdit_4.text()
        self.progressBar.setValue(15)

        teff, lum, rad = self.conversion(teff, lum, rad)       
        self.inputs = np.array([teff, lum, rad, rad, rad**3, rad**2, rad, rad**2])
        self.progressBar.setValue(40)
        self.run_fusion(u=True)

    def run_fusion(self, u=False):
        if self.inputs.shape[-1] == 9:
            self.outputs = prediction.model(self.inputs[:, -8:])
        else:
            self.outputs = prediction.model(self.inputs)
        self.progressBar.setValue(80)
        if u:
            self.update_output_text()
            self.stackedWidget.setCurrentIndex(1)
        self.progressBar.setValue(100)
        self.progressBar.setValue(0)

    def import_csv(self):
        file_path, _ = QFileDialog.getOpenFileName()
        self.progressBar.setValue(15)
        if os.path.exists(file_path) and str(file_path).endswith(".csv"):
            self.inputs = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            self.progressBar.setValue(35)
            self.run_fusion()
            self.progressBar.setValue(75)
            self.export_csv()
            self.progressBar.setValue(100)
            self.progressBar.setValue(0)
        else:
            self.error = "Invalid File Path or File Type"

    def export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName()
        if self.inputs.ndim != 2:
            ins = self.inputs.reshape((1, -1))
            outs = self.outputs.reshape((1, -1))
            combined = np.hstack((ins, outs), dtype="str")
            combined = combined.reshape((1, -1))
        else:
            if self.inputs.shape[-1] > 8:
                combined = np.hstack((self.inputs[:, -8:], self.outputs), dtype="str")
        bcs = combined[:, 9].astype(np.float32) - combined[:, 8].astype(np.float32)
        combined = np.insert(combined, 10, bcs.astype(str), axis=1)
        combined = np.column_stack((combined, [np.nan for _ in combined]))
        combined[:, 21] = np.vectorize(self.get_spec_class)(combined[:, 21].astype(np.float32), combined[:, 0].astype(np.float32))
        combined[:, 22] = np.vectorize(self.get_lum_class)(combined[:, 22].astype(np.float32))
        combined[:, 24] = np.vectorize(self.get_star_type)(combined[:, 24].astype(np.float32))
        combined[:, 25] = np.char.add(combined[:, 21].astype("str"), combined[:, 22].astype("str"))
        
        if self.inputs.shape[-1] > 8:
            combined = np.insert(combined, 0, self.inputs[:, 0], axis=1)
        else:
            combined = np.insert(combined, 0, self.name, axis=1)

        header_vals = ",EffectiveTemperature(Teff)(K),Luminosity(L/Lo),Radius(R/Ro),Diameter(D/Do),Volume(V/Vo),SurfaceArea(SA/SAo),GreatCircleCircumference(GCC/GCCo),GreatCircleArea(GCA/GCAo),AbsoluteBolometricMagnitude(Mbol),AbsoluteMagnitude(M)(Mv),BolometricCorrection(BC)(mag),AbsoluteBolometricLuminosity(Lbol)(log(W)),Mass(M/Mo),AverageDensity(D/Do),CentralPressure(log(N/m^2)),CentralTemperature(log(K)),Lifespan(SL/SLo),SurfaceGravity(log(g)...log(N/kg)),GravitationalBindingEnergy(log(J)),BolometricFlux(log(W/m^2)),Metallicity(log(MH/MHo)),SpectralClass,LuminosityClass,StarPeakWavelength(nm),StarType,StellarClassification"
        if os.path.exists(os.path.dirname(file_path)):
            np.savetxt(str(file_path) + ".csv", combined, delimiter=',', header=header_vals, comments='', fmt='%s')
        else:
            self.error = "Invalid Specified Path"

    def open_instructions(self):
        webbrowser.open_new_tab("https://github.com/anshmenghani/Fusion")

    def get_image(self, st):
        # add example label
        if st == "Brown Dwarf":
            return "/Users/anshmenghani/Documents/GitHub/Fusion/FusionUI/images/star_examples/brown-dwarf.png"
        elif st == "Red Dwarf":
            return "/Users/anshmenghani/Documents/GitHub/Fusion/FusionUI/images/star_examples/red-dwarf.png"
        elif st == "White Dwarf":
            return "/Users/anshmenghani/Documents/GitHub/Fusion/FusionUI/images/star_examples/white-dwarf.png"
        elif st == "Main Sequence":
            return "/Users/anshmenghani/Documents/GitHub/Fusion/FusionUI/images/star_examples/main-sequence.png"
        elif st == "Supergiant":
            return "/Users/anshmenghani/Documents/GitHub/Fusion/FusionUI/images/star_examples/supergiant.png"
        elif st == "Hypergiant":
            return "/Users/anshmenghani/Documents/GitHub/Fusion/FusionUI/images/star_examples/hypergiant.png"


    def update_output_text(self):
        inputs = [round(i, 3) for i in self.inputs]
        outputs = np.round(self.outputs, decimals=3)
        self.label_22.setText(QCoreApplication.translate("Fusion", f"{self.name}", None))
        self.label_25.setText(QCoreApplication.translate("Fusion", f"Great Circle Area (GCA/GCA\u2299): {inputs[7]}", None))
        self.label_26.setText(QCoreApplication.translate("Fusion", f"Absolute Bolometric Luminosity (Lbol)(log(W)): {outputs[0][2]}", None))
        self.label_24.setText(QCoreApplication.translate("Fusion", f"Great Circle Circumference (GCC/GCC\u2299): {inputs[6]}", None))
        self.label_33.setText(QCoreApplication.translate("Fusion", f"Average Density (D/D\u2299): {outputs[0][4]}", None))
        self.label_15.setText(QCoreApplication.translate("Fusion", f"Effective Temperature (K): {inputs[0]}", None))
        self.label_23.setText(QCoreApplication.translate("Fusion", f"Surface Area (SA/SA\u2299): {inputs[5]}", None))
        self.label_27.setText(QCoreApplication.translate("Fusion", f"Absolute Bolometric Magnitude (Mbol): {outputs[0][0]}", None))
        self.label_16.setText(QCoreApplication.translate("Fusion", f"Radius (R/R\u2299): {inputs[2]}", None))
        self.label_43.setText(QCoreApplication.translate("Fusion", f"Stellar Classification: {str(self.get_spec_class(outputs[0][12], inputs[0])) + str(self.get_lum_class(outputs[0][13]))}", None))
        self.label_38.setText(QCoreApplication.translate("Fusion", f"Surface Gravity (log(g)...log(N/kg)): {outputs[0][8]}", None))
        self.label_35.setText(QCoreApplication.translate("Fusion", f"Spectral Class: {self.get_spec_class(outputs[0][12], inputs[0])}", None))
        self.label_17.setText(QCoreApplication.translate("Fusion", f"Luminosity (L/L\u2299): {inputs[1]}", None))
        self.label_40.setText(QCoreApplication.translate("Fusion", f"Star Peak Wavelength (nm): {outputs[0][14]}", None))
        self.label_31.setText(QCoreApplication.translate("Fusion", f"Central Pressure (log(N/m^2)): {outputs[0][5]}", None))
        self.label_34.setText(QCoreApplication.translate("Fusion", f"Luminosity class: {self.get_lum_class(outputs[0][13])}", None))
        self.label_28.setText(QCoreApplication.translate("Fusion", f"Absolute Magnitude (M)(Mv): {outputs[0][1]}", None))
        self.label_44.setText(QCoreApplication.translate("Fusion", f"Star Type: {self.get_star_type(outputs[0][15])}", None))
        self.label_39.setText(QCoreApplication.translate("Fusion", f"Bolometric Correction: {round(outputs[0][1]-outputs[0][0], 3)}", None))
        self.label_20.setText(QCoreApplication.translate("Fusion", f"Diameter (D/D\u2299): {inputs[3]}", None))
        self.label_42.setText(QCoreApplication.translate("Fusion", f"Metallicity (MH/MH\u2299): {outputs[0][11]}", None))
        self.label_41.setText(QCoreApplication.translate("Fusion", f"Gravitational Binding Energy (log(J)): {outputs[0][9]}", None))
        self.label_32.setText(QCoreApplication.translate("Fusion", f"Mass (M/M\u2299): {outputs[0][3]}", None))
        self.label_30.setText(QCoreApplication.translate("Fusion", f"Centeral Temperature (log(K)): {outputs[0][6]}", None))
        self.label_37.setText(QCoreApplication.translate("Fusion", f"Bolometric Flux (log(W/m^2)): {outputs[0][10]}", None))
        self.label_36.setText(QCoreApplication.translate("Fusion", f"Lifespan (SL/SL\u2299): {outputs[0][7]}", None))
        self.label_45.setText(QCoreApplication.translate("Fusion", f"Volume (V/V\u2299): {inputs[4]}", None))
        self.label_13.setPixmap(QPixmap(self.get_image(self.get_star_type(outputs[0][15]))))
        self.label_14.setText("Example " + self.get_star_type(outputs[0][15]) + " Star")
    
    def clear_and_reset(self):
        self.inputs = None
        self.outputs = None
        self.name = "Modeled Star"
        self.error = ""
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")
        self.lineEdit_3.setText("")
        self.lineEdit_4.setText("")
        self.progressBar.setValue(0)
        self.stackedWidget.setCurrentIndex(0)
        self.label_22.setText(QCoreApplication.translate("Fusion", u"Modeled Star", None))
        self.label_25.setText(QCoreApplication.translate("Fusion", u"Great Circle Area (GCA/GCA\u2299): ", None))
        self.label_26.setText(QCoreApplication.translate("Fusion", u"Absolute Bolometric Luminosity (Lbol)(log(W)): ", None))
        self.label_24.setText(QCoreApplication.translate("Fusion", u"Great Circle Circumference (GCC/GCC\u2299): ", None))
        self.label_33.setText(QCoreApplication.translate("Fusion", u"Average Density (D/D\u2299): ", None))
        self.label_15.setText(QCoreApplication.translate("Fusion", u"Effective Temperature (K): ", None))
        self.label_23.setText(QCoreApplication.translate("Fusion", u"Surface Area (SA/SA\u2299): ", None))
        self.label_27.setText(QCoreApplication.translate("Fusion", u"Absolute Bolometric Magnitude (Mbol): ", None))
        self.label_16.setText(QCoreApplication.translate("Fusion", u"Radius (R/R\u2299): ", None))
        self.label_43.setText(QCoreApplication.translate("Fusion", u"Stellar Classification: ", None))
        self.label_38.setText(QCoreApplication.translate("Fusion", u"Surface Gravity (log(g)...log(N/kg)): ", None))
        self.label_35.setText(QCoreApplication.translate("Fusion", u"Spectral Class: ", None))
        self.label_17.setText(QCoreApplication.translate("Fusion", u"Luminosity (L/L\u2299): ", None))
        self.label_40.setText(QCoreApplication.translate("Fusion", u"Star Peak Wavelength (nm): ", None))
        self.label_31.setText(QCoreApplication.translate("Fusion", u"Central Pressure (log(N/m^2)): ", None))
        self.label_34.setText(QCoreApplication.translate("Fusion", u"Luminosity class: ", None))
        self.label_28.setText(QCoreApplication.translate("Fusion", u"Absolute Magnitude (M)(Mv): ", None))
        self.label_44.setText(QCoreApplication.translate("Fusion", u"Star Type: ", None))
        self.label_39.setText(QCoreApplication.translate("Fusion", u"Bolometric Correction: ", None))
        self.label_20.setText(QCoreApplication.translate("Fusion", u"Diameter (D/D\u2299): ", None))
        self.label_42.setText(QCoreApplication.translate("Fusion", u"Metallicity (MH/MH\u2299): ", None))
        self.label_41.setText(QCoreApplication.translate("Fusion", u"Gravitational Binding Energy (log(J)): ", None))
        self.label_32.setText(QCoreApplication.translate("Fusion", u"Mass (M/M\u2299): 0.811", None))
        self.label_30.setText(QCoreApplication.translate("Fusion", u"Centeral Temperature (log(K)): ", None))
        self.label_37.setText(QCoreApplication.translate("Fusion", u"Bolometric Flux (log(W/m^2)): ", None))
        self.label_36.setText(QCoreApplication.translate("Fusion", u"Lifespan (SL/SL\u2299): ", None))
        self.label_22.setText(QCoreApplication.translate("Fusion", u"Modeled Star", None))
        self.label_45.setText(QCoreApplication.translate("Fusion", u"Volume (V/V\u2299): ", None))

    def setupUi(self, Fusion):
        if not Fusion.objectName():
            Fusion.setObjectName(u"Fusion")
        Fusion.resize(450, 625)
        Fusion.setAutoFillBackground(False)
        Fusion.setWindowIcon(QIcon('images/FusionIcon.svg'))
        self.stackedWidget = QStackedWidget(Fusion)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setGeometry(QRect(0, 0, 461, 651))
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.pushButton_3 = QPushButton(self.page_3)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(130, 80, 31, 32))
        self.pushButton_3.setStyleSheet(u"color: rgb(221, 251, 210);\n"
"")
        self.pushButton_3.clicked.connect(self.open_instructions)
        self.label_3 = QLabel(self.page_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(85, 160, 191, 16))
        self.label_3.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_4 = QLabel(self.page_3)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(85, 190, 191, 16))
        self.label_4.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label = QLabel(self.page_3)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(57, 45, 341, 31))
        self.label.setStyleSheet(u"font: 24pt \"Bodoni 72\";\n"
"color: rgb(188, 247, 236)")
        self.label_10 = QLabel(self.page_3)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(85, 250, 191, 16))
        self.label_10.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.pushButton_2 = QPushButton(self.page_3)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(210, 80, 31, 32))
        self.pushButton_2.clicked.connect(self.import_csv)
        self.model = QPushButton(self.page_3)
        self.model.setObjectName(u"model")
        self.model.clicked.connect(self.run_fusion_from_inputs)
        self.model.setGeometry(QRect(175, 287, 100, 32))
        self.model.setStyleSheet(u"color: rgb(107, 127, 215);")
        self.lineEdit = QLineEdit(self.page_3)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(275, 160, 91, 21))
        self.lineEdit.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"\n"
"")
        self.pushButton_4 = QPushButton(self.page_3)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(290, 80, 31, 32))
        self.label_8 = QLabel(self.page_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(298, 87, 15, 15))
        self.label_8.setPixmap(QPixmap(u"images/icons/settings.svg"))
        self.label_8.setScaledContents(True)
        self.label_9 = QLabel(self.page_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(218, 87, 15, 15))
        self.label_9.setPixmap(QPixmap(u"images/icons/upload.svg"))
        self.label_9.setScaledContents(True)
        self.label_5 = QLabel(self.page_3)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(85, 220, 191, 16))
        self.label_5.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.lineEdit_2 = QLineEdit(self.page_3)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setGeometry(QRect(275, 190, 91, 21))
        self.lineEdit_2.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"")
        self.lineEdit_3 = QLineEdit(self.page_3)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setGeometry(QRect(275, 220, 91, 21))
        self.lineEdit_3.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"")
        self.label_2 = QLabel(self.page_3)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(82, 130, 291, 16))
        self.label_2.setStyleSheet(u"font: 16pt \"Andale Mono\";")
        self.frame = QFrame(self.page_3)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(28, 27, 391, 331))
        self.frame.setStyleSheet(u"background-color: rgb(76, 42, 133); border-radius: 10px")
        self.label_6 = QLabel(self.frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 10, 371, 311))
        self.label_6.setPixmap(QPixmap(u"images/stars.png"))
        self.label_6.setScaledContents(True)

        self.progressBar = QProgressBar(self.page_3)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(168, 317, 118, 23))
        self.progressBar.setValue(0)
        self.label_7 = QLabel(self.page_3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(138, 87, 15, 15))
        self.label_7.setPixmap(QPixmap(u"images/icons/info.svg"))
        self.label_7.setScaledContents(True)
        self.lineEdit_4 = QLineEdit(self.page_3)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setGeometry(QRect(275, 250, 91, 21))
        self.lineEdit_4.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"")
        self.stackedWidget.addWidget(self.page_3)
        self.frame.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label.raise_()
        self.label_10.raise_()
        self.pushButton_2.raise_()
        self.model.raise_()
        self.lineEdit.raise_()
        self.lineEdit_2.raise_()
        self.pushButton_4.raise_()
        self.label_5.raise_()
        self.lineEdit_3.raise_()
        self.label_2.raise_()
        self.progressBar.raise_()
        self.label_7.raise_()
        self.lineEdit_4.raise_()
        self.pushButton_3.raise_()
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.label_25 = QLabel(self.page_4)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(30, 240, 221, 16))
        self.label_25.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_26 = QLabel(self.page_4)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(30, 320, 231, 16))
        self.label_26.setStyleSheet(u"font: 9pt \"Avenir\";")
        self.label_24 = QLabel(self.page_4)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(30, 220, 251, 20))
        self.label_24.setStyleSheet(u"font: 10pt \"Avenir\";")
        self.label_33 = QLabel(self.page_4)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setGeometry(QRect(30, 360, 191, 16))
        self.label_33.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_15 = QLabel(self.page_4)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(30, 100, 220, 16))
        self.label_15.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_23 = QLabel(self.page_4)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(30, 200, 191, 16))
        self.label_23.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_27 = QLabel(self.page_4)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(30, 260, 231, 16))
        self.label_27.setStyleSheet(u"font: 11pt \"Avenir\";")
        self.label_16 = QLabel(self.page_4)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(30, 140, 191, 16))
        self.label_16.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.frame_2 = QFrame(self.page_4)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setGeometry(QRect(10, 30, 441, 551))
        self.frame_2.setStyleSheet(u"background-color: rgb(76, 42, 133); border-radius: 10px")
        self.label_21 = QLabel(self.frame_2)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(10, 10, 421, 311))
        self.label_21.setPixmap(QPixmap(u"images/stars.png"))
        self.label_21.setScaledContents(True)
        self.label_29 = QLabel(self.frame_2)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setGeometry(QRect(10, 190, 421, 351))
        self.label_29.setPixmap(QPixmap(u"images/stars.png"))
        self.label_29.setScaledContents(True)
        self.label_29.raise_()
        self.label_21.raise_()
        self.label_43 = QLabel(self.page_4)
        self.label_43.setObjectName(u"label_43")
        self.label_43.setGeometry(QRect(250, 160, 191, 16))
        self.label_43.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_38 = QLabel(self.page_4)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setGeometry(QRect(30, 440, 241, 16))
        self.label_38.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.model_3 = QPushButton(self.page_4)
        self.model_3.setObjectName(u"model_3")
        self.model_3.setGeometry(QRect(280, 530, 100, 32))
        self.model_3.setStyleSheet(u"color: rgb(107, 127, 215);")
        self.model_3.clicked.connect(self.export_csv)
        self.label_35 = QLabel(self.page_4)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setGeometry(QRect(250, 100, 191, 16))
        self.label_35.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_17 = QLabel(self.page_4)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(30, 120, 191, 16))
        self.label_17.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_40 = QLabel(self.page_4)
        self.label_40.setObjectName(u"label_40")
        self.label_40.setGeometry(QRect(250, 140, 211, 16))
        self.label_40.setStyleSheet(u"font: 11pt \"Avenir\";")
        self.label_31 = QLabel(self.page_4)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setGeometry(QRect(30, 380, 221, 16))
        self.label_31.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_34 = QLabel(self.page_4)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setGeometry(QRect(250, 120, 191, 16))
        self.label_34.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_18 = QLabel(self.page_4)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(60, 45, 341, 31))
        self.label_18.setStyleSheet(u"font: 24pt \"Bodoni 72\";\n"
"color: rgb(188, 247, 236)")
        self.label_28 = QLabel(self.page_4)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setGeometry(QRect(30, 280, 211, 16))
        self.label_28.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_44 = QLabel(self.page_4)
        self.label_44.setObjectName(u"label_44")
        self.label_44.setGeometry(QRect(250, 180, 191, 16))
        self.label_44.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_14 = QLabel(self.page_4)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(262, 350, 160, 16))
        self.label_14.setStyleSheet(u"font: 11pt; color: rgb(252, 255, 76);")
        self.label_39 = QLabel(self.page_4)
        self.label_39.setObjectName(u"label_39")
        self.label_39.setGeometry(QRect(30, 300, 191, 16))
        self.label_39.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.model_2 = QPushButton(self.page_4)
        self.model_2.setObjectName(u"model_2")
        self.model_2.setGeometry(QRect(90, 530, 100, 32))
        self.model_2.setStyleSheet(u"color: rgb(107, 127, 215);")
        self.model_2.clicked.connect(self.clear_and_reset)
        self.label_20 = QLabel(self.page_4)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(30, 160, 191, 16))
        self.label_20.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_13 = QLabel(self.page_4)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(270, 210, 141, 141))
        self.label_13.setPixmap(QPixmap("/images/star_examples/brown-dwarf.png"))
        self.label_13.setScaledContents(True)
        self.label_42 = QLabel(self.page_4)
        self.label_42.setObjectName(u"label_42")
        self.label_42.setGeometry(QRect(30, 500, 191, 16))
        self.label_42.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_41 = QLabel(self.page_4)
        self.label_41.setObjectName(u"label_41")
        self.label_41.setGeometry(QRect(30, 460, 271, 16))
        self.label_41.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_32 = QLabel(self.page_4)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setGeometry(QRect(30, 340, 191, 16))
        self.label_32.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_30 = QLabel(self.page_4)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setGeometry(QRect(30, 400, 231, 16))
        self.label_30.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_37 = QLabel(self.page_4)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setGeometry(QRect(30, 480, 231, 16))
        self.label_37.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_36 = QLabel(self.page_4)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setGeometry(QRect(30, 420, 191, 16))
        self.label_36.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.label_22 = QLabel(self.page_4)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(30, 80, 300, 16))
        self.label_22.setStyleSheet(u"font: 16pt \"Andale Mono\";\n"
"color: rgb(107, 127, 215);")
        self.label_45 = QLabel(self.page_4)
        self.label_45.setObjectName(u"label_45")
        self.label_45.setGeometry(QRect(30, 180, 191, 16))
        self.label_45.setStyleSheet(u"font: 13pt \"Avenir\";")
        self.stackedWidget.addWidget(self.page_4)
        self.frame_2.raise_()
        self.label_25.raise_()
        self.label_26.raise_()
        self.label_24.raise_()
        self.label_33.raise_()
        self.label_15.raise_()
        self.label_23.raise_()
        self.label_27.raise_()
        self.label_16.raise_()
        self.label_43.raise_()
        self.label_38.raise_()
        self.model_3.raise_()
        self.label_35.raise_()
        self.label_17.raise_()
        self.label_40.raise_()
        self.label_31.raise_()
        self.label_34.raise_()
        self.label_18.raise_()
        self.label_28.raise_()
        self.label_44.raise_()
        self.label_14.raise_()
        self.label_39.raise_()
        self.model_2.raise_()
        self.label_20.raise_()
        self.label_42.raise_()
        self.label_41.raise_()
        self.label_32.raise_()
        self.label_30.raise_()
        self.label_37.raise_()
        self.label_36.raise_()
        self.label_22.raise_()
        self.label_45.raise_()
        self.label_13.raise_()

        self.retranslateUi(Fusion)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Fusion)
    # setupUi

    def retranslateUi(self, Fusion):
        Fusion.setWindowTitle(QCoreApplication.translate("Fusion", u"Fusion", None))
        self.pushButton_3.setText("")
        self.label_3.setText(QCoreApplication.translate("Fusion", u"Effective Temperature (Kelvin):", None))
        self.label_4.setText(QCoreApplication.translate("Fusion", u"Luminosity (solar luminosities):", None))
        self.label_5.setText(QCoreApplication.translate("Fusion", u"Radius (solar radii):", None))
        self.label.setText(QCoreApplication.translate("Fusion", u"FUSION: Stellar Parameter Modeling", None))
        self.label_10.setText(QCoreApplication.translate("Fusion", u"Simulation Name:", None))
        self.pushButton_2.setText("")
        self.model.setText(QCoreApplication.translate("Fusion", u"Model Star(s)", None))
        self.pushButton_4.setText("")
        self.label_8.setText("")
        self.label_9.setText("")
        self.label_2.setText(QCoreApplication.translate("Fusion", u"Enter Star's Input Parameters:", None))
        self.label_6.setText("")
        self.label_7.setText("")
        self.label_25.setText(QCoreApplication.translate("Fusion", u"Great Circle Area (GCA/GCA\u2299): ", None))
        self.label_26.setText(QCoreApplication.translate("Fusion", u"Absolute Bolometric Luminosity (Lbol)(log(W)): ", None))
        self.label_24.setText(QCoreApplication.translate("Fusion", u"Great Circle Circumference (GCC/GCC\u2299): ", None))
        self.label_33.setText(QCoreApplication.translate("Fusion", u"Average Density (D/D\u2299): ", None))
        self.label_15.setText(QCoreApplication.translate("Fusion", u"Effective Temperature (K): ", None))
        self.label_23.setText(QCoreApplication.translate("Fusion", u"Surface Area (SA/SA\u2299): ", None))
        self.label_27.setText(QCoreApplication.translate("Fusion", u"Absolute Bolometric Magnitude (Mbol): ", None))
        self.label_16.setText(QCoreApplication.translate("Fusion", u"Radius (R/R\u2299): ", None))
        self.label_21.setText("")
        self.label_29.setText("")
        self.label_43.setText(QCoreApplication.translate("Fusion", u"Stellar Classification: ", None))
        self.label_38.setText(QCoreApplication.translate("Fusion", u"Surface Gravity (log(g)...log(N/kg)): ", None))
        self.model_3.setText(QCoreApplication.translate("Fusion", u"Export", None))
        self.label_35.setText(QCoreApplication.translate("Fusion", u"Spectral Class: ", None))
        self.label_17.setText(QCoreApplication.translate("Fusion", u"Luminosity (L/L\u2299): ", None))
        self.label_40.setText(QCoreApplication.translate("Fusion", u"Star Peak Wavelength (nm): ", None))
        self.label_31.setText(QCoreApplication.translate("Fusion", u"Central Pressure (log(N/m^2)): ", None))
        self.label_34.setText(QCoreApplication.translate("Fusion", u"Luminosity class: ", None))
        self.label_18.setText(QCoreApplication.translate("Fusion", u"FUSION: Stellar Parameter Modeling", None))
        self.label_28.setText(QCoreApplication.translate("Fusion", u"Absolute Magnitude (M)(Mv): ", None))
        self.label_44.setText(QCoreApplication.translate("Fusion", u"Star Type: ", None))
        self.label_14.setText(QCoreApplication.translate("Fusion", u"Example Star", None))
        self.label_39.setText(QCoreApplication.translate("Fusion", u"Bolometric Correction: ", None))
        self.model_2.setText(QCoreApplication.translate("Fusion", u"Home", None))
        self.label_20.setText(QCoreApplication.translate("Fusion", u"Diameter (D/D\u2299): ", None))
        self.label_42.setText(QCoreApplication.translate("Fusion", u"Metallicity (MH/MH\u2299): ", None))
        self.label_41.setText(QCoreApplication.translate("Fusion", u"Gravitational Binding Energy (log(J)): ", None))
        self.label_32.setText(QCoreApplication.translate("Fusion", u"Mass (M/M\u2299): ", None))
        self.label_30.setText(QCoreApplication.translate("Fusion", u"Centeral Temperature (log(K)): ", None))
        self.label_37.setText(QCoreApplication.translate("Fusion", u"Bolometric Flux (log(W/m^2)): ", None))
        self.label_36.setText(QCoreApplication.translate("Fusion", u"Lifespan (SL/SL\u2299): ", None))
        self.label_22.setText(QCoreApplication.translate("Fusion", u"Modeled Star", None))
        self.label_45.setText(QCoreApplication.translate("Fusion", u"Volume (V/V\u2299): ", None))
        self.label_13.setText("")
    # retranslateUi

