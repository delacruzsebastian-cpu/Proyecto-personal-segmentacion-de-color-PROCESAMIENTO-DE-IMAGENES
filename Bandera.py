import numpy as np
import cv2
from hough import *
from orientation_estimate import *

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import os

#SEBASTIAN DE LA CRUZ GUTIERREZ PUJ
class Bandera:  #Crear clase
    def __init__(self, path , image_name):  # se define el constructor
        self.path = path
        self.image_name=image_name
        self.path_file = os.path.join(path, image_name)
        self.image = cv2.imread(self.path_file)  # se carga la imagen via opencv
        self.image_copy = cv2.imread(self.path_file)
        self.labels = 0

    def Colores(self):

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        n_colors = 4
        self.image = np.array(self.image, dtype=np.float64) / 255
        rows, cols, ch = self.image.shape
        assert ch == 3
        image_array = np.reshape(self.image, (rows * cols, ch))
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        self.labels = model.predict(image_array)

        if np.max(self.labels) == 0:
            print('la bandera tiene 1 colores')
        if np.max(self.labels) == 1:
            print('la bandera tiene 2 colores')
        if np.max(self.labels) == 2:
            print('la bandera tiene 3 colores')
        if np.max(self.labels) == 3:
            print('la bandera tiene 4 colores')

    def Porcentaje(self):

        unique, counts = np.unique(self.labels, return_counts=True)
        porc = (counts * 100) / 24480

        if np.max(self.labels) == 0:
            print('el porcentaje de color 1 es')
            print(porc[0])
        if np.max(self.labels) == 1:
            print('el porcentaje de color 1 es')
            print(porc[0])
            print('porciento')
            print('el porcentaje de color 2 es')
            print(porc[1])
        if np.max(self.labels) == 2:
            print('el porcentaje de color 1 es')
            print(porc[0])
            print('porciento')
            print('el porcentaje de color 2 es')
            print(porc[1])
            print('porciento')
            print('el porcentaje de color 3 es')
            print(porc[2])
            print('porciento')
        if np.max(self.labels) == 3:
            print('el porcentaje de color 1 es')
            print(porc[0])
            print('porciento')
            print('el porcentaje de color 2 es')
            print(porc[1])
            print('porciento')
            print('el porcentaje de color 3 es')
            print(porc[2])
            print('porciento')
            print('el porcentaje de color 4 es')
            print(porc[3])
            print('porciento')

    def Orientacion(self):

        self.image_copy = cv2.resize(self.image_copy, (600, 600))
        high_thresh = 300
        bw_edges = cv2.Canny(self.image_copy, high_thresh * 0.3, high_thresh, L2gradient=True)
        hough1 = hough(bw_edges)
        accumulator = hough1.standard_HT()
        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hough1.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = self.image.shape[:2]

        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough1.theta[peaks[i][1]]
            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough1.center_x
            y0 = b * rho + hough1.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

        if 85 < np.abs(theta_) < 95:
            print("bandera horizontal")
        if 175 < np.abs(theta_) < 185:
            print("bandera vertical")
        if 0 < np.abs(theta_) < -5:
            print("bandera mixta")

