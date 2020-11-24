import math
import cv2
import numpy as np
from det_pato import *

# Define range of color in HSV
lower_white = np.array([0, 0, 155])
upper_white = np.array([255, 55, 255])
lower_yellow = np.array([20, 110, 170])
upper_yellow = np.array([40, 255, 255])

# Morfologias
kernel_dimensions = 4    # 5
erode_iterations = 1    # 1
dilate_iterations = 1    # 1

# Dibujo
yellow_figure_color = (255, 0, 255)
yellow_figure_thickness = 2
white_figure_color = (255, 0, 255)
white_figure_thickness = 2

# Dibujo centros de blobs detectados
show_centers_yellow = False
show_centers_white = False
yellow_centers_color = (0, 0, 255)
white_centers_color = (0, 0, 255)
centers_radius = 5
centers_thickness = 5

# Deteccion
minimum_deepness = 125
minimum_ratio_yellow = 1.5
minimum_ratio_white = 3.0    # 1.71

image = obs, reward, done, info = env.step(action)
class LineDetector():

    def __init__(self):

        # Subscribirce al topico "/duckiebot/camera_node/image/rect"
        self.image_subscriber = image

        # Publicar imagen con lineas al topico "duckiebot/camera_node/image/rect/streets"
        #self.image_publisher = rospy.Publisher("duckiebot/camera_node/image/rect/streets", Image, queue_size=1)

        # Publicar datos de lineas al topico "/duckiebot/street_data"
        #self.data_publisher = rospy.Publisher("/duckiebot/street_data", dict, queue_size=1)

        # Clase necesaria para transformar el tipo de imagen
        #self.bridge = CvBridge()

        # Ultima imagen adquirida
        #self.cv_image = Image()

        # Chiste
        print("encontrando lineas de calzada en 3, 2, 1...")

    #Ver imagen del simulador a traves de la clase
    def get_image(self):
        print(self.image_subscriber)

line = LineDetector()
line.get_image()




env.close()