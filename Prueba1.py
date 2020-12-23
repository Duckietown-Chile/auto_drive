import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2


#nuevos import
import math
#from cv_bridge import CvBridge, CvBridgeError



    
def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }
#aca supongo que hay que implementar los campos potenciales.
    return actions.get(key, np.array([0.0, 0.0]))



# Define range of color in HSV
lower_white = np.array([0, 0, 155])
upper_white = np.array([255, 55, 255])
lower_yellow = np.array([20, 80, 150])
upper_yellow = np.array([45, 255, 255])

# Morfologias
kernel_dimensions = 4    # 5
erode_iterations = 3    # 1
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


def morfologies(mask):
    kernel = np.ones((kernel_dimensions, kernel_dimensions), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode_iterations)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    return mask

def center(rect):
    x = rect[0][0]
    y = rect[0][1]
    center = {'coordx':None,'coordy': None}
    center['coordx'] = x
    center['coordy'] = y
    return center

def ratio(rect):
    ratios = {'coordx':None,'coordy': None}
    w = float(rect[1][0])
    h = float(rect[1][1])
    ratios['coordx'] = h/w
    ratios['coordy'] = w/h
    return ratios


#como en det_pato
if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='loop_obstacles')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            seed = args.seed,
            map_name = args.map_name,
            draw_curve = args.draw_curve,
            draw_bbox = args.draw_bbox,
            domain_rand = args.domain_rand,
            frame_skip = args.frame_skip,
            distortion = args.distortion,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(30)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        action = mov_duckiebot(key)
        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)
        # obs consiste en un imagen RGB de 640 x 480 x 3

        # done significa que el Duckiebot chocó con un objeto o se salió del camino
        if done:
            print('done!')
            # En ese caso se reinicia el simulador
            env.reset()

        # Se deja en frame la imagen actual
        frame = obs 
        # Creamos los espacios de color
        gray_space = cv2.COLOR_BGR2GRAY
        color_space = cv2.COLOR_RGB2HSV

        # Encontramos las mascaras de colores blanco y amarillo
        frame_hsv = cv2.cvtColor(frame, color_space)   #aca color_space -> cv2.COLOR_RGB2HSV
        mask_yellow = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(frame_hsv, lower_white, upper_white)
    
        # Realizamos las operaciones morfologicas para borrar manchas pequenas
        mask_yellow = morfologies(mask_yellow) 
        mask_white = morfologies(mask_white)
    
        # Filtramos la imagen con esos colores
        frame_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        frame_white = cv2.bitwise_and(frame, frame, mask=mask_white)




        #parte nuevaaaaaa, para cambiar de RGB  a  BGR
        frame_yellow = cv2.cvtColor(frame_yellow, cv2.COLOR_RGB2BGR)
        frame_white = cv2.cvtColor(frame_white, cv2.COLOR_RGB2BGR)



        # Cambio a espacio de color en blanco y negro
        frame_yellow = cv2.cvtColor(frame_yellow, gray_space)
        frame_white = cv2.cvtColor(frame_white, gray_space)

        # Deteccion de contornos
        #image__yellow, sacamos esto pq genera problemas
        contours_yellow, hierarchy_yellow = cv2.findContours(frame_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #image_white, sacamos esto pq genera problemas
        contours_white, hierarchy_white = cv2.findContours(frame_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Creamos el diccionario con los datos
        data = {'yellow': 0, 'white': 0, 'yellow_data': [], 'white_data': []}
    
    
        # Manejo de datos amarillos
        for cnt_yellow in contours_yellow:

                # Encontrar rectangulos rotados y sus puntos centro
                rect_yellow = cv2.minAreaRect(cnt_yellow)   
                center_yellow = center(rect_yellow)

                # Dibujar puntos centro
                if show_centers_yellow:
                    center_yellow_coordinates = (int(center_yellow['coordx']), int(center_yellow['coordy']))
                    frame = cv2.circle(frame, center_yellow_coordinates, centers_radius, yellow_centers_color, centers_thickness)

                # Condiciones para ser calzada amarilla
                ratio_yellow = ratio(rect_yellow)
                condition_yellow = center_yellow['coordy'] >= minimum_deepness and (ratio_yellow['coordx'] >= minimum_ratio_yellow or ratio_yellow['coordy'] >= minimum_ratio_yellow)
#todos los .x tienen que ser ['coordy']
                if condition_yellow:
                    # Dibujo de rectangulos en la imagen
                    box__yellow = np.int0(cv2.boxPoints(rect_yellow))
                    frame = cv2.drawContours(frame, [box__yellow], 0, yellow_figure_color, yellow_figure_thickness)

                    # Extraccion de datos amarillos
                    data['yellow'] += 1
                    data['yellow_data'].append(rect_yellow)

        # Manejo de datos blancos
        for cnt_white in contours_white:

                # Encontrar rectangulos rotados y sus puntos centro
                rect_white = cv2.minAreaRect(cnt_white)
                center_white = center(rect_white)

                # Dibujar puntos centro
                if show_centers_white:
                    center_white_coordinates = (int(center_white['coordx']), int(center_white['coordy']))
                    frame = cv2.circle(frame, center_white_coordinates, centers_radius, white_centers_color, centers_thickness)

                # Condiciones para ser calzada blanca
                ratio_white = ratio(rect_white)
                condition_white = center_white['coordy'] >= minimum_deepness and (ratio_white['coordx'] >= minimum_ratio_white or ratio_white['coordy'] >= minimum_ratio_white)

                if condition_white:
                    # Dibujo de rectangulos en la imagen
                    box_white = np.int0(cv2.boxPoints(rect_white))
                    frame = cv2.drawContours(frame, [box_white], 0, white_figure_color, white_figure_thickness)
                
                    # Extraccion de datos blancos
                    data['white'] += 1
                    data['white_data'].append(rect_white)
         
            
        #Ventana con imagen normal del duckiebot           
        #cv2.imshow('Vista Normal', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        #Ventana con la deteccion
        cv2.imshow('Filtrado', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# Se cierra el environment y termina el programa
env.close() 
