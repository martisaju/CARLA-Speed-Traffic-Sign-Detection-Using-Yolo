#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time

import os
import subprocess
import numpy as np
import threading
import multiprocessing
from datetime import datetime



try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from PIL import Image
from pexpect import popen_spawn
import io


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180
detected_speed = ''
flg_warning = 0
current_speed = ''
frame = 0

def darknet(message):
    os.chdir("C:/Users/NAME/Desktop/darknet-master/build/darknet/x64")
    process = popen_spawn.PopenSpawn('darknet.exe detector test data/obj.data \
                                    yolov3-tiny-obj.cfg yolov3-tiny-obj_X.weights \
                                    -dont_show -ext_output -save_labels') #Running Darknet
    print(message)
    return process
message = 'Darknet Started'
darknet_process = darknet(message)



def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=10,
        NumberOfPedestrians=5,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(0.30, 0, 1.30)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args): 
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self._m = ''

    def execute(self, args):
        """Launch the PyGame."""
        global frame
        pygame.init()
        self._initialize_game() 
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                frame = frame +1
                self._on_loop(args)
                self._on_render(args)
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self, args):
        self._timer.tick()
        measurements, sensor_data = self.client.read_data()
 

        self._main_image = sensor_data.get('CameraRGB', None) 
        
        # Print measurements every second.        
        if self._timer.elapsed_seconds_since_lap() > 1.0: 

            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)                        
            
            else:
                self._print_player_measurements(measurements.player_measurements, args)

            # Plot position on the map as well.

            self._timer.lap()
        
        control = self._get_keyboard_control(pygame.key.get_pressed(), args)
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents
        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)
        
    def _get_keyboard_control(self, keys, args):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        global flg_warning

        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot

        if args.app == 'Control':
            if flg_warning == -1:
                control.throttle = 0.0

        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements, args):
        global detected_speed
        global current_speed
        global flg_warning
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        current_speed = '{speed:.2f}' 
        current_speed = current_speed.format(speed = player_measurements.forward_speed * 3.6)
        current_speed_f = float(current_speed)

        #####################################################################
        
        if current_speed_f != 0.00:
            if detected_speed != '':
                detected_speed_f = float(detected_speed)
                if args.app == 'Control':
                    detected_speed_plus_margin = detected_speed_f+detected_speed_f*(0.1) #ALLOW +10% MAX DETECTED SPEED 
                    if (detected_speed_plus_margin-current_speed_f)< 0.0:
                        flg_warning = -1
                    else:
                        flg_warning = 1
                elif args.app == 'Warning':  
                    if (detected_speed_f-current_speed_f)< 0.0:
                        flg_warning = -1
                    else:
                        flg_warning = 1

    def _traffic_sign_recogniser(self, new, detected_speed_loop):
        global darknet_process
        carla_scene_exists = os.path.isfile("Y:/Temp/carla_scene.jpg")
        if carla_scene_exists:
            carla_scene = (b"Y:/Temp/carla_scene.jpg")
            darknet_process.send(carla_scene+b'\n')

            labels_file_exists = os.path.isfile("Y:/Temp/carla_scene.txt")
            if labels_file_exists:
                labels_file = open("Y:/Temp/carla_scene.txt","r")
                predicted_labels = labels_file.read()
                if os.stat("Y:/Temp/carla_scene.txt").st_size != 0: 
                    #Check if any speed traffic sign has been detected
                    sign_tsr = predicted_labels[0]
                    if int(sign_tsr) == 0:
                        detected_speed_loop = '30'
                    elif int(sign_tsr) == 1:
                        detected_speed_loop = '60'
                    elif int(sign_tsr) == 2:
                        detected_speed_loop = '90'
                    new = True

                else:
                    detected_speed_loop = ''
        return detected_speed_loop, new

    def _on_render(self, args):
        global detected_speed
        global flg_warning
        global current_speed
        global frame

        self._timer.tick()
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            if args.app:               
                new = False
                detected_speed_loop = ''
               
                if (frame % 5 ==0):  
                    a = Image.fromarray(array, 'RGB')
                    a.save('Y:/Temp/carla_scene.jpg')     
                    detected_speed_loop, new = self._traffic_sign_recogniser(new, detected_speed_loop)
   
                if new:
                    detected_speed = detected_speed_loop
                
                if flg_warning == -1:
                    r = 255
                    g = 0
                    b = 0
                elif flg_warning == 1:
                    r = 0
                    g = 255
                    b = 0
                else:
                    r = 0
                    g = 0
                    b = 255
                
                basicfont = pygame.font.SysFont(None, 80)
                text_detected = basicfont.render(detected_speed, True, (0,0,255))
                textrec = text_detected.get_rect()
                textrec.top = surface.get_rect().top
                textrec.midtop = surface.get_rect().midtop
                surface.blit(text_detected, textrec)
                
                text_current = basicfont.render(current_speed+'km/h', True, (r,g,b))
                textrec = text_current.get_rect()
                textrec.top = surface.get_rect().top
                textrec.bottomright = surface.get_rect().bottomright
                surface.blit(text_current, textrec)
                
                if args.app == 'Warning':
                    if flg_warning==-1:
                        basicfont = pygame.font.SysFont(None, 60)
                        text_warning = basicfont.render('REDUCE YOUR SPEED', True, (r,g,b))
                        textrec = text_warning.get_rect()
                        textrec.top = surface.get_rect().top
                        textrec.bottomleft = surface.get_rect().bottomleft
                        surface.blit(text_warning, textrec)
               
                if args.app == 'Control':
                    if flg_warning==-1:
                        basicfont = pygame.font.SysFont(None, 60)
                        text_control = basicfont.render('SPEED REDUCED', True, (r,g,b))
                        textrec = text_control.get_rect()
                        textrec.top = surface.get_rect().top
                        textrec.bottomleft = surface.get_rect().bottomleft
                        surface.blit(text_control, textrec)

            self._display.blit(surface, (0, 0))
        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-app', '--app',
        choices = ['Warning', 'Control'],
        help='Warns in case that the current speed of the agent is higher than the detected one.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    
    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute(args)
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
