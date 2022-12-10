# visualization of pantograph
# reference (serial communication):  https://m.blog.naver.com/PostView.nhn?blogId=chandong83&logNo=220941128858&proxyReferer=https:%2F%2Fwww.google.com%2F


# -----------------------------------------------------------------------------------------------
# Predefined User Parameters
debug = False
# -----------------------------------------------------------------------------------------------
# Library
from numpy.core.multiarray import zeros
from numpy.lib.function_base import average
from numpy.lib.type_check import real
from pynput import keyboard

import serial
import cv2 as cv

import time
import signal
import threading
import numpy as np
from math import *
import random

# -----------------------------------------------------------------------------------------------
# System variables
line = []

port = 'COM5'  # serial port number
baud = 9600

exitThread = False
pantograph = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # pantograph: 9 floats [q1,q2,tau1,tau2,x,y,fx,fy,dt]

horizon_height = 30
img_width = 1500
img_height = 750
workspace = 0.45
radius = 0.013
cool_time = 0
cool_time_p = 0


# -----------------------------------------------------------------------------------------------
# User-defined Functions
def handler(signum, frame):
    exitThread = True


def parsing_data(data_list):
    global pantograph

    data_str = "".join(data_list)

    data = data_str.split(',')
    if len(data) == 8:
        pantograph = [float(i) for i in data]



def readThread(ser):
    global exitThread

    while not exitThread:
        for c in ser.read():
            line.append(chr(c))

            if c == 10:
                parsing_data(line)
                del line[:]


def draw_obstacle(img):
    global img_height, img_width, workspace, horizon_height
    real_to_img_scale = img_width / workspace

    center = (0.0, 0.160)
    radius_s = 0.005

    def real2img(real_pos):
        img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                   floor(real_pos[1] * real_to_img_scale) + horizon_height)
        return img_pos

    # draw obstacle
    cv.circle(img, real2img(center), radius_s=floor(radius_s * real_to_img_scale), color=(255, 255, 0), thickness=-1)

    return img


def draw_pantograph(img):
    global pantograph, img_height, img_width, workspace, horizon_height, coordinates, past_coordinates, l_x, h_x, l_y, h_y, c_x, c_y, c_r
    d = 0.05
    l1 = 0.130
    l2 = 0.130

    q1 = pantograph[0]
    q2 = pantograph[1]
    x = pantograph[4]
    y = pantograph[5]
    fx = pantograph[6]
    fy = pantograph[7]
    # dt = pantograph[8]

    real_to_img_scale = img_width / workspace
    force_scale = 0.1  # 1N --> 100mm

    def real2img(real_pos):
        img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                   floor(real_pos[1] * real_to_img_scale) + horizon_height)
        return img_pos

    def img2real(img_pos):
        real_pos = (-(real_to_img_scale * workspace - 2 * img_pos[0]) / (2 * real_to_img_scale),
                    -(horizon_height - img_pos[1]) / real_to_img_scale)
        return real_pos

    if debug:
        coordinates = list(coordinates)
        if (l_x < coordinates[0] < h_x):
            coordinates[0] = coordinates[0]
        else:
            coordinates[0] = past_coordinates[0]

        if (l_y < coordinates[1] < h_y):
            coordinates[1] = coordinates[1]
        else:
            coordinates[1] = past_coordinates[1]

        target = img2real(coordinates)

        l13 = sqrt((target[1]) ** 2 + (target[0] - (-d / 2)) ** 2)
        alpha_1 = acos((l13 ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1 * l13))
        beta_1 = atan2(abs(target[1]), (target[0] - (-d / 2)))
        q1 = (alpha_1 + beta_1) * 180 / pi

        l15 = sqrt((target[1]) ** 2 + (target[0] - (d / 2)) ** 2)
        alpha_5 = pi - atan2(target[1], target[0] - (d / 2))
        beta_5 = acos((l15 ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1 * l15))
        q2 = (pi - alpha_5 - beta_5) * 180 / pi
        past_coordinates = coordinates
    else:
        target = (x, y)

    motor1 = (-d / 2, 0)
    motor2 = (d / 2, 0)

    elbow1 = (-d / 2 + l1 * cos(q1 / 180 * pi), l1 * sin(q1 / 180 * pi))
    elbow2 = (d / 2 + l1 * cos(q2 / 180 * pi), l1 * sin(q2 / 180 * pi))

    force_end = (target[0] + fx * force_scale, target[1] + fy * force_scale)

    # draw base
    cv.rectangle(img, real2img((-workspace * 1 / 3, -0.05)), real2img((workspace * 1 / 3, 0)), color=(200, 100, 100),
                 thickness=-1)

    # draw links
    cv.line(img, real2img(motor1), real2img(elbow1), color=(100, 100, 100), thickness=3)
    cv.line(img, real2img(motor2), real2img(elbow2), color=(100, 100, 100), thickness=3)
    cv.line(img, real2img(elbow1), real2img(target), color=(100, 100, 100), thickness=3)
    cv.line(img, real2img(elbow2), real2img(target), color=(100, 100, 100), thickness=3)

    # draw joints
    cv.circle(img, real2img(motor1), radius=30, color=(0, 0, 0), thickness=-1)
    cv.circle(img, real2img(motor2), radius=30, color=(0, 0, 0), thickness=-1)
    cv.circle(img, real2img(elbow1), radius=15, color=(0, 0, 0), thickness=-1)
    cv.circle(img, real2img(elbow2), radius=15, color=(0, 0, 0), thickness=-1)

    # draw effective force
    # TODO: need to analyze closed chain
    cv.arrowedLine(img, real2img(target), real2img(force_end), color=(0, 0, 255), thickness=3, tipLength=0.3)

    # draw end-effector
    cv.circle(img, real2img(target), radius=15, color=(255, 0, 255), thickness=-1)

    # draw workspace
    cv.circle(img, real2img((c_x, c_y)), int(c_r * real_to_img_scale), (0, 0, 255), 3)

    # write cycle timeout
    # cv.putText(img, "cycle time: {}ms".format(dt), (img_width-200, img_height-30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0))
    return img


def mouse_callback(event, x, y, flags, param):
    global coordinates, click_coordinates, click_state

    if event == cv.EVENT_LBUTTONDOWN:
        click_coordinates = (x, y)
        click_state = True

    coordinates = (x, y)


def on_press(key):
    global coordinates, click_state, pantograph, click_coordinates, shoot_count, click_key, cool_time, cool_time_p
    cool_time = time.time()
    if key == keyboard.Key.space and cool_time - cool_time_p > 1.0:
        cool_time_p = cool_time
        click_coordinates = (pantograph[4], pantograph[5])
        click_state = True
        shoot_count = shoot_count + 1;
        # print('Trigger');
        click_key = key;


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def draw_crosshair(img):
    global coordinates, pantograph, img_height, img_width, workspace, horizon_height, click_state
    x = pantograph[4]
    y = pantograph[5]

    def real2img(real_pos):
        real_to_img_scale = img_width / workspace
        img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                   floor(real_pos[1] * real_to_img_scale) + horizon_height)
        return img_pos

    (i_x, i_y) = real2img((x, y))

    if click_state:
        img = cv.line(img, (i_x - 1, 0), (i_x - 1, img_height), (0, 0, 255), 5)
        img = cv.line(img, (0, i_y - 1), (img_width, i_y - 1), (0, 0, 255), 5)
    else:
        img = cv.line(img, (i_x - 1, 0), (i_x - 1, img_height), (0, 255, 0), 2)
        img = cv.line(img, (0, i_y - 1), (img_width, i_y - 1), (0, 255, 0), 2)

    return img


def kill_target(img, target_pos):
    global img_height, img_width, workspace, horizon_height, click_state, click_coordinates, hit, radius, radius_px, packet
    real_to_img_scale = img_width / workspace

    def real2img(real_pos):
        real_to_img_scale = img_width / workspace
        img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                   floor(real_pos[1] * real_to_img_scale) + horizon_height)
        return img_pos

    target_pos = packet

    cv.circle(img, real2img(target_pos), floor(radius * real_to_img_scale), color=(255, 0, 0), thickness=-1)
    cv.circle(img, real2img(target_pos), floor(radius * real_to_img_scale * 0.3), color=(0, 0, 255), thickness=-1)
    if click_state == True:
        target_coordinates = target_pos
        if (click_coordinates[0] - target_coordinates[0]) ** 2 + (
                click_coordinates[1] - target_coordinates[1]) ** 2 < radius ** 2:
            cv.circle(img, real2img(target_pos), floor(radius * real_to_img_scale), color=(0, 255, 255), thickness=-1)
            cv.circle(img, real2img(target_pos), floor(radius * real_to_img_scale * 0.3), color=(255, 255, 0),
                      thickness=-1)
            hit = True
            print("hit")
        click_state = False

    return img


def transferI2R(pos):
    global img_height, img_width, workspace, horizon_height
    real_to_img_scale = img_width / workspace

    def img2real(img_pos):
        real_pos = [-(real_to_img_scale * workspace - 2 * img_pos[0]) / (2 * real_to_img_scale),
                    -(horizon_height - img_pos[1]) / real_to_img_scale]
        return real_pos

    return img2real(pos)


def writeThread(ser):
    global exitThread, packet, img_height, img_width, workspace, horizon_height, new_vx, new_vy

    real_to_img_scale = img_width / workspace

    def real2img(real_pos):
        img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                   floor(real_pos[1] * real_to_img_scale) + horizon_height)
        return img_pos

    while not exitThread:
        packet_i = real2img(packet)
        packet_i_x_L = packet_i[0] >> 0 & 0b11111111
        packet_i_x_H = packet_i[0] >> 8 & 0b11111111
        packet_i_y_L = packet_i[1] >> 0 & 0b11111111
        packet_i_y_H = packet_i[1] >> 8 & 0b11111111
        vacket_i = real2img((new_vx, new_vy))
        vacket_i_x_L = vacket_i[0] >> 0 & 0b11111111
        vacket_i_x_H = vacket_i[0] >> 8 & 0b11111111
        vacket_i_y_L = vacket_i[1] >> 0 & 0b11111111
        vacket_i_y_H = vacket_i[1] >> 8 & 0b11111111
        ser.write((packet_i_x_L, packet_i_x_H, packet_i_y_L, packet_i_y_H, vacket_i_x_L, vacket_i_x_H, vacket_i_y_L,
                   vacket_i_y_H))


def randomThread():
    global exitThread, packet, v_angle, v_magnitude, kill_pos, kill_count, c_y, c_r, radius, inc_dir, inc_mag_a, inc_mag_b, inc_mag_o, img_height, img_width, workspace, horizon_height, new_vx, new_vy

    minR = -(c_r - radius);
    maxR = c_r - radius;

    randomize()

    inc = 0.0001
    a = 0
    b = 0
    offset = c_r * 0.1;
    offset_o = inc_mag_o;

    o = offset_o;

    b = offset;

    direction_a = 1;
    direction_b = 1;

    old_kill_count = kill_count
    while cv.getWindowProperty("pantograph", 0) >= 0:
        new_x = a * cos(o);
        new_y = b * sin(o) + c_y;
        new_vx = - a * sin(o)
        new_vy = b * cos(o)

        if a > maxR:
            direction_a = -1;
        elif a < minR:
            direction_a = +1;

        a = a + inc * inc_mag_a * direction_a

        if b > maxR:
            direction_b = -1;
        elif b < minR:
            direction_b = +1;

        b = b + inc * inc_mag_b * direction_b

        o = o + inc * inc_dir * 10

        packet = (new_x, new_y)
        real_to_img_scale = img_width / workspace

        def real2img(real_pos):
            img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                       floor(real_pos[1] * real_to_img_scale) + horizon_height)
            return img_pos

        packet_i = real2img(packet)
        packet_i_x_L = packet_i[0] >> 0 & 0b11111111
        packet_i_x_H = packet_i[0] >> 8 & 0b11111111
        packet_i_y_L = packet_i[1] >> 0 & 0b11111111
        packet_i_y_H = packet_i[1] >> 8 & 0b11111111

        def vel2img(real_pos):
            img_pos = (floor((real_pos[0] + workspace / 2) * real_to_img_scale),
                       floor(real_pos[1] * real_to_img_scale) + horizon_height * 10)
            return img_pos

        vacket_i = vel2img((new_vx, new_vy))
        vacket_i_x_L = vacket_i[0] >> 0 & 0b11111111
        vacket_i_x_H = vacket_i[0] >> 8 & 0b11111111
        vacket_i_y_L = vacket_i[1] >> 0 & 0b11111111
        vacket_i_y_H = vacket_i[1] >> 8 & 0b11111111
        ser.write((packet_i_x_L, packet_i_x_H, packet_i_y_L, packet_i_y_H, vacket_i_x_L, vacket_i_x_H, vacket_i_y_L,
                   vacket_i_y_H))
        # ser.write((packet_i_x_L, packet_i_x_H, packet_i_y_L, packet_i_y_H))

        if o > 2 * pi:
            o = o - 2 * pi

        if kill_count != old_kill_count:
            randomize()

        if kill_count == kill_total:
            break

        old_kill_count = kill_count


def randomize():
    global inc_dir, inc_mag_a, inc_mag_b, inc_mag_o

    if random.random() > 0.5:
        inc_dir = +1
    else:
        inc_dir = -1

    inc_mag_a = random.uniform(1.6, 2);
    inc_mag_b = random.uniform(1.6, 2);
    inc_mag_o = random.uniform(pi, 1.2 * pi);
    # print("!", inc_dir,',',inc_mag_a,',',inc_mag_b)


# -----------------------------------------------------------------------------------------------
# Main function
if __name__ == "__main__":
    # Global variable initialization
    global coordinates, click_coordinates, click_state, hit, kill_total, kill_count, kill_pos, kill_time, packet, l_x, l_y, h_x, h_y, c_x, c_y, c_r, shoot_count, click_key, new_vx, new_vy

    shoot_count = 0;
    coordinates = (0, 0)
    click_coordinates = (0, 0)
    click_state = False
    packet = [0] * 2;
    new_vx = 0;
    new_vy = 0;

    # image workspace limit
    c_x = 0;
    c_y = 0.12;
    c_r = 0.08

    shoot_count = 0

    l_x = int(img_width * 0.2)
    h_x = int(img_width * (1 - 0.2))
    l_y = int(img_height * 0.3)
    h_y = int(img_height * (1 - 0.2))
    # print(l_x)
    # print(h_x)
    # print(l_y)
    # print(h_y)

    kill_pos = [[0, 0]] * 20
    kill_total = len(kill_pos)
    kill_time = zeros(kill_total)
    kill_count = 0
    kill_hitpos = np.full((kill_total, 2), (0.0, 0.0))
    kill_accuracy = zeros(kill_total)
    kill_deviation = zeros(kill_total, dtype=float)
    hit = False
    click_key = keyboard.Key.enter

    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    # Serial communication establishment
    if not (debug):
        signal.signal(signal.SIGINT, handler)

        ser = serial.Serial(port, baud, timeout=3)

        thread = threading.Thread(target=readThread, args=(ser,), daemon=True)
        thread.start()
        # thread2 = threading.Thread(target=writeThread, args=(ser,), daemon=True)
        # thread2.start();

    thread3 = threading.Thread(target=randomThread, daemon=True)

    # -----------------------------------------------------------------------------------------------
    # Initialization Sequence
    name = input("Insert your name: ");

    print("Release device from singularity")
    print("Press space to start")
    while click_key != keyboard.Key.space:
        shoot_count = 0

    time.sleep(1)
    # Image frame initilization
    cv.namedWindow('pantograph')
    sim = np.zeros((img_height, img_width, 3), np.uint8)
    # -----------------------------------------------------------------------------------------------
    # Game Sequence

    shoot_count = 0
    start = time.time()
    thread3.start();

    while cv.getWindowProperty("pantograph", 0) >= 0:

        # Painting Sequence
        sim = 255 * np.ones((img_height, img_width, 3), np.uint8)  # Color background as white
        #        sim = draw_obstacle(sim)                                    # Overlay obstacle image
        sim = draw_pantograph(sim)  # Overlay pantograph image
        sim = draw_crosshair(sim)  # Overlay crosshair image
        sim = kill_target(sim, kill_pos[kill_count])  # Overlay designated kill targets
        cv.imshow("pantograph", sim)  # Show image
        # print(hit)

        # Data Acquisition Sequence
        if debug:
            cv.setMouseCallback('pantograph', mouse_callback)

        if hit:
            kill_time[kill_count] = time.time()

            # print(click_coordinates)
            kill_hitpos[kill_count] = click_coordinates
            kill_pos[kill_count] = [round(packet[0], 3), round(packet[1], 3)]

            kill_deviation[kill_count] = sqrt((kill_hitpos[kill_count][0] - kill_pos[kill_count][0]) ** 2 + (
                    kill_hitpos[kill_count][1] - kill_pos[kill_count][1]) ** 2)
            kill_accuracy[kill_count] = -1.0 / radius * kill_deviation[kill_count] + 1

            kill_count = kill_count + 1
            hit = False

            if kill_count >= kill_total:
                for i in range(kill_total - 1, -1, -1):
                    if i == 0:
                        kill_time[0] = kill_time[0] - start

                    else:
                        kill_time[i] = kill_time[i] - kill_time[i - 1]
                break

        # Break Sequence
        keyCode = cv.waitKey(33) & 0xFF  # Wait for break trigger
        if keyCode == 27:
            # print("ESC")
            break
    # -----------------------------------------------------------------------------------------------
    # Termination Sequence

    strFormat = '%s.txt'
    strOut = strFormat % (name)
    file = open(strOut, 'w')

    strFormat = '\t\t%-15s%-15s%-15s%-25s%-10s\n'
    strOut = strFormat % ('Time [sec]', 'Accuracy [%]', 'Deviation [m]', 'Target Pos. [m]', 'Hit Pos. [m]')
    print(strOut)
    file.write(strOut)
    strFormat = '%s%d\t%-15s%-15s%-15s%-25s%-10s\n'
    for i in range(0, kill_total):
        strOut = strFormat % ('Target #', i, "{:.3f}".format(kill_time[i]), "{:.3f}".format(kill_accuracy[i] * 100.0),
                              "{:.3f}".format(kill_deviation[i]),
                              "[{:.3f}, {:.3f}]".format(kill_pos[i][0], kill_pos[i][1]), kill_hitpos[i])
        print(strOut)
        file.write(strOut)

    print("Total Time\t\t: {:.2f}".format(sum(kill_time)), "\t[sec]")
    print("Average Accuracy\t: {:.2f}".format(average(kill_accuracy) * 100.0), "[%]")
    print("Total Shots\t\t: {:.0f}".format(shoot_count))
    print("Hit to Miss Ratio\t: {:.2f}".format((kill_total / shoot_count) * 100.0), "[%]")

    strList = ("Total Time\t\t: {:.2f}".format(sum(kill_time)), "\t[sec]\n")
    strOut = ' '.join(s for s in strList)
    file.write(strOut)
    strList = ("Average Accuracy\t: {:.2f}".format(average(kill_accuracy) * 100.0), "[%]\n")
    strOut = ' '.join(s for s in strList)
    file.write(strOut)
    strList = ("Total Shots\t\t: {:.0f}".format(shoot_count), "\n")
    strOut = ' '.join(s for s in strList)
    file.write(strOut)
    strList = ("Hit to Miss Ratio\t: {:.2f}".format((kill_total / shoot_count) * 100.0), "[%]\n")
    strOut = ' '.join(s for s in strList)
    file.write(strOut)

    exitThread = True

    if not (debug):
        thread.join(1000)
        # thread2.join(1000)
        ser.close()

    thread3.join(1000)
    cv.destroyAllWindows()
    file.close()
    print("Terminating program")
    time_duration = 1.0
    time.sleep(time_duration)
    exit()
    # -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------