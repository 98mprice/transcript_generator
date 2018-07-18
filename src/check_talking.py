import sys
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import math
import random
import os
from os import listdir
from os.path import isfile, join
import numpy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json
from PIL import Image
import time
import progressbar
from collections import Counter

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

fnt = ImageFont.truetype('../res/Roboto-BlackItalic.ttf', 30)

def generate_padded_image(raw_image):
    """
        takes in a raw image of an open mouth,
        (blue background, red mouth)
        ...and padds it to 128*128 resolution to maintain
        a standard size for calculations
    """
    image = Image.new('RGB', (128, 128), (0, 0, 255))

    wpercent = (128/float(raw_image.size[0]))
    hsize = int((float(raw_image.size[1])*float(wpercent)))
    if hsize > 0:
        resized_raw_image = raw_image.resize((128,hsize), Image.ANTIALIAS)

        image.paste(resized_raw_image, (0, 64 - resized_raw_image.height/2))
        return image
    else:
        return None

def flip(image):
    """
        flips given image horizontally
    """
    flipped_arr = numpy.flipud(image)
    return Image.fromarray(flipped_arr, 'RGB')

# unused TODO
def count_similar_pixels(a, b):
    """
        returns the number of equal pixels between image a and image b
    """
    similarity_count = 0
    for i in range(128):
        for j in range(128):
            r_a, g_a, b_a = a.getpixel((i,j))
            r_b, g_b, b_b = b.getpixel((i,j))
            if (r_a == r_b) and (b_a == b_b):
                similarity_count +=  1
    return similarity_count

# unused TODO
def similarity(a, b):
    """
        returns the ratio of equal pixels to total number of pixels in image a
        also flips the image to check if the flipped version has a higher ratio
    """
    similarity_count = count_similar_pixels(a, b)
    similarity = (float(similarity_count)/(128.0 * 128.0))
    if similarity < THRESHOLD:
        similarity_count = count_similar_pixels(flip(a), b)
        similarity_flipped = (float(similarity_count)/(128.0 * 128.0))
        if similarity_flipped > similarity:
            return similarity_flipped
    return similarity

def percentage_open(image):
    """
        returns the ratio of red pixels in the image to the total number of pixels
    """
    red_count = 0
    for i in range(128):
        for j in range(128):
            r, g, b = image.getpixel((i,j))
            if r > b:
                red_count +=  1
    return float(red_count)/(128.0 * 128.0)

# unused TODO
def PolyArea(x,y):
    """
        returns the area
    """
    return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))

def find_link_between(p1, p2, list):
    """
        finds the common link between the top and bottom lip
    """
    p1_index = list.index(p1)
    p2_index = list.index(p2)
    new_list = []
    if p2_index > p1_index:
        new_list = list[p1_index:p2_index+1]
    elif p1_index > p2_index:
        new_list = list[p2_index:p1_index+1]
    return new_list

# unused TODO
def find_area(p):
    """
        returns area of polygon
    """
    x = list(map(list, zip(*p)))
    area = PolyArea(x[0], x[1])
    return area

def find_lips(pil_image, face_landmarks):
    """
        finds relative area & draws image for a single face
    """
    # Print the location of each facial feature in this image
    if LOGGING_LEVEL == 'VERBOSE':
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    naked_image = pil_image.copy()

    # Let's trace out each facial feature in the image with a line!
    loc3 = list(set(face_landmarks['top_lip']).intersection(face_landmarks['bottom_lip']))
    d = ImageDraw.Draw(pil_image)

    # area of chin
    area_chin = find_area(face_landmarks['chin'])
    # draw area of chin
    d.polygon(face_landmarks['chin'], fill=(0, 0, 255, 128))

    # extract bottom of top lip & top of bottom lip
    x_values = sorted(loc3, key=lambda x: x[0])
    x_values.pop(0)
    x_values.pop(len(x_values) -1)
    top_link = find_link_between(x_values[0], x_values[1], face_landmarks['top_lip'])
    bottom_link = find_link_between(x_values[0], x_values[1], face_landmarks['bottom_lip'])
    # join these together
    inner_mouth = top_link + bottom_link

    # draw area of mouth
    d.polygon(inner_mouth, fill=(255, 0, 0, 128))

    return {
        'lip': pil_image.crop((min(inner_mouth, key = lambda t: t[0])[0], min(inner_mouth, key = lambda t: t[1])[1], max(inner_mouth, key = lambda t: t[0])[0], max(inner_mouth, key = lambda t: t[1])[1])),
        'naked_lip': naked_image.crop((min(inner_mouth, key = lambda t: t[0])[0], min(inner_mouth, key = lambda t: t[1])[1], max(inner_mouth, key = lambda t: t[0])[0], max(inner_mouth, key = lambda t: t[1])[1]))
    }

def find_faces(image):
    """
        finds faces in image and returns an array of their find_lips
    """
    pil_image = Image.fromarray(image)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    if LOGGING_LEVEL == 'VERBOSE':
        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    for face_landmarks in face_landmarks_list:
        data = find_lips(pil_image, face_landmarks)
        pil_lip = data['lip']
        pil_naked_lip = data['naked_lip']
        return {
            'lip': pil_lip,
            'naked_lip': pil_naked_lip
        }
    return None

def setup_video_input(name):
    vidcap = cv2.VideoCapture('%s.mp4' % name)
    '''
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidcap.get(cv2.CAP_PROP_FPS)
    '''
    return vidcap

def setup_directories():
    if not os.path.exists("videos"):
        os.makedirs("videos")
    if not os.path.exists("relative_percs"):
        os.makedirs("relative_percs")
    if not os.path.exists("percs"):
        os.makedirs("percs")
    if not os.path.exists("json"):
        os.makedirs("json")

def setup_video_output(name, dir, fps=30.0, width=256, height=256):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(dir + "/%s.mp4" % name, fourcc, fps, (width, height))
    return video

def reject_outliers(data, m = 2.):
    """
        removes outliers from list
        used to unify the list of lip areas somewhat
    """
    d = numpy.abs(data - numpy.median(data))
    mdev = numpy.median(d)
    s = d/mdev if mdev else 0.
    data_without_outliers = numpy.zeros(len(data))
    removedcount = 0
    for i in range(len(data)):
        if s[i]<m:
            data_without_outliers[i] = data[i]
        else:
            data_without_outliers[i] = 0
            removedcount += 1
    return data_without_outliers

#generate_weights('modric', batch_size=800)
#test_weights('modric')
