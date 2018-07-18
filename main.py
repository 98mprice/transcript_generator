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

fnt = ImageFont.truetype('Roboto-BlackItalic.ttf', 30)

LOGGING_LEVEL = 'MINIMAL'
THRESHOLD = 0.9

def generate_padded_image(raw_image):
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
    flipped_arr = numpy.flipud(image)
    return Image.fromarray(flipped_arr, 'RGB')

def count_similar_pixels(a, b):
    similarity_count = 0
    for i in range(128):
        for j in range(128):
            r_a, g_a, b_a = a.getpixel((i,j))
            r_b, g_b, b_b = b.getpixel((i,j))
            if (r_a == r_b) and (b_a == b_b):
                similarity_count +=  1
    return similarity_count

def similarity(a, b):
    similarity_count = count_similar_pixels(a, b)
    similarity = (float(similarity_count)/(128.0 * 128.0))
    if similarity < THRESHOLD:
        similarity_count = count_similar_pixels(flip(a), b)
        similarity_flipped = (float(similarity_count)/(128.0 * 128.0))
        if similarity_flipped > similarity:
            return similarity_flipped
    return similarity

def percentage_open(image):
    red_count = 0
    for i in range(128):
        for j in range(128):
            r, g, b = image.getpixel((i,j))
            if r > b:
                red_count +=  1
    '''
    pixels = image.getdata()
    counter = Counter(pixels)
    print counter[(255, 0, 0)], counter[(0, 0, 255)], (128 * 128)
    return float(counter[(255, 0, 0)])/(128.0 * 128.0)
    '''
    return float(red_count)/(128.0 * 128.0)

def PolyArea(x,y):
    return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))

def find_link_between(p1, p2, list):
    p1_index = list.index(p1)
    p2_index = list.index(p2)
    new_list = []
    if p2_index > p1_index:
        new_list = list[p1_index:p2_index+1]
    elif p1_index > p2_index:
        new_list = list[p2_index:p1_index+1]
    return new_list

def find_area(p):
    x = list(map(list, zip(*p)))
    area = PolyArea(x[0], x[1])
    return area

# finds relative area & draws image for a single face
def find_relative_area(pil_image, face_landmarks):
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

    # area of mouth
    area_inner_mouth = find_area(inner_mouth)
    # draw area of mouth
    d.polygon(inner_mouth, fill=(255, 0, 0, 128))

    # draw rest of facial features overlayed
    #for facial_feature in face_landmarks.keys():
    #    d.line(face_landmarks[facial_feature], width=5)

    # percentage of facial area that is open mouth
    relative_area = (area_inner_mouth / area_chin) * 100

    d.rectangle([face_landmarks['chin'][0], (face_landmarks['chin'][0][0] + 50, face_landmarks['chin'][0][1] + 40)], fill=(255, 255, 255, 128))
    d.text(face_landmarks['chin'][0], str(int(round(relative_area))) + "%", font=fnt, fill=(0, 0, 0, 128))
    return {
        'area': relative_area,
        'frame': pil_image,
        'lip': pil_image.crop((min(inner_mouth, key = lambda t: t[0])[0], min(inner_mouth, key = lambda t: t[1])[1], max(inner_mouth, key = lambda t: t[0])[0], max(inner_mouth, key = lambda t: t[1])[1])),
        'naked_lip': naked_image.crop((min(inner_mouth, key = lambda t: t[0])[0], min(inner_mouth, key = lambda t: t[1])[1], max(inner_mouth, key = lambda t: t[0])[0], max(inner_mouth, key = lambda t: t[1])[1]))
    }

def find_faces_and_relative_area(image):
    # Load the jpg file into a numpy array
    #image = face_recognition.load_image_file("test1.jpg")
    pil_image = Image.fromarray(image)
    #pil_image.thumbnail((426, 240), Image.ANTIALIAS)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    if LOGGING_LEVEL == 'VERBOSE':
        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    for face_landmarks in face_landmarks_list:
        data = find_relative_area(pil_image, face_landmarks)
        pil_image = data['frame']
        pil_lip = data['lip']
        pil_naked_lip = data['naked_lip']
        return {
            'frame': pil_image,
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

def setup_video_output(name):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter("videos/%s.mp4" % name, fourcc, 30.0, (256, 256))
    return video

def generate_weights(name, batch_size):

    vidcap = setup_video_input(name)
    success, image = vidcap.read()
    if batch_size == None:
        batch_size = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    setup_directories()

    video = setup_video_output(name)

    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)

    previous_perc = 0
    relative_percs = numpy.zeros(batch_size)
    percs = numpy.zeros(batch_size)
    x = numpy.arange(batch_size)
    bar = progressbar.ProgressBar(max_value = batch_size)
    for i in range(batch_size):
        background = Image.new('RGB', (256, 256), (255, 255, 255))
        try:
            data = find_faces_and_relative_area(image)
            if data != None:
                lip_image = data['lip']
                padded_lip_image = generate_padded_image(lip_image)
                if padded_lip_image != None:
                    # binary colour lip
                    background.paste(padded_lip_image, (0, 0))

                    # setup arrays
                    d = ImageDraw.Draw(background)
                    perc = percentage_open(padded_lip_image)
                    perc_relative = perc - previous_perc
                    relative_percs[i] = perc_relative
                    percs[i] = perc
                    previous_perc = perc

                    # percentage
                    d.text((128, 0), str(int(round(perc*100))) + "%", font=fnt, fill=(0, 0, 0, 128))

                    # percentage graph
                    ax2.plot(x, numpy.array(percs), color='r')
                    f2.savefig('percs/%s.png' % name)

                    # relative percentage
                    d.text((128, 25), str(int(round(perc_relative*100))) + "% ~ rel", font=fnt, fill=(0, 0, 0, 128))

                    # relative percentage graph
                    ax1.plot(x, numpy.array(relative_percs), color='b')
                    f1.savefig('relative_percs/%s.png' % name)
                # lip without any overlay
                naked_lip_image = data['naked_lip']
                background.paste(naked_lip_image, (128, 64))
            else:
                previous_perc = 0
        except:
            print "an error occoured"

        try:
            if os.path.exists('percs/%s.png' % name):
                graph = Image.open('percs/%s.png' % name)
                graph.thumbnail((128, 128))
                background.paste(graph, (128, 128))

            if os.path.exists('relative_percs/%s.png' % name):
                graph = Image.open('relative_percs/%s.png' % name)
                graph.thumbnail((128, 128))
                background.paste(graph, (0, 128))
        except:
            print 'an error occoured'

        video.write(numpy.array(background))
        success, image = vidcap.read()
        bar.update(i)

    # write weights to file
    with open('json/%s.json' % name, 'w') as outfile:
        json.dump(relative_percs.tolist(), outfile)
    cv2.destroyAllWindows()
    video.release()

def reject_outliers(data, m = 2.):
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
    print 'removed', removedcount
    return data_without_outliers

def test_weights(name):
    vidcap = cv2.VideoCapture('%s.mp4' % name)
    success, image = vidcap.read()
    height, width, layers =  image.shape
    fps    = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
    print fps

    if not os.path.exists("test_videos"):
        os.makedirs("test_videos")

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter("test_videos/%s.mp4" % name, fourcc, fps, (width, height))

    with open('json/%s.json' % name) as json_data:
        relative_percs = json.load(json_data)
    talking = []
    sum = 0
    #relative_percs = reject_outliers(numpy.array(relative_percs), m=10)
    '''plt.plot(numpy.arange(len(relative_percs_without_outliers)), relative_percs_without_outliers)
    plt.savefig('huu.png')'''
    try:
        relative_percs_without_outliers = reject_outliers(numpy.array(relative_percs[0:30]), m=1.5)
    except:
        relative_percs_without_outliers = numpy.array(relative_percs[0:30])
    internal_i = 0
    print len(relative_percs) / fps
    for i in range(len(relative_percs) / fps):
        t = abs(relative_percs_without_outliers)*100
        sum = t.sum(axis = 0)
        avg = sum/fps
        print avg
        try:
            relative_percs_without_outliers = reject_outliers(numpy.array(relative_percs[(i + 1) *30: (i + 1)*30+30]))
        except:
            relative_percs_without_outliers = numpy.array(relative_percs[(i + 1) *30: (i + 1)*30+30])
        #relative_percs_without_outliers = reject_outliers(numpy.array(relative_percs[i*30:i*30+30]))
        internal_i = 0
        if avg > 0.4:
            talking.append(True)
        else:
            talking.append(False)
    print talking
    bar = progressbar.ProgressBar(max_value = len(relative_percs))
    for i in range(len(relative_percs)):
        drawable_image = Image.fromarray(image)
        d = ImageDraw.Draw(drawable_image)
        d.rectangle([(32, 32), (128, 64)], fill=(255, 255, 255, 128))
        if len(talking) > int(math.floor(i/30)):
            d.text((32, 32), '%s' % talking[int(math.floor(i/30))], font=fnt, fill=(0, 0, 0, 128))
        video.write(numpy.array(drawable_image))
        bar.update(i)
        success, image = vidcap.read()
    cv2.destroyAllWindows()
    video.release()
    '''current_talking_count = 0
    bar = progressbar.ProgressBar(max_value = len(relative_percs_without_outliers))
    for i in range(len(relative_percs_without_outliers)):
        if i % fps == 0 and i != 0:
            current_talking_count += 1
        drawable_image = Image.fromarray(image)
        d = ImageDraw.Draw(drawable_image)
        try:
            d.rectangle([(32, 32), (128, 64)], fill=(255, 255, 255, 128))
            d.text((32, 32), '%s' % talking[current_talking_count], font=fnt, fill=(0, 0, 0, 128))
        except:
            print 'an error occoured'
        video.write(numpy.array(drawable_image))
        bar.update(i)
        success, image = vidfps.read()
    cv2.destroyAllWindows()
    video.release()'''

#generate_weights('modric', batch_size=800)
test_weights('modric')
