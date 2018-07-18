from argparse import ArgumentParser
from check_talking import *


parser = ArgumentParser()
parser.add_argument("-n", "--name", dest="name", required=True,
                    help="name of video without extension")
parser.add_argument("-b", "--batch_size", dest="batch_size",
                    help="number of frames to be read from the video. leave blank to read whole video")

def generate_weights(name, batch_size, make_video=True):
    """
        generates the percentages that the lips are open in each frame and saves to a
        json file
        also generates a mp4 video for debugging, unless make_video is set to false
    """

    vidcap = setup_video_input(name)
    success, image = vidcap.read()
    if batch_size == None:
        batch_size = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    setup_directories()

    if make_video == True:
        video = setup_video_output(name, "videos")

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
        if make_video == True:
            video.write(numpy.array(background))
        success, image = vidcap.read()
        bar.update(i)

    # write weights to file
    with open('json/%s.json' % name, 'w') as outfile:
        json.dump(relative_percs.tolist(), outfile)
    if make_video == True:
        cv2.destroyAllWindows()
        video.release()

if __name__ == "__main__":
    args = parser.parse_args()
    d = vars(args)
    name, batch_size = d['name'], d['batch_size']
    generate_weights(name, batch_size=batch_size)
