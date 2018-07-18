from argparse import ArgumentParser
from check_talking import *


parser = ArgumentParser()
parser.add_argument("-n", "--name", dest="name", required=True,
                    help="name of video without extension")

def test_weights(name):
    """
        generates video that shows when person is talking or not
        requires json file with wieghts in json directory
    """

    vidcap = setup_video_input(name)
    success, image = vidcap.read()
    height, width, layers =  image.shape

    fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))

    if not os.path.exists("test_videos"):
        os.makedirs("test_videos")

    video = setup_video_output(name, "test_videos", fps=fps, width=width, height=height)

    with open('json/%s.json' % name) as json_data:
        relative_percs = json.load(json_data)
    talking = []
    sum = 0

    try:
        relative_percs_without_outliers = reject_outliers(numpy.array(relative_percs[0:30]), m=1.5)
    except:
        relative_percs_without_outliers = numpy.array(relative_percs[0:30])

    internal_i = 0
    for i in range(len(relative_percs) / fps):
        t = abs(relative_percs_without_outliers)*100
        sum = t.sum(axis = 0)
        avg = sum/fps
        if LOGGING_LEVEL == 'VERBOSE':
            print avg
        try:
            relative_percs_without_outliers = reject_outliers(numpy.array(relative_percs[(i + 1) *30: (i + 1)*30+30]), m=1.5)
        except:
            relative_percs_without_outliers = numpy.array(relative_percs[(i + 1) *30: (i + 1)*30+30])
        internal_i = 0
        if avg > 0.4:
            talking.append(True)
        else:
            talking.append(False)
    if LOGGING_LEVEL == 'VERBOSE':
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

if __name__ == "__main__":
    args = parser.parse_args()
    d = vars(args)
    name = d['name']
    test_weights(name)
