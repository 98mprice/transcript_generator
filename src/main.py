from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-n", "--name", dest="name",
                    help="name of video without extension")

args = parser.parse_args()
d = vars(args)
print d['name']
