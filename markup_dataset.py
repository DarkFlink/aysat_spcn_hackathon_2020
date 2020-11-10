import os,sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--dir_path", type=str,
                   help="path to dir with input images")
parser.add_argument("-o", "--output", type=str,
                   help="JSON output filename")
args = vars(parser.parse_args())

tiles = ["right", "left", "straight", "three_cross", "four_cross", "empty"]
filename = ""
markup = []


def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == '1':
        markup.append({filename: tiles[0]})
        plt.close('all')
    if event.key == '2':
        markup.append({filename: tiles[1]})
        plt.close('all')
    if event.key == '3':
        markup.append({filename: tiles[2]})
        plt.close('all')
    if event.key == '4':
        markup.append({filename: tiles[3]})
        plt.close('all')
    if event.key == '5':
        markup.append({filename: tiles[4]})
        plt.close('all')
    if event.key == '6':
        markup.append({filename: tiles[5]})
        plt.close('all')


def markup_dataset(dir='./data/'):
    list = os.listdir(dir)
    # sort files names in list
    list.sort()
    for f in list:
        img_format = os.path.splitext(f)[1]
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', press)
        ax = fig.subplots()
        global filename
        filename = dir + f
        img = mpimg.imread(filename)
        ax.imshow(img)
        plt.text(10, 25, "1 - Right rot", fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        plt.text(10, 65, "2 - Left rot", fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        plt.text(10, 105, "3 - Straight way", fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        plt.text(10, 145, "4 - Three way center", fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        plt.text(10, 185, "5 - Four way center", fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        plt.text(10, 225, "6 - Tile empty", fontsize=12, bbox=dict(facecolor='red', alpha=0.7))
        plt.show()

    return markup


res = markup_dataset(args['dir_path'])
with open(args['output'], "w") as outfile:
    json.dump(markup, outfile)