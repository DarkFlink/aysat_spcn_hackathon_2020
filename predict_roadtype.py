from src.utils import get_avi
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--input_path", type=str,
                   help="path to dir with input video")
parser.add_argument("-o", "--output_path", type=str, default='./out.mp4',
                   help="video output filename")
parser.add_argument("--cpu", default="False",
                   help="cpu or gpu prediction", action='store_true')
parser.add_argument("--canny", default="False",
                   help="cpu or gpu prediction", action='store_true')
args = vars(parser.parse_args())
print(args)

get_avi(args['input_path'],args['output_path'],args['canny'],args['cpu'])