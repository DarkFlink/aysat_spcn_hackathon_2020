from src.utils import get_class,preproc
from argparse import ArgumentParser
import cv2
tiles = {0:"right", 1:"left", 2:"straight", 3:"three_cross", 4:"four_cross", 5:"empty"}
parser = ArgumentParser()
parser.add_argument("-i", "--input_path", type=str,
                   help="path to dir with input video")
parser.add_argument("-o", "--output_path", type=str, default='./out.mp4',
                   help="video output filename")
parser.add_argument("--gpu", default="False",
                   help="cpu or gpu prediction", action='store_true')
parser.add_argument("--canny", default="False",
                   help="cpu or gpu prediction", action='store_true')
args = vars(parser.parse_args())


def get_avi(path_to_video,output_file,save_gray= False,gpu = True):
    cap = cv2.VideoCapture(path_to_video)
    result_frame_array = []
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        #predict function need to be here
        preproc_frame = preproc(frame)
        if save_gray:
            gray_three = cv2.merge([preproc_frame, preproc_frame, preproc_frame])
            cv2.putText(gray_three, get_class(3), (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            result_frame_array.append(gray_three)
        else:
            cv2.putText(frame, get_class(3), (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            result_frame_array.append(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    height, width, layers = result_frame_array[0].shape
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width,height))
    for i in result_frame_array:
        out.write(i)
    out.release()
