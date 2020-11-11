from src.utils import get_class,preproc
from argparse import ArgumentParser
import torch
import cv2
import numpy as np
from src.models.roadnet import RoadNet

cpu_model = './pretrained_model/model_cpu.pt'
gpu_model = './pretrained_model/model_gpu.pt'#'./model.pt'

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

def load_model(cpu):
    if cpu:
        classifier = torch.load(cpu_model)
    else:
        classifier = torch.load(gpu_model)
    classifier.eval()
    return classifier


def get_avi(path_to_video, output_file,save_gray= False, cpu = False):
    cap = cv2.VideoCapture(path_to_video)
    if cpu is None:
        if torch.cuda.is_available():
            cpu = False
    model = load_model(cpu)
    result_frame_array = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preproc_frame = cv2.resize(preproc(frame), (224, 224))
        preproc_frame = np.array([preproc_frame, preproc_frame, preproc_frame]).reshape(3, 224, 224)
        if cpu:
            pred = model(torch.tensor(preproc_frame)[None, ...]).to(torch.device('cpu'))
        else:
            pred = model(torch.tensor(preproc_frame)[None, ...].cuda()).to(torch.device('cpu'))
        pred = pred.detach().numpy().tolist()[0]
        max1 = np.argmax(pred)
        max2_val = -100000
        for i, el in enumerate(pred):
            if el != pred[max1] and el > max2_val:
                max2_val = el
                max2 = i
        pred = get_class(max1) + '/' + get_class(max2)
        if save_gray:
            gray_three = cv2.merge(preproc_frame)
            cv2.putText(gray_three, pred, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            result_frame_array.append(gray_three)
        else:
            cv2.putText(frame, pred, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            result_frame_array.append(frame)

    cap.release()
    height, width, layers = result_frame_array[0].shape
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
    for i in result_frame_array:
        out.write(i)
    out.release()


get_avi(args['input_path'], args['output_path'], args['canny'])
