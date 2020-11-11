import torch
from src.utils import load_images_from_path, get_class
import torch.nn as nn
import numpy as np
from src.models.roadnet import RoadNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = torch.load('./model.pt')
classifier.eval()

imgs = load_images_from_path(['./data/video1/frame_72_.jpg'])
pr = classifier(torch.tensor(imgs[0].reshape(3, 224, 224))[None, ...].cuda()).to(torch.device('cpu'))
pr = get_class(np.argmax(pr.detach().numpy()))

print(pr)
