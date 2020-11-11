import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from src.utils import get_train_x_y
from sklearn.model_selection import train_test_split
from src.models.roadnet import RoadNet

epochs = 30
batch_size = 20
learn_rate = 0.1
num_classes = 6
tiles = {"right":0, "left":1, "straight":2, "three_cross":3, "four_cross":4, "empty":5}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DuckieRoadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Y = Y
        self.X = X

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.X[idx].reshape((3,224,224)), self.Y[idx]]

        return sample


classifier = RoadNet()
crossentropy_loss = nn.CrossEntropyLoss()
opt = Adam(classifier.parameters(), lr=learn_rate)

x_data, y_data = get_train_x_y('../data')
x_data = x_data / 255.0
#x_data = x_data[400:600]
#y_data = y_data[400:600]
y_data = [tiles[i] for i in y_data]

train_X, test_X, train_Y, test_Y = train_test_split(x_data, y_data, test_size=0.2)

train_data = DuckieRoadDataset(
    X = train_X,
    Y = train_Y
)
test_data = DuckieRoadDataset(
    X = test_X,
    Y = test_Y
)

#train_data = CIFAR10(root='./datasets/', train=True, download=True, transform=transform)
#test_data = CIFAR10(root='./datasets/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

history_test_loss = []
history_train_loss = []
history_test_acc = []
history_train_acc = []

# fit model
# main epochs for
for epoch in range(epochs):
    classifier.train()
    # fitting procedure
    train_loss = 0.0
    train_acc = 0.0
    for idx, (X, Y) in enumerate(train_loader):
        opt.zero_grad()

        # forward -->
        model_output = classifier.forward(X)

        # classify model output
        loss_output = crossentropy_loss(model_output, Y)

        # backward <--
        loss_output.backward()

        train_acc += (model_output.argmax(dim=1) == Y).float().mean()
    #    print("TRAINACC:", train_acc)

        # optimizer need to update weights
        opt.step()

        train_loss += loss_output.cpu().item() * X.size(0)

    # testing
    classifier.eval()
    test_loss = 0.0
    test_acc = 0.0
    correct = 0.0
    total = 0.0
    for idx, (X,Y) in enumerate(test_loader):

        model_output = classifier.forward(X)

        loss_output = crossentropy_loss(model_output, Y)

        test_loss += loss_output.cpu().item() * X.size(0)

        correct += (model_output.argmax(dim=1) == Y).sum().item()
        total += Y.size(0)
       # test_acc += (model_output.argmax(dim=1) == Y).float().mean()

    # for tracking history
    test_loss = test_loss / test_X.__len__()
    train_loss = train_loss / train_X.__len__()
 #   test_acc = test_acc * batch_size / test_X.__len__()
    train_acc = train_acc * batch_size / train_X.__len__()
    history_train_loss.append(train_loss)
    history_test_loss.append(test_loss)
    history_train_acc.append(train_acc)
    history_test_acc.append(correct/total)

    print(f'Epoch: {epoch}, loss: {train_loss:.5f}, val_loss: {test_loss:.5f}')
    print(f'                val_acc: {correct/total:.5f}')

plt.figure(1, figsize=(10, 10))
plt.title("Loss")
plt.plot(history_test_loss, 'r', label='test')
plt.plot(history_train_loss, 'b', label='train')
plt.legend()
plt.show()
plt.clf()

plt.figure(1, figsize=(10, 10))
plt.title("Loss")
plt.plot(history_test_acc, 'r', label='test')
plt.plot(history_train_acc, 'b', label='train')
plt.legend()
plt.show()
plt.clf()