from torch import nn

class RoadNet(nn.Module):
    def __init__(self):
        super(RoadNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.batn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.batn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.batn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3 = nn.Dropout(p=0.1)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.batn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.drop4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(in_features=14 * 14 * 64, out_features=6)

    def forward(self, input):

        output = self.conv1(input.float())
        output = self.relu1(output)
        output = self.batn1(output)
        output = self.pool1(output)
        output = self.drop1(output)

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.batn2(output)
        output = self.pool2(output)
        output = self.drop2(output)

        output = self.conv3(output)
        output = self.relu3(output)
        output = self.batn3(output)
        output = self.pool3(output)
        output = self.drop3(output)

        output = self.conv4(output)
        output = self.relu4(output)
        output = self.batn4(output)
        output = self.pool4(output)
        output = self.drop4(output)

        output = output.view(-1, 14 * 14 * 64)
        output = self.fc(output)

        return output