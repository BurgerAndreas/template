import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyModel(nn.Module):
    """My model.
    """
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_mymodel():
    # training data

    batch_size = 2
    image_size = 256

    model = MyModel(image_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    img = torch.randn(batch_size, 3, image_size, image_size)
    target = torch.randint(10, (batch_size,))

    model = MyModel(image_size)

    # train

    optimizer.zero_grad()
    output = model(img)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print('loss:', loss.detach())


if __name__ == '__main__':
    test_mymodel()