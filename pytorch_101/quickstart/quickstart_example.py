"""
This module provides a quick start to pytorch
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


BATCH_SIZE = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X_, y_ in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X_.shape}")
    print(f"Shape of y: {y_.shape} {y_.dtype}")
    break

# Get cpu or gpu device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


# Define model
class NeuralNetwork(nn.Module):
    """
    A basic NeuralNetwork
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, data_x):
        """
        forward part of the neural net
        :param data_x:
        :return:
        """
        data_x = self.flatten(data_x)
        logits = self.linear_relu_stack(data_x)
        return logits


MODEL = NeuralNetwork().to(DEVICE)
print(MODEL)

# optimizing the model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MODEL.parameters(), lr=1e-3)


# define train and tes


def train(dataloader: datasets, input_model, input_loss_fn, input_optimizer) -> None:
    """
    Training function
    :param dataloader:
    :param input_model:
    :param input_loss_fn:
    :param input_optimizer:
    :return:
    """
    size = len(dataloader.dataset)
    input_model.train()
    # pylint: disable=C0103,W0621
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = input_model(X)
        loss = input_loss_fn(pred, y)

        # Backpropagation
        input_optimizer.zero_grad()
        loss.backward()
        input_optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, input_model, input_loss_fn):
    """
    Test function
    :param dataloader:
    :param input_model:
    :param input_loss_fn:
    :return:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    input_model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        # pylint: disable=C0103,W0621
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = input_model(X)
            test_loss += input_loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%,"
        f"Avg loss: {test_loss:>8f} \n"
    )


# train and test the model

EPOCHS = 5
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, MODEL, loss_fn, optimizer)
    test(test_dataloader, MODEL, loss_fn)
print("Done!")

# save the model
torch.save(MODEL.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# load the model

MODEL = NeuralNetwork()
MODEL.load_state_dict(torch.load("model.pth"))

# Make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

MODEL.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = MODEL(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
