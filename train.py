from ast import arg
import numpy as np
import pandas as pd
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, Dataset

default_batch_size = 64
default_epochs = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


class AtpDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_csv(file_name, usecols=["AvgL", "AvgW"])
        df = df.dropna()

        # Loser avg and Winner avg
        x = df.iloc[:, 1].values
        y = df.iloc[:, 0].values

        self.x_train = torch.from_numpy(x)
        self.y_train = torch.from_numpy(y)
        self.x_train.type(torch.LongTensor)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx].float(), self.y_train[idx].float()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss (using {loss_fn}): {test_loss:>8f} \n")
    return test_loss


def setup_args():
    args_parser = argparse.ArgumentParser(prefix_chars="-")
    args_parser.add_argument("-b", "--batchSize", type=int, default=default_batch_size)
    args_parser.add_argument("-e", "--epochs", type=int, default=default_epochs)
    return args_parser.parse_args()


print(f"Using {device} device")

args = setup_args()
batch_size = args.batchSize

plant_test = AtpDataset("atp_test.csv")
plant_train = AtpDataset("atp_train.csv")

train_dataloader = DataLoader(plant_train, batch_size=batch_size)
test_dataloader = DataLoader(plant_test, batch_size=batch_size)

for i, (data, labels) in enumerate(train_dataloader):
    print(data.shape, labels.shape)
    print(data, labels)
    break

model = MLP()
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = args.epochs

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Finish!")

torch.save(model.state_dict(), "./model.zip")
print("Model saved in ./model.zip file.")
