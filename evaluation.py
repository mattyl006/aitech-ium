import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from train import MLP, AtpDataset, test

def load_model():
    model = MLP()
    model.load_state_dict(torch.load('./model.zip'))
    return model

def load_dev_dataset(batch_size=64):
    atp_dev = AtpDataset('atp_dev.csv')
    return DataLoader(atp_dev, batch_size=batch_size)

def make_plot(values):
    build_nums = list(range(1, len(values) + 1))
    plt.xlabel('Build number')
    plt.ylabel('MSE Loss')
    plt.plot(build_nums, values, label='Model MSE Loss over builds')
    plt.legend()
    plt.savefig('plot.png')

model = load_model()
dataloader = load_dev_dataset()

loss_fn = torch.nn.MSELoss()

loss = test(dataloader, model, loss_fn)
with open('eval_result.txt', 'a+') as f:
    f.write(f'{str(loss)}\n')
with open('eval_result.txt', 'r') as f:
    values = [float(line) for line in f.readlines() if line]
    make_plot(values)
