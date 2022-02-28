import os
import torch
import torch.nn as nn
from snn_layers import *
import datasets
import argparse


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()

        self.conv1_s = tdLayer(nn.Conv2d(3, 8, 5, bias=False))
        self.spike1 = LIFSpike()
        self.pool1_s = tdLayer(nn.AvgPool2d(2))
        self.spike2 = LIFSpike()
        self.conv2_s = tdLayer(nn.Conv2d(8, 16, 3, bias=False))
        self.spike3 = LIFSpike()
        self.pool2_s = tdLayer(nn.AvgPool2d(2))
        self.spike4 = LIFSpike()
        self.conv3_s = tdLayer(nn.Conv2d(16, 32, 3, bias=False))
        self.spike5 = LIFSpike()
        self.fc1_s = tdLayer(nn.Linear(4*4*32, 10, bias=False))
        self.spike6 = LIFSpike()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike1(x)
        x = self.pool1_s(x)
        x = self.spike2(x)
        x = self.conv2_s(x)
        x = self.spike3(x)
        x = self.pool2_s(x)
        x = self.spike4(x)
        x = self.conv3_s(x)
        x = self.spike5(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike6(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


def load_snn(model, ann2snn_checkpoint_path):

    checkpoint_path = ann2snn_checkpoint_path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        i = 0
        for name, child in model.named_children():
            try:
                if not isinstance(child, LIFSpike):
                    child.layer = checkpoint[i].layer
                else:
                    child.Vth = checkpoint[i].Vth
            except TypeError as e:
                print("ERROR: the pretrained snn network architecture is not the same as current model!")

            i += 1

        print('Model loaded.')

    return model


def test_snn(model, test_loader):
    print("Performing validation for SNN")

    # switch to evaluate mode
    model.eval()

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, _ = torch.broadcast_tensors(data, torch.zeros((16,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('SNN Prec@1: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR10',
                        type=str, help='the location of the dataset')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for dataset')
    parser.add_argument('--save_file', default="./out_snn.pth",
                        type=str, help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--device', default='1', type=str, help='choose cuda devices')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    model = ExampleNet()
    ann2snn_checkpoint_path = './out_snn.pth'
    model = load_snn(model, ann2snn_checkpoint_path)

    # Preparing the dataset
    train_loader, test_loader = datasets.get_dataset(args)
    test_snn(model, test_loader)
