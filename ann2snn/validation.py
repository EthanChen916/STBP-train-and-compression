import torch
from snn_layers import SpikeTensor


def validate_snn(net, test_loader, device, criterion, timesteps):
    """Perform validation for SNN"""
    print("Performing validation for SNN")

    # switch to evaluate mode
    net.eval()

    correct = 0

    with torch.no_grad():
        for data_test, target in test_loader:
            data, target = data_test.to(device), target.to(device)

            data, _ = torch.broadcast_tensors(data, torch.zeros((timesteps,) + data.shape))
            data = SpikeTensor(data.permute(1, 2, 3, 4, 0).to(device), timesteps)

            output = net(data).firing_ratio()

            loss = criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('SNN Prec@1: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def validate_ann(net, test_loader, device, criterion):
    """Perform validation for ANN"""
    print("Performing validation for ANN")

    # switch to evaluate mode
    net.eval()
    correct = 0

    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test
            data = data.to(device)
            target = target.to(device)

            output = net(data)

            loss = criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('ANN Prec@1: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
