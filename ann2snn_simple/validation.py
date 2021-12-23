import torch
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate_snn(net,test_loader,device,criterion,timesteps):
    """Perform validation for SNN"""
    print("Performing validation for SNN")

    # switch to evaluate mode
    net.eval()

    correct = 0

    with torch.no_grad():
        for data_test, target in test_loader:
            data, target = data_test.to(device), target.to(device)

            data, _ = torch.broadcast_tensors(data, torch.zeros((timesteps,) + data.shape))

            output = net(data)

            loss = criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('SNN Prec@1: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def validate_ann(net,test_loader, device, criterion):
    """Perform validation for ANN"""
    print("Performing validation for ANN")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()
    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test
            data = data.to(device)
            output = net(data)

            target = target.to(device)
            if isinstance(output,tuple):
                loss=0
                for o in output:
                    loss+=criterion(o,target)
                output=output[0]
            else:
                loss = criterion(output, target)
            if len(output[0])>=5:
                topk=(1,5)
            else:
                topk=(1,1)
            prec1, prec5 = accuracy(output.data, target, topk=topk)
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(f'ANN Prec@{topk[0]} {top1.avg:.3f}, Prec@{topk[1]} {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}')
    return top1.avg, losses.avg