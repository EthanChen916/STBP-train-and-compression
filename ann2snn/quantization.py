import copy
import torch
import torch.nn as nn
import torch.optim as optim

quantized_layers = []


def quantize_tensor(tensor, bitwidth, channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
    scale = (2 ** (bitwidth - 1) - 1) / _max
    if tensor.dim() == 4:
        scale = scale.view(-1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1)

    #new_tensor = torch.round(scale * tensor)
    new_tensor = scale * tensor
    new_tensor = (new_tensor.round() - new_tensor).detach() + new_tensor
    return new_tensor, scale


def init_quantize_net(net):
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            quantized_layers.append(m)


def quantize_layers(bitwidth, rescale=True):
    for i, layer in enumerate(quantized_layers):
        with torch.no_grad():
            quantized_w, scale_w = quantize_tensor(layer.weight, bitwidth, False)
            layer.weight[...] = quantized_w/scale_w if rescale else quantized_w

            if layer.bias != None:
                quantized_b, scale_b = quantize_tensor(layer.bias, bitwidth, False)
                layer.bias[...] = quantized_b/scale_b if rescale else quantized_b


def weightsdistribute(model):
    print("================show every layer's weights distribute================")
    for key, value in model.named_parameters():
        print("================="+key+"=================")
        unique, count = torch.unique(value.detach(), sorted=True, return_counts= True)
        print(unique.shape)


def quantize_train(epoch, net, trainloader, optimizer, criterion, device, bitwidth):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        quantize_layers(bitwidth)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 60 == 59:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def quantize_finetune(raw_net, trainloader, criterion, device, args):
    net = copy.deepcopy(raw_net).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.finetune_lr)
    init_quantize_net(net)
    for epoch in range(0, args.finetune_epochs):
        quantize_train(epoch, net, trainloader, optimizer, criterion, device, args.weight_bitwidth)
    quantize_layers(args.weight_bitwidth)
    quantized_layers.clear()
    return net
