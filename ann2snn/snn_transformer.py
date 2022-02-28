from snn_layers import *


def is_spike_layer(layer):
    return isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.AvgPool2d)


def add_snn_op(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear) or isinstance(child, nn.AvgPool2d):
            module._modules[name] = nn.Sequential(tdLayer(child), LIFSpike())
        elif isinstance(child, nn.ReLU):
            module._modules[name] = nn.Sequential(tdLayer(child))

    return module


class SNNTransformer(nn.Module):
    def __init__(self, args, ann_net, device):
        super(SNNTransformer, self).__init__()
        """
        The basic transformer to generate the snn for inference
        """
        self.ann_net = ann_net
        self.timesteps = args.timesteps
        self.device = device

    def generate_snn(self, train_loader, num_iters):
        """
        generate the snn model.
        """
        snn_net = add_snn_op(self.ann_net)
        # print(snn_net)
        snn_net.to(self.device)

        num_iters = min(num_iters, len(train_loader))
        spike_inputs = []
        for i, (data, target) in enumerate(train_loader):
            if i >= num_iters:
                break
            tensor_data, _ = torch.broadcast_tensors(data, torch.zeros((self.timesteps,) + data.shape))
            spike_data = SpikeTensor(tensor_data.permute(1, 2, 3, 4, 0).to(self.device), self.timesteps)
            spike_inputs.append(spike_data)

        out_scale = [1]
        view_op = 0   # view操作
        spike_view_op = 0
        for layer_name, layer_child in snn_net.named_children():
            spike_outputs = []
            tensor_outputs = []
            for tdname, tdchild in layer_child.named_children():
                if isinstance(tdchild, tdLayer):
                    # 计算Tensor输出
                    if is_spike_layer(tdchild.layer):
                        if tdchild.bn is None:
                            for name, child in tdchild.named_children():
                                if not isinstance(child, nn.Linear):
                                    for i in range(num_iters):
                                        tensor_input = spike_inputs[i].data.permute(4, 0, 1, 2, 3)
                                        tensor_input = torch.split(tensor_input, 1, dim=0)
                                        tensor_input = torch.cat(tensor_input, dim=1)
                                        tensor_input = tensor_input.squeeze(0)
                                        tensor_output = child(tensor_input)
                                        tensor_outputs.append(tensor_output)
                                else:
                                    if not view_op:
                                        view_op = 1
                                        for i in range(num_iters):
                                            tensor_input = spike_inputs[i].data.view(spike_inputs[i].data.shape[0], -1,
                                                                                     spike_inputs[i].data.shape[4])
                                            tensor_input = tensor_input.permute(2, 0, 1)
                                            tensor_input = torch.split(tensor_input, 1, dim=0)
                                            tensor_input = torch.cat(tensor_input, dim=1)
                                            tensor_input = tensor_input.squeeze(0)
                                            tensor_output = child(tensor_input)
                                            tensor_outputs.append(tensor_output)
                                    else:
                                        for i in range(num_iters):
                                            tensor_input = spike_inputs[i].data.permute(2, 0, 1)
                                            tensor_input = torch.split(tensor_input, 1, dim=0)
                                            tensor_input = torch.cat(tensor_input, dim=1)
                                            tensor_input = tensor_input.squeeze(0)
                                            tensor_output = child(tensor_input)
                                            tensor_outputs.append(tensor_output)

                        else:
                            raise NotImplementedError

                        # 计算Vth
                        delta_mem_potential_max = []
                        for i in range(num_iters):
                            b = tensor_outputs[i].size(0)
                            delta_mem_potential_max.append(
                                torch.quantile(tensor_outputs[i].detach().view(b, -1), 0.99, dim=1).mean())

                        delta_mem_potential_max = sum(delta_mem_potential_max) / len(delta_mem_potential_max)
                        out_scale.append(delta_mem_potential_max)
                        print("vth:", delta_mem_potential_max)

                    # 计算SpikeTensor输出
                    if not isinstance(tdchild.layer, nn.Linear):
                        for i in range(num_iters):
                            spike_outputs.append(tdchild(spike_inputs[i]))
                    else:
                        if not spike_view_op:
                            spike_view_op = 1
                            for i in range(num_iters):
                                spike_input = spike_inputs[i].data.view(spike_inputs[i].data.shape[0], -1,
                                                                        spike_inputs[i].data.shape[4])
                                spike_outputs.append(tdchild(spike_input))
                        else:
                            for i in range(num_iters):
                                spike_outputs.append(tdchild(spike_inputs[i]))

                elif isinstance(tdchild, LIFSpike):
                    tdchild.Vth = out_scale[-1]
                    for i in range(num_iters):
                        spike_inputs[i] = tdchild(spike_outputs[i])

        return snn_net

    def finetune_snn(self, snn_net, train_loader, criterion, epochs):
        optimizer = optim.SGD(snn_net.parameters(), lr=0.0001, momentum=0.9)
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            for inputs, labels in train_loader:
                # 训练数据
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                inputs, _ = torch.broadcast_tensors(inputs, torch.zeros((self.timesteps,) + inputs.shape))
                inputs = SpikeTensor(inputs.permute(1, 2, 3, 4, 0).to(self.device), self.timesteps)

                optimizer.zero_grad()

                output = snn_net(inputs).firing_ratio()
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                # print("epoch: ", epoch)

            print('epoch {}: SNN Prec@1: {}/{} ({:.2f}%)\n'.format(
                epoch, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

        return snn_net

    def save_snn(self, snn_net, save_path):
        snn = nn.ModuleList()
        for seqname, seqchild in snn_net.named_children():
            for tdname, tdchild in seqchild.named_children():
                if not isinstance(tdchild, LIFSpike):
                    if isinstance(tdchild.layer, nn.ReLU):
                        continue

                snn.append(tdchild)

        # Save the SNN
        torch.save(snn, save_path)
        # print(snn.state_dict())
        torch.save(snn.state_dict(), save_path + 'weight.pth')
        print("Save the SNN in {}".format(save_path))

    def load_snn(self, snn_net, ann2snn_checkpoint_path):
        checkpoint = torch.load(ann2snn_checkpoint_path)
        i = 0
        for name, child in snn_net.named_children():
            try:
                if not isinstance(child, LIFSpike):
                    child.layer = checkpoint[i].layer
                else:
                    child.Vth = checkpoint[i].Vth
            except TypeError as e:
                print("ERROR: the pretrained ann2snn network architecture is not the same as current model!")

            i += 1

        print('Model loaded.')

        return snn_net
