from snn_layers import *
from generate_snn_network import SNNGenerate


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
        snn_net = SNNGenerate(self.ann_net)
        # print(snn_net)
        snn_net.to(self.device)

        num_iters = min(num_iters, len(train_loader))
        spike_inputs = []
        for i, (data, target) in enumerate(train_loader):
            if i >= num_iters:
                break
            spike_data, _ = torch.broadcast_tensors(data, torch.zeros((self.timesteps,) + data.shape))
            spike_inputs.append(spike_data.to(self.device))

        out_scale = [1]
        spike_outputs = []
        tensor_outputs = []
        view_fc = 0   # view操作
        spike_view_fc = 0
        for layer_name, layer_child in snn_net.named_children():
            for tdname, tdchild in layer_child.named_children():
                if isinstance(tdchild, tdLayer):
                    # 计算Tensor输出
                    if tdchild.bn is None:
                        for name, child in tdchild.named_children():
                            if not isinstance(child, nn.Linear) or view_fc:
                                for i in range(num_iters):
                                    tensor_input = torch.split(spike_inputs[i], 1, dim=0)
                                    tensor_input = torch.cat(tensor_input, dim=1)
                                    tensor_input = tensor_input.squeeze()
                                    tensor_output = child(tensor_input * 1)
                                    tensor_outputs.append(tensor_output)
                            else:
                                view_fc = 1
                                for i in range(num_iters):
                                    tensor_input = spike_inputs[i].view(spike_inputs[i].shape[0], spike_inputs[i].shape[1], -1)
                                    tensor_input = torch.split(tensor_input, 1, dim=0)
                                    tensor_input = torch.cat(tensor_input, dim=1)
                                    tensor_input = tensor_input.squeeze()
                                    tensor_output = child(tensor_input * 1)
                                    tensor_outputs.append(tensor_output)

                    else:
                        raise NotImplementedError
                        # for name, child in tdchild.named_children():
                        #     if not isinstance(child, nn.BatchNorm2d):
                        #         for i in range(num_iters):
                        #             tensor_input = torch.split(spike_inputs[i], 1, dim=0)
                        #             tensor_input = torch.cat(tensor_input, dim=1)
                        #             tensor_input = tensor_input.squeeze()
                        #             tensor_output = child(tensor_input * 1)
                        #             tensor_outputs.append(tensor_output)
                        #     else:
                        #         for i in range(num_iters):
                        #             tensor_outputs[i] = child(tensor_outputs[i])

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
                    if not isinstance(tdchild.layer, nn.Linear) or spike_view_fc:
                        for i in range(num_iters):
                            spike_outputs.append(tdchild(spike_inputs[i] * 1))
                    else:
                        spike_view_fc = 1
                        for i in range(num_iters):
                            spike_input = spike_inputs[i].view(spike_inputs[i].shape[0], spike_inputs[i].shape[1], -1)
                            spike_outputs.append(tdchild(spike_input * 1))

                elif isinstance(tdchild, LIFSpike):
                    tdchild.Vth = out_scale[-1]
                    for i in range(num_iters):
                        spike_inputs[i] = tdchild(spike_outputs[i])

                    spike_outputs.clear()
                    tensor_outputs.clear()

        return snn_net


if __name__ == '__main__':
    # just for debuging the function of the transformer
    import torch.nn as nn
    import argparse

    args = argparse.Namespace(timesteps=16, weight_bitwidth=4)
    net = nn.Sequential(nn.Conv2d(2, 2, 3), nn.BatchNorm2d(2), nn.ReLU(), nn.Conv2d(2, 5, 1), nn.BatchNorm2d(5), nn.ReLU(), nn.Conv2d(5, 2, 3),
                        nn.ReLU(), nn.Linear(2*4*4, 2))
    net[0].weight.data.fill_(1.01)
    net[3].weight.data.fill_(1.01)
    net[6].weight.data.fill_(1.01)
    net[8].weight.data.fill_(1.01)
    net[0].weight.alpha = 0.5
    net[3].weight.alpha = 1
    net[6].weight.alpha = 10
    # net = nn.Sequential(nn.Conv2d(2, 2, 3), nn.ReLU())
    # net[0].weight.alpha=0.1
    snn = SNNTransformer(args, net, torch.device('cpu'))

    input_src = torch.rand([4, 2, 8, 8])

    input, _ = torch.broadcast_tensors(input_src, torch.zeros((steps,) + input_src.shape))
    input = input.permute(1, 2, 3, 4, 0)

    snn_net = snn.generate_snn(30, input_src)

    output = snn_net(input)
    print(output)
