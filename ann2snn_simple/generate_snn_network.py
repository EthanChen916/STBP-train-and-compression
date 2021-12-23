from snn_layers import *


class SNNGenerate(nn.Module):
    def __init__(self, net):
        super(SNNGenerate, self).__init__()
        """
        The basic transformer to generate the snn for inference on cpu
        """

        self.layers1 = nn.Sequential()
        self.layers2 = nn.Sequential()

        childs = []
        for name, child in net.named_children():
            childs.append(child)

        pool_cnt = 0
        conv_cnt = 0
        fc_cnt = 0
        LIFspike_cnt = 0
        for i in range(len(childs)):
            if isinstance(childs[i], nn.Conv2d):
                if isinstance(childs[i+1], nn.BatchNorm2d):
                    self.layers1.add_module('conv'+str(conv_cnt)+'_s', tdLayer(childs[i], childs[i+1]))
                else:
                    self.layers1.add_module('conv'+str(conv_cnt)+'_s', tdLayer(childs[i]))
                self.layers1.add_module('spike'+str(LIFspike_cnt), LIFSpike())
                conv_cnt += 1
                LIFspike_cnt += 1

            elif isinstance(childs[i], nn.AvgPool2d):
                self.layers1.add_module('pool'+str(pool_cnt)+'_s', tdLayer(childs[i]))
                self.layers1.add_module('spike'+str(LIFspike_cnt), LIFSpike())
                pool_cnt += 1
                LIFspike_cnt += 1

            elif isinstance(childs[i], nn.Linear):
                if i < len(childs)-1 and isinstance(childs[i + 1], nn.BatchNorm2d):
                    self.layers2.add_module('fc'+str(fc_cnt)+'_s', tdLayer(childs[i], childs[i+1]))
                else:
                    self.layers2.add_module('fc'+str(fc_cnt)+'_s', tdLayer(childs[i]))
                self.layers2.add_module('spike'+str(LIFspike_cnt), LIFSpike())
                fc_cnt += 1
                LIFspike_cnt += 1

    def forward(self, x):
        x = self.layers1(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.layers2(x)
        x = torch.sum(x, dim=0) / steps

        return x


if __name__ == '__main__':
    # just for debuging the function of the transformer
    import torch.nn as nn
    import argparse
    from snn_transformer_tdlayers import SNNTransformer

    args = argparse.Namespace(timesteps=16, relu_threshold=12, quantization_channel_wise=False,
                              reset_mode='subtraction', weight_bitwidth=4, )
    net = nn.Sequential(nn.Conv2d(2, 2, 3), nn.ReLU(), nn.Conv2d(2, 5, 1), nn.BatchNorm2d(5), nn.ReLU(), nn.Conv2d(5, 2, 3),
                        nn.ReLU(), nn.Linear(2*4*4, 2))
    net[0].weight.data.fill_(1.01)
    net[2].weight.data.fill_(1.01)
    net[5].weight.data.fill_(1.01)
    net[7].weight.data.fill_(1.01)
    net[0].weight.alpha = 0.5
    net[2].weight.alpha = 1
    net[5].weight.alpha = 10
    # net = nn.Sequential(nn.Conv2d(2, 2, 3), nn.ReLU())
    # net[0].weight.alpha=0.1
    snn_net = SNNTransformer(args, net, torch.device('cpu'))

    input = torch.rand([1, 2, 8, 8])

    input, _ = torch.broadcast_tensors(input, torch.zeros((steps,) + input.shape))
    input = input.permute(1, 2, 3, 4, 0)

    output = snn_net(input)
    print(output)

    print("==weight==", snn_net.layers1.conv1_s.Vth)
    # loader = [(input, torch.ones([4]))]
    # # t.inference_get_status(loader, 1)
    # snn = t.generate_snn_network()
    # replica_data = torch.cat([input for _ in range(args.timesteps)], 0)
    # data = SpikeTensor(replica_data, args.timesteps, scale_factor=1)
    # a_out = snn(data)
    # print(a_out.to_float())

    # t = SNNTransformer(args, net, torch.device('cpu'))

    # print("==weight==",snn.conv1.weight,v1snn.conv1.weight)
    # print("==weight==",snn.conv2.weight,v1snn.conv2.weight)
    # print("==bias==",snn.conv2.bias,v1snn.conv2.bias)
    # print("==vthr==",snn.conv2.Vthr,v1snn.conv2.Vthr)
