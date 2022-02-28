from snn_layers import *


class SNNGenerate(nn.Module):
    def __init__(self, net):
        super(SNNGenerate, self).__init__()
        """
        The basic transformer to generate the snn for inference on cpu
        """

        self.sequential1 = nn.Sequential()
        self.sequential2 = nn.Sequential()

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
                    self.sequential1.add_module('conv'+str(conv_cnt)+'_s', tdLayer(childs[i], childs[i+1]))
                else:
                    self.sequential1.add_module('conv'+str(conv_cnt)+'_s', tdLayer(childs[i]))
                self.sequential1.add_module('spike'+str(LIFspike_cnt), LIFSpike())
                conv_cnt += 1
                LIFspike_cnt += 1

            elif isinstance(childs[i], nn.AvgPool2d):
                self.sequential1.add_module('pool'+str(pool_cnt)+'_s', tdLayer(childs[i]))
                self.sequential1.add_module('spike'+str(LIFspike_cnt), LIFSpike())
                pool_cnt += 1
                LIFspike_cnt += 1

            elif isinstance(childs[i], nn.Linear):
                if i < len(childs)-1 and isinstance(childs[i + 1], nn.BatchNorm2d):
                    self.sequential2.add_module('fc'+str(fc_cnt)+'_s', tdLayer(childs[i], childs[i+1]))
                else:
                    self.sequential2.add_module('fc'+str(fc_cnt)+'_s', tdLayer(childs[i]))
                self.sequential2.add_module('spike'+str(LIFspike_cnt), LIFSpike())
                fc_cnt += 1
                LIFspike_cnt += 1

    def forward(self, x):
        x = self.sequential1(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.sequential2(x)
        x = torch.sum(x, dim=2) / steps

        return x


def add_snn_op(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear) or isinstance(child, nn.AvgPool2d):
            module._modules[name] = nn.Sequential(tdLayer(child), LIFSpike())
        elif isinstance(child, nn.ReLU):
            module._modules[name] = nn.Sequential(tdLayer(child))

    return module


