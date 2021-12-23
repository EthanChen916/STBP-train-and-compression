import models.example_net as example_net
import re
import torch
import models.mynet as mynet


def get_net_by_name(net_name):
    try:
        version=int(re.findall('(\d+)$',net_name)[0])
    except:
        version=0
    if 'example_small' in net_name:
        net=example_net.ExampleSmall()
    elif 'example_net' in net_name:
        net=getattr(example_net,f'ExampleNet{version}')()
    elif 'mynet' in net_name:
        net = mynet.MyNet()
    else:
        raise NotImplementedError
    return net


if __name__ == '__main__':
    model = get_net_by_name('test_net')
    print(model)
