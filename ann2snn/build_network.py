import ann_models.example_net as example_net


def get_net_by_name(net_name):
    if 'example_net' in net_name:
        net = getattr(example_net, f'ExampleNet')()
    elif 'mnist_net' in net_name:
        net = getattr(example_net, f'MNISTNet')()
    elif 'cifar10_net' in net_name:
        net = getattr(example_net, f'Cifar10Net')()
    else:
        raise NotImplementedError
    return net

