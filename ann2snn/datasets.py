import torch
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data


def get_dataset(args:argparse.Namespace):
    """ Preparing Datasets, args: 
        dataset (required): MNIST, cifar10/100
        dataset_root: str, default='./datasets'
        num_workers: int
        batch_size: int
        test_batch_size: int
    """
    dataset_root = getattr(args, 'dataset_root', './datasets')
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False} if args.device == 'cuda' else {}
    if args.dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.1307], std=[0.3081])
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_root, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.1307], std=[0.3081])
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(dataset_root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(dataset_root, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(dataset_root, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                              ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(dataset_root, train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                              ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader
