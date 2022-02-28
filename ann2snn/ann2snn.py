import torch
import torch.nn as nn
from quantization import quantize_finetune
import datasets
from validation import validate_snn, validate_ann
import argparse
from build_network import get_net_by_name
from snn_transformer import SNNTransformer
import os
from snn_layers import config_snn_param


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', default='mnist_net', type=str, help='network name to train')
    parser.add_argument('--ann_weight', type=str, default='./checkpoint/mnist_net.pth',
                        help='the location of the trained weights')
    parser.add_argument('--dataset', default='MNIST',
                        type=str, help='the location of the dataset')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for dataset')
    parser.add_argument('--save_file', default="./out_snn.pth",
                        type=str, help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--timesteps', '-T', default=16, type=int)
    parser.add_argument('--tau', '-ta', type=int, default=1.0,
                        help='parameter leaky tau of LIF neuron')
    parser.add_argument('--finetune_epochs', '-e', default=50,
                        type=int, help='finetune epochs')
    parser.add_argument('--statistics_iters', default=30, type=int,
                        help='iterations for gather activation statistics')
    parser.add_argument('--device', default='0', type=str, help='choose cuda devices')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    config_snn_param(args)

    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # device = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Preparing the dataset
    train_loader, test_loader = datasets.get_dataset(args)

    # Build Model
    net = get_net_by_name(args.net_name)
    print(net)

    # Load weight
    if args.ann_weight:
        net.load_state_dict(torch.load(args.ann_weight)['model'], strict=False)
    net.to(device)

    # Validate
    criterion = nn.CrossEntropyLoss()
    validate_ann(net, test_loader, device, criterion)

    # Transform
    transformer = SNNTransformer(args, net, device)

    snn = transformer.generate_snn(train_loader, args.statistics_iters)
    snn = transformer.finetune_snn(snn, train_loader, criterion, args.finetune_epochs)

    # Test the results
    validate_snn(snn, test_loader, device, criterion, args.timesteps)

    # Save the network
    transformer.save_snn(snn, args.save_file)
