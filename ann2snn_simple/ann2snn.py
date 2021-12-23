import torch
import numpy as np
from quantization import quantize_finetune
import datasets
from validation import validate_snn, validate_ann
import argparse
from build_network import get_net_by_name
from build_criterion import get_criterion_by_name
from snn_transformer import SNNTransformer
import os


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', default='example_net', type=str, help='network name to train')
    parser.add_argument('--ann_weight', type=str, default='./checkpoint/example_net_ann.pth',
                        help='the location of the trained weights')
    parser.add_argument('--dataset', default='CIFAR10',
                        type=str, help='the location of the dataset')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for dataset')
    parser.add_argument('--save_file', default="./out_snn.pth",
                        type=str, help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--timesteps', '-T', default=16, type=int)
    parser.add_argument('--weight_bitwidth', default=4,
                        type=int, help='weight quantization bitwidth')
    parser.add_argument('--criterion', default='cross_entropy',
                        type=str, help='name of criterion used in finetune')
    parser.add_argument('--finetune_lr', default=0.005,
                        type=float, help='finetune learning rate')
    parser.add_argument('--quantization_channel_wise', '-qcw',
                        action='store_true', help='quantize in each channel')
    parser.add_argument('--finetune_epochs', '-e', default=5,
                        type=int, help='finetune epochs')
    parser.add_argument('--finetune_wd', default=5e-4,
                        type=float, help='finetune weight decay')
    parser.add_argument('--finetune_momentum', default=0.9,
                        type=float, help='finetune momentum')
    parser.add_argument('--statistics_iters', default=30, type=int,
                        help='iterations for gather activation statistics')
    parser.add_argument('--device', default='0,1,2,3', type=str, help='choose cuda devices')

    args = parser.parse_args()
    args.activation_bitwidth = np.log2(args.timesteps)
    torch.cuda.empty_cache()
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
        net.load_state_dict(torch.load(args.ann_weight), strict=False)
    net.to(device)

    # Validate
    criterion = get_criterion_by_name(args.criterion)
    net_top1, net_loss = validate_ann(net, test_loader, device, criterion)

    # Quantization and Finetune
    qnet = quantize_finetune(net, train_loader, criterion, device, args)
    qnet_top1, qnet_loss = validate_ann(qnet, test_loader, device, criterion)

    # Transform
    transformer = SNNTransformer(args, qnet, device)

    snn = transformer.generate_snn(train_loader, args.statistics_iters)
    
    # Test the results
    validate_snn(snn, test_loader, device, criterion, args.timesteps,)

    # Save the SNN
    torch.save(snn, args.save_file)
    torch.save(snn.state_dict(), args.save_file+'.weight')
    print("Save the SNN in {}".format(args.save_file))
