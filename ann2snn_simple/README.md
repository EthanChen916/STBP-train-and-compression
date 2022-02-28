# A Simple Implementation of ANN2SNN

Spiking Neural Network (SNN) has been recognized as one of the next generation of neural networks. Conventionally, SNN can be convertedfrom a pre-trained ANN by only replacing the ReLU activation to spike activation while keeping the parameters intact. There, we implemented the conversion by **Spike-Norm** algorithm in **PyTorch**. However, this simple version **do** **not** support some operations such as bais, max-pooling, softmax, batch-normalization，etc. We will add them to our repository in the future.

## Usage

### 1. Example ann network conversion

To convert the example ann network:

```
python ann2snn.py --net_name example_net --ann_weight ./checkpoint/example_net_ann.pth --dataset CIFAR10
```

You can set the following parameters to adjust SNN conversion parameters:

- **--net_name**                # ann network name
- **--ann_weight**             # ann weight path
- **--dataset**                    # dataset name, we support CIFAR10 and MNIST now
- **--batch_size**               # batch size for finetuning
- **--test_batch_size**      # batch size for testing
- **--timesteps**                # number of timesteps each neuron can fire, default: 16
- **--finetune_epochs**    # epochs for finetuning

### 2. Custom ann network conversion

If you need to convert a custom network module called ```custom_net```. The network structure must be defined like the example networks in ```ann_model/example_net.py```.

We need to add the following to ```build_network.py```:

```**import** mynet
import mynet

def get_net_by_name(net_name):
    ...
    elif net_name=='mynet':
    net=mynet.MyNet()
    else:
    raise NotImplementedError
    
    return net
```

Then, you can convert your network as the above 1 done :clap:

## Accuracy

Note: we did't finetune the snn model on following results.

| Datasets |  ANN   |  SNN   |
| :------: | :----: | :----: |
|  MNIST   | 98.14% | 97.01% |
| CIFAR10  | 69.86% | 52.22% |



## Reference

+ [Abhronil Sengupta, Yuting Ye, Robert Wang, Chiao Liu, Kaushik Roy (2018). Going Deeper in Spiking Neural Networks: VGG and Residual Architectures](https://arxiv.org/abs/1802.02627)

  

