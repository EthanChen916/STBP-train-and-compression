# A Simple Implementation of ANN2SNN

Spiking Neural Network (SNN) has been recognized as one of the next generation of neural networks. Conventionally, SNN can be convertedfrom a pre-trained ANN by only replacing the ReLU activation to spike activation while keeping the parameters intact. There, we implemented the conversion by **Spike-Norm** algorithm in **PyTorch** and use STBP algorithm for finetuning. However, this simple version **do** **not** support some operations such as bais, max-pooling, softmax, batch-normalization，etc. We will add them to our repository in the future.

## Usage

### 1. Convert example ann network

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
- **--finetune_epochs**    # epochs for finetuning by STBP

### 2. Convert custom ann network

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

Then, you can convert your network as the part 1 done :clap:

### 3.Load snn models after conversion

When you finish converting and want to load the snn model for inferencing, you need to add the imports required:

​		```from load_ann2snn_model import load_snn```

```load_snn``` will attempt to load a PyTorch module list into a model.

Then, you should definite the snn network as the following:

```
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()

        self.conv1_s = tdLayer(nn.Conv2d(3, 8, 5, bias=False))
        self.spike1 = LIFSpike()
        self.pool1_s = tdLayer(nn.AvgPool2d(2))
        self.spike2 = LIFSpike()
        self.conv2_s = tdLayer(nn.Conv2d(8, 16, 3, bias=False))
        self.spike3 = LIFSpike()
        self.pool2_s = tdLayer(nn.AvgPool2d(2))
        self.spike4 = LIFSpike()
        self.conv3_s = tdLayer(nn.Conv2d(16, 32, 3, bias=False))
        self.spike5 = LIFSpike()
        self.fc1_s = tdLayer(nn.Linear(4*4*32, 10, bias=False))
        self.spike6 = LIFSpike()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike1(x)
        x = self.pool1_s(x)
        x = self.spike2(x)
        x = self.conv2_s(x)
        x = self.spike3(x)
        x = self.pool2_s(x)
        x = self.spike4(x)
        x = self.conv3_s(x)
        x = self.spike5(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike6(x)
        out = torch.sum(x, dim=2) / steps
        
        return out
```

## Accuracy

| Datasets |  ANN   |  SNN   |
| :------: | :----: | :----: |
|  MNIST   | 98.14% | 97.01% |
| CIFAR10  | 69.86% | 52.22% |

Note: We did't finetune the snn model.

## Reference

+ [Abhronil Sengupta, Yuting Ye, Robert Wang, Chiao Liu, Kaushik Roy (2018). Going Deeper in Spiking Neural Networks: VGG and Residual Architectures](https://arxiv.org/abs/1802.02627)

  

