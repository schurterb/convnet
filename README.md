# convolutional_network
######Python classes for creating, training and testing 3D convolutional neural networks.

   The convolutional neural networks generated by CNN() are purely convolutional,
no pooling, dropout, or regression. It can be of any depth or width with any 
size of convolutional filters, provided the user's machine has the memory for 
exceptionally large networks. The networks are intended to produce affinity graphs
indicating the probability that adjacent pixels are part of the same object. These
affinity graphs can then be connected via a connected-components algorith to 
produce a segmented image.

-------------------------------------------------------------------------------
###Defining Networks
All user-definable parameters for building, training, and testing a network are
defined in network.ini. Parameters defining the structure of the network are
listed under the [Network] section.

```
[Network]
activation = relu
cost_func = MSE
load_weights = True
weights_folder = test/weights/

num_layers = 5
num_filters = 6
filter_size = 5
```

Current options for activation functions include
    * relu for Rectified Linear Units
    * tanh for Hyperbolic Tangent
    * sig for Sigmoid

Current options for cost functions include MSE and binary crossentropy.

If load_weights is True, then it does not matter what the last three parameters
are set to. CNN will determine the structure of the network based on the format
of the weight files.

-------------------------------------------------------------------------------
## Training Networks
Parameters defining the training regiment for the network are listed under the
[Training] section. By default, all training is done stochastically.

```
[Training]
use_malis = True
chunk_size = 10
learning_method = RMSprop
learning_rate = 1e-05
beta1 = 0.9
beta2 = 0.9
damping = 1.0e-8
batch_size = 100
num_epochs = 10
early_stop = False
trainer_folder = trainer/
log_folder =  
log_interval = 100
print_updates = True
```

Current options for learning methods include
* standardSGD with a constant learning rate (default)
* RMSprop - beta2 is the decay factor
* ADAM - beta2 is the decay for the variance term (as with RMSprop)
beta1 is the decay for the momentum term
* malis - see this [paper](http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation)

Epoch size is defined to be approximately the entire training set and is, thus, not directly configurable.

Setting the early_stop flag allows the trainer to automatically stop when the cost stops decreasing.

If a trainer folder (created by a previous training period) is available, 
the training can be restarted without any discontinuity.

A network from the networks folder, such as conv-8.25.5 can be trained directly by calling:
```
python train_network.py -n conv-8.25.5
```

A network can also be defined in any properly formated config file and trained using the -c flag:
```
python train_network.py -c /path/to/mynetwork.cfg
```
-------------------------------------------------------------------------------
### Using MALIS
MALIS training can be activated simply by setting the use_malis flag, but is 
substantially slower as the malisLoss calculation is not performed on the GPU. 
As such, it should only be used to perfect networks that cannot be further
improved using the Mean Square Error or Binary Cross-Entropy loss functions.

MALIS training is done with one contiguous block of training data at a time, 
which is assigned at the CPU level. The size of the block of training labels is
determined by the chunk_size parameter, which indicates the number of voxels 
across the subsection of data.

-------------------------------------------------------------------------------
## Making Predictions



-------------------------------------------------------------------------------
## Testing Predictions