# convnet
######Python classes for creating, training and testing 3D convolutional neural networks.

   The convolutional neural networks generated by CNN() are purely convolutional,
no pooling, dropout, or regression. It can be of any depth or width with any 
size of convolutional filters, provided the user's machine has the memory for 
exceptionally large networks. The networks are intended to produce affinity graphs
indicating the probability that adjacent pixels are part of the same object. These
affinity graphs can then be connected via a connected-components algorith to 
produce a segmented image.

-------------------------------------------------------------------------------
### Defining Networks
All user-definable parameters for building, training, and testing a network are
defined in [network.cfg](https://github.com/schurterb/convolutional_network/wiki/Network-Configuration). Parameters defining the structure of the network are
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
Parameters defining the [training regiment](https://github.com/schurterb/convolutional_network/wiki/Training-a-Network) for the network are listed under the
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

Epoch size is defined to be approximately the entire training set and is, thus, not directly configurable.

Setting the early_stop flag allows the trainer to automatically stop when the cost stops decreasing.

If a trainer folder (created by a previous training period) is available, 
the training can be restarted without any discontinuity.

A network from the `networks/` folder, such as testnet can be trained directly by calling:
```
python train -n testnet
```

A network can also be defined in any properly formated config file and trained using the -c flag:
```
python train -c /path/to/mynetwork.cfg
```
-------------------------------------------------------------------------------
### Using MALIS
MALIS is treated as training with the rand Index as a cost function. Thus, it 
can be enabled by setting the `cost_func` variable in the trainer to `'rand'`. 
Other cost function options include the Mean-Square Error (`'MSE'`) and the 
binary crossentropy function (`'class'`, since it is generally used for binary 
classification).

For more information about the rand Index and MALIS training, see this [paper](http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation).

-------------------------------------------------------------------------------
## Making Predictions
[Predictions can be made](https://github.com/schurterb/convolutional_network/wiki/Making-Predictions) on the test data referenced in [Testing Data] in the 
[network config file](https://github.com/schurterb/convolutional_network/wiki/Network-Configuration) with the following command
```
python predict -n testnet
```
for networks from the `networks/` folder.

Alternatively, any network defined by properly formated config file can be used
to make predictions on the [Testing Data].
```
python predict -c /path/to/mynetwork.cfg
```

The name of the file containing the prediction, as well as the folder it is 
stored in, are defined in the [Predicting] section of the config file. 
The prediction is automatically stored as a hdf5 file.

-------------------------------------------------------------------------------
## Testing Predictions

If a groundtruth affinity graph and corresponding segmentation is available 
(and provided in the config), the predicted affinity graph can be tested against
the groundtruth by calling the test script.
```
python test -n testnet
```
The results of the test are stored in errors_new.mat and a plot of those results
in errors_new.png. 

-------------------------------------------------------------------------------
## Viewing Results

The results of training and testing a network can be viewed for single network 
or for multiple networks simultaneously by calling ShowResults from the Ipython
console. 
```python
from showresults import ShowResults
res = ShowResults(network='conv-8.25.5')
res.learning('MSE', 10)
res.performance()
res.display('conv-8.25.5')
```

* ShowResults.learning() displays the learning curve associated with the specified
cost function (if it exists). It can also average over in intervals the designated
number of values across to produce a smoother plot.
* ShowResults.performance() displays the performance metrics measured by the test script.
* ShowResults.display() shows the predicted affinity graph along side the raw 
image and target affinities.

