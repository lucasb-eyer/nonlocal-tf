# Non-local Neural Networks in TensorFlow

This is a TensorFlow (no Keras) implementation of the building blocks described in [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) by Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
It can simply be dropped into any existing model, and is compatible with TensorFlow's pre-trained ResNet models.

# Usage

The core code for the block is located in the `nonlocal_resnet_utils.py` file, you can just drop it into your code and use it as-is.
Usage is described in the heredoc comment and should be straightforward.

An example of a ResNet50, dramatically simplified from TensorFlow's "official" implementation, can be found in `nonlocal_resnet_v1_50_nl3.py`, again with usage described in the heredoc.

The nice thing is that simplified implementation is compatible with the [ImageNet pre-trained weights released by Google](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
So overall, to create a ResNet50 with one non-local block at stage 3, and loading pre-trained ImageNet weights, it's as simple as:

```python
import nonlocal_resnet_v1_50_nl3 as model
endpoints, body_prefix = model.endpoints(images, is_training=True)

# BEFORE DEFINING THE OPTIMIZER (because it creates new global variables):

model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

# IF LOADING PRE-TRAINED WEIGHTS (inside your with Session... block, in the first iteration):

saver = tf.train.Saver(model_variables)
saver.restore(sess, args.initial_checkpoint)
```
