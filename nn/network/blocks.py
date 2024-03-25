import numpy as np
import tensorflow as tf
import torch.nn as pnn
import torchvision.transforms as tvtrans
""" Useful subnetwork components """


# def unet(inp, base_channels, out_channels, upsamp=True):
class UNet(pnn.Module):
    def __init__(self, in_features, hidden_dim, out_features, upsamp=True):
        # h = inp
        # h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        c1 = pnn.Conv2d(in_features, hidden_dim, kernel_size=3, padding="same")
        rel1 = pnn.ReLU()

        # h1 = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        c2 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        rel2 = pnn.ReLU()

        # h = tf.compat.v1.layers.max_pooling2d(h1, 2, 2)
        pool1 = pnn.MaxPool2d((2, 2))

        # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        c3 = pnn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding="same")
        rel3 = pnn.ReLU()

        # h2 = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        c4 = pnn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding="same")
        rel4 = pnn.ReLU()

        # h = tf.compat.v1.layers.max_pooling2d(h2, 2, 2)
        pool2 = pnn.MaxPool2d((2, 2))


        # h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        c5 = pnn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding="same")
        rel5 = pnn.ReLU()

        # h3 = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        c6 = pnn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding="same")
        rel6 = pnn.ReLU()

        # h = tf.compat.v1.layers.max_pooling2d(h3, 2, 2)
        pool3 = pnn.MaxPool2d((2, 2))

        # h = tf.compat.v1.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")
        c7 = pnn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding="same")
        rel7 = pnn.ReLU()

        # h4 = tf.compat.v1.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")

        if upsamp:
            c8 = pnn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, padding="same")
            rel8 = pnn.ReLU()

            # h = tf.image.resize(h4, h3.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
            up = tvtrans.Resize((hidden_dim*4, hidden_dim*4), interpolation=tvtrans.InterpolationMode.BILINEAR)
            # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
            h9 = pnn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding="same")
            rel9 = pnn.ReLU()
        else:
            # h = tf.compat.v1.layers.conv2d_transpose(h, base_channels*4, 3, 2, activation=None, padding="SAME")
            h9 = pnn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size=3, padding="same")
            rel9 = pnn.ReLU()

        # h = tf.concat([h, h3], axis=-1) TODO: Add this to forward()

        # TODO: how the fuck does this affect dimensionality
        # h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        h10 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding="same")
        rel10 = pnn.ReLU()

        # h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        h11 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding="same")
        rel11 = pnn.ReLU()

        if upsamp:
            h = tf.image.resize(h, h2.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
            h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
        else:
            h = tf.compat.v1.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")
        h = tf.concat([h, h2], axis=-1)
        h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        if upsamp:
            h = tf.image.resize(h, h1.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
            h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
        else:
            h = tf.compat.v1.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
        h = tf.concat([h, h1], axis=-1)
        h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")

        h = tf.compat.v1.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
        return h


def shallow_unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h1 = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.compat.v1.layers.max_pooling2d(h1, 2, 2)
    h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h2 = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.compat.v1.layers.max_pooling2d(h2, 2, 2)
    h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    #h = tf.concat([h, h3], axis=-1)
    #h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    #h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize(h, h2.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
        h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.compat.v1.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h2], axis=-1)
    h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize(h, h1.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
        h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.compat.v1.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h1], axis=-1)
    h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")

    h = tf.compat.v1.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
    return h


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = tf.ones([1,10])
    var = tf.compat.v1.layers.dense(var, 200, activation=tf.tanh)
    var = tf.compat.v1.layers.dense(var, np.prod(shape), activation=None)
    var = tf.reshape(var, shape)
    return var
