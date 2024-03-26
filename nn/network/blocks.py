import numpy as np
import tensorflow as tf
import torch
import torch.nn as pnn
import torchvision.transforms as tvtrans
""" Useful subnetwork components """


# def unet(inp, base_channels, out_channels, upsamp=True):
class UNet(pnn.Module):
    def __init__(self, in_features, hidden_dim, out_features, upsamp=True):

        super(UNet, self).__init__()
        # h = inp
        # h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        self.upsamp = upsamp
        self.c1 = pnn.Conv2d(in_features, hidden_dim, kernel_size=3, padding="same")
        self.rel1 = pnn.ReLU()

        # h1 = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        self.c2 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.rel2 = pnn.ReLU()

        # h = tf.compat.v1.layers.max_pooling2d(h1, 2, 2)
        self.pool1 = pnn.MaxPool2d((2, 2))

        # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        self.c3 = pnn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding="same")
        self.rel3 = pnn.ReLU()

        # h2 = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        self.c4 = pnn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding="same")
        self.rel4 = pnn.ReLU()

        # h = tf.compat.v1.layers.max_pooling2d(h2, 2, 2)
        self.pool2 = pnn.MaxPool2d((2, 2))


        # h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        self.c5 = pnn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding="same")
        self.rel5 = pnn.ReLU()

        # h3 = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        self.c6 = pnn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding="same")
        self.rel6 = pnn.ReLU()

        # h = tf.compat.v1.layers.max_pooling2d(h3, 2, 2)
        self.pool3 = pnn.MaxPool2d((2, 2))

        # h = tf.compat.v1.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")
        self.c7 = pnn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding="same")
        self.rel7 = pnn.ReLU()

        # h4 = tf.compat.v1.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")
        # Put this inside the if statement
        if upsamp:
            self.c8 = pnn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, padding="same")
            self.rel8 = pnn.ReLU()

            # h = tf.image.resize(h4, h3.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
            self.up = tvtrans.Resize((hidden_dim*4, hidden_dim*4), interpolation=tvtrans.InterpolationMode.BILINEAR)
            # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
            self.c9 = pnn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding="same")
        else:
            # h = tf.compat.v1.layers.conv2d_transpose(h, base_channels*4, 3, 2, activation=None, padding="SAME")
            self.c9 = pnn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size=3, padding="same")


        # h = tf.concat([h, h3], axis=-1) TODO: Add this to forward()

        # TODO: how the fuck does this affect dimensionality
        # h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        self.c10 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding="same")
        self.rel10 = pnn.ReLU()

        # h = tf.compat.v1.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
        self.c11 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding="same")
        self.rel11 = pnn.ReLU()

        if upsamp:
            # h = tf.image.resize(h, h2.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
            self.up2 = tvtrans.Resize((hidden_dim*4, hidden_dim*4), interpolation=tvtrans.InterpolationMode.BILINEAR)

            # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
            self.c12 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding="same")

        else:
            # h = tf.compat.v1.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")
            self.c12 = pnn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, output_padding="same")


        # h = tf.concat([h, h2], axis=-1) # TODO ADD CONCAT TO FORWARD

        # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        self.c13 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding="same")
        self.rel13 = pnn.ReLU()
        # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
        self.c14 = pnn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding="same")
        self.rel14 = pnn.ReLU()

        if upsamp:
            # h = tf.image.resize(h, h1.get_shape()[1:3], method=tf.image.ResizeMethod.BILINEAR)
            self.up3 = tvtrans.Resize((hidden_dim*4, hidden_dim*4), interpolation=tvtrans.InterpolationMode.BILINEAR)

            # h = tf.compat.v1.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
            self.c15 = pnn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding="same")

        else:
            # h = tf.compat.v1.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
            self.c15 = pnn.ConvTranspose2d(hidden_dim * 4, hidden_dim, kernel_size=3, stride=2, output_padding="same")

        # h = tf.concat([h, h1], axis=-1) # TODO: Add to forward
        # h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        self.c16 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.rel16 = pnn.ReLU()


        # h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        self.c17 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.rel17 = pnn.ReLU()
        # h = tf.compat.v1.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
        self.c18 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding="same")

    def forward(self, x):
        """
        Input should be of shape (batch, channels, height, width)
        """
        x = self.c1(x)
        x = self.rel1(x)

        x = self.c2(x)
        x = self.rel2(x)
        print(f"\n\n\n{x.shape}\n\n\n")
        x = self.pool1(x)

        x = self.c3(x)
        x = self.rel3(x)

        x = self.c4(x)
        x2 = self.rel4(x)

        x = self.pool2(x2)

        x = self.c5(x)
        x = self.rel5(x)

        x = self.c6(x)
        x = self.rel6(x)

        x3 = self.pool3(x)

        x = self.c7(x3)
        x = self.rel7(x)

        if self.upsamp:
            x = self.c8(x)
            x = self.rel8(x)
            x = self.up(x)
        x = self.c9(x)
        x = torch.concat((x, x3), dim=-1)

        x = self.c10(x)
        x = self.rel10(x)

        x = self.c11(x)
        x = self.rel11(x)

        if self.upsamp:
            x = self.up2(x)
        x = self.c12(x)
        x = torch.concat((x, x2), dim=-1)

        x = self.c13(x)
        x = self.rel13(x)

        x = self.rel14(self.c14(x))

        if self.upsamp:
            x = self.up3(x)
        x = self.c15(x)

        x = self.rel16(self.c16(x))

        x = self.rel17(self.c17(x))

        x = self.rel18(x)
        return x





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
