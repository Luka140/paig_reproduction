import numpy as np
import tensorflow as tf
import torch
import torch.nn as pnn
import torchvision.transforms as tvtrans
""" Useful subnetwork components """


class VelocityEncoder(pnn.Module):
    ...

# class STDecoder(pnn.Module):
#     def __init__(self, physics_input_shape, n_objs, logsig=1.0) -> None:
#         super(STDecoder, self).__init__()
#         self.logsigma = np.log(logsig)
#         self.sigma = np.exp(self.logsigma)
#         self.input_shape = physics_input_shape
#         self.n_objs = n_objs
#
#     def forward(self,inp):
#         batch_size = tf.shape(inp)[0]
#         tmpl_size = self.conv_input_shape[0] // 2


class ConvolutionalEncoder(pnn.Module):
    def __init__(self, in_features, hidden_dim, out_features, n_objects):
        """
        :param in_features: input shape [channels, height, width]

        """
        super().__init__()
        self.input_shape = in_features
        self.conv_ch = in_features[0]
        self.n_objs = n_objects
        self.shallow_unet = ShallowUNet(in_features, 8, n_objects, upsamp=True)
        self.unet = UNet(in_features, 16, n_objects)
        self.relu = pnn.ReLU()
        self.tanh = pnn.Tanh()
        self.softmax = pnn.Softmax(dim=1)
        self.pool1 = pnn.AvgPool2d((2, 2))
        # This input dim is kinda wack
        if self.input_shape[1] < 40:
            self.l1 = pnn.Linear(in_features[1] * in_features[1] * self.conv_ch, hidden_dim)
        else:
            self.l1 = pnn.Linear(in_features[1]//2 * in_features[1]//2 * self.conv_ch, hidden_dim)
        self.l2 = pnn.Linear(hidden_dim, hidden_dim)
        self.l3 = pnn.Linear(hidden_dim, out_features)

    def forward(self, inp):
        x = inp
        if self.input_shape[1] < 40:
            x = self.shallow_unet(x)
        else:
            x = self.unet(x)

        # Adds background
        x = torch.concat([x, torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])], dim=1)
        x = self.softmax(x)

        enc_masks = x
        masked_objs = [enc_masks[:, i:i+1, :, :] * inp for i in range(self.n_objs)]
        x = torch.concat(masked_objs, dim=0)

        if self.input_shape[1] < 40:
            # Reshape to batch * the number of entries in a single image (w * h * channels)
            x = torch.reshape(x, [x.shape[0], self.input_shape[1] * self.input_shape[2] * self.conv_ch])
        else:
            x = self.pool1(x)
            x = torch.flatten(x, start_dim=1)

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        x = torch.concat(torch.split(x, self.n_objs, 0), dim=1)
        x = self.tanh(x) * (self.input_shape[0] / 2) + (self.input_shape[0] / 2)
        return x, enc_masks, masked_objs


class UNet(pnn.Module):
    def __init__(self, in_features, hidden_dim, out_features, upsamp=True):
        # TODO THIS CURRENTLY ONLY WORKS FOR UPSAMP=TRUE BUT I DON'T SEE HOW THE ORIGINAL WAS SUPPOSED TO WORK FOR FALSE
        # TODO this option is also never used so maybe just remove it
        in_channels, height, width = in_features
        super(UNet, self).__init__()
        # h = inp
        # h = tf.compat.v1.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
        self.upsamp = upsamp
        self.c1 = pnn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding="same")
        self.rel1 = pnn.ReLU()
        self.c2 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.rel2 = pnn.ReLU()
        self.pool1 = pnn.MaxPool2d((2, 2))

        self.c3 = pnn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding="same")
        self.rel3 = pnn.ReLU()
        self.c4 = pnn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding="same")
        self.rel4 = pnn.ReLU()
        self.pool2 = pnn.MaxPool2d((2, 2))

        self.c5 = pnn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding="same")
        self.rel5 = pnn.ReLU()
        self.c6 = pnn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding="same")
        self.rel6 = pnn.ReLU()
        self.pool3 = pnn.MaxPool2d((2, 2))

        self.c7 = pnn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding="same")
        self.rel7 = pnn.ReLU()

        if upsamp:
            self.c8 = pnn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, padding="same")
            self.rel8 = pnn.ReLU()
            self.up = tvtrans.Resize((width//4, height//4), interpolation=tvtrans.InterpolationMode.BILINEAR)
            self.c9 = pnn.Conv2d(hidden_dim*8, hidden_dim*2, kernel_size=3, padding="same")
        else:
            self.c9 = pnn.Conv2d(hidden_dim*8, hidden_dim*2, kernel_size=3, padding="same")

        self.c10 = pnn.Conv2d(hidden_dim * 6, hidden_dim * 4, kernel_size=3, padding="same")
        self.rel10 = pnn.ReLU()
        self.c11 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding="same")
        self.rel11 = pnn.ReLU()

        if upsamp:
            self.up2 = tvtrans.Resize((width//2, height//2), interpolation=tvtrans.InterpolationMode.BILINEAR)
            self.c12 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding="same")

        else:
            self.c12 = pnn.ConvTranspose2d(hidden_dim * 4, hidden_dim*2, kernel_size=3, output_padding="same")

        self.c13 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding="same")
        self.rel13 = pnn.ReLU()
        self.c14 = pnn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding="same")
        self.rel14 = pnn.ReLU()

        if upsamp:
            self.up3 = tvtrans.Resize((width, height), interpolation=tvtrans.InterpolationMode.BILINEAR)
            self.c15 = pnn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding="same")

        else:
            self.c15 = pnn.ConvTranspose2d(hidden_dim * 2, hidden_dim*2, kernel_size=3, stride=2, output_padding="same")

        self.c16 = pnn.Conv2d(hidden_dim*3, hidden_dim, kernel_size=3, padding="same")
        self.rel16 = pnn.ReLU()
        self.c17 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.rel17 = pnn.ReLU()
        self.c18 = pnn.Conv2d(hidden_dim, out_features, kernel_size=1, padding="same")

    def forward(self, x):
        """
        Input should be of shape (batch, channels, height, width)
        """
        x = self.c1(x)
        x = self.rel1(x)
        x = self.c2(x)
        x1 = self.rel2(x)

        x = self.pool1(x1)

        x = self.c3(x)
        x = self.rel3(x)

        x = self.c4(x)
        x2 = self.rel4(x)

        x = self.pool2(x2)

        x = self.c5(x)
        x = self.rel5(x)

        x = self.c6(x)
        x3 = self.rel6(x)

        x = self.pool3(x3)

        x = self.c7(x)
        x = self.rel7(x)

        if self.upsamp:
            x = self.c8(x)
            x = self.rel8(x)

            x = self.up(x)

        x = self.c9(x)

        x = torch.concat((x, x3), dim=1)

        x = self.c10(x)
        x = self.rel10(x)

        x = self.c11(x)
        x = self.rel11(x)

        if self.upsamp:
            x = self.up2(x)
        x = self.c12(x)
        x = torch.concat((x, x2), dim=1)

        x = self.c13(x)
        x = self.rel13(x)

        x = self.rel14(self.c14(x))

        if self.upsamp:
            x = self.up3(x)
        x = self.c15(x)

        x = torch.concat((x, x1), dim=1)

        x = self.rel16(self.c16(x))
        x = self.rel17(self.c17(x))
        x = self.c18(x)
        return x


class ShallowUNet(pnn.Module):
    def __init__(self, in_features, hidden_dim, out_features, upsamp=True):
        super(ShallowUNet, self).__init__()
        in_channels, height, width = in_features
        self.upsamp = upsamp

        self.c1 = pnn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding="same")
        self.ReLU = pnn.ReLU()

        self.c2 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.pool1 = pnn.MaxPool2d((2, 2))

        self.c3 = pnn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding="same")
        self.c4 = pnn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding="same")
        self.pool2 = pnn.MaxPool2d((2, 2))

        self.c5 = pnn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding="same")
        self.c6 = pnn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding="same")

        if upsamp:
            self.up1 = tvtrans.Resize((width//2, height//2), interpolation=tvtrans.InterpolationMode.BILINEAR)
            self.c7 = pnn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding="same")
        else:
            raise NotImplementedError("Using ShallowUNet without upsamp is not implemented yet")

        self.c8 = pnn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding="same")
        self.c9 = pnn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding="same")

        if upsamp:
            self.up2 = tvtrans.Resize((width, height), interpolation=tvtrans.InterpolationMode.BILINEAR)
            self.c10 = pnn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding="same")
        else:
            raise NotImplementedError("Using ShallowUNet without upsamp is not implemented yet")

        self.c11 = pnn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=3, padding="same")
        self.c12 = pnn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same")
        self.c13 = pnn.Conv2d(hidden_dim, out_features, kernel_size=1, padding="same")

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x1 = self.ReLU(self.c2(x))
        x = self.pool1(x1)
        x = self.ReLU(self.c3(x))
        x2 = self.ReLU(self.c4(x))
        x = self.pool2(x2)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))

        if self.upsamp:
            x = self.up1(x)
            x = self.c7(x)
        else:
            raise NotImplementedError("Using ShallowUNet without upsamp is not implemented yet")
        x = torch.concat([x, x2], dim=1)
        x = self.ReLU(self.c8(x))
        x = self.ReLU(self.c9(x))

        if self.upsamp:
            x = self.up2(x)
            x = self.c10(x)
        else:
            raise NotImplementedError("Using ShallowUNet without upsamp is not implemented yet")

        x = torch.concat([x, x1], dim=1)

        x = self.ReLU(self.c11(x))
        x = self.ReLU(self.c12(x))
        x = self.ReLU(self.c13(x))
        return x


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = tf.ones([1,10])
    var = tf.compat.v1.layers.dense(var, 200, activation=tf.tanh)
    var = tf.compat.v1.layers.dense(var, np.prod(shape), activation=None)
    var = tf.reshape(var, shape)
    return var


if __name__ == "__main__":
    n_objs = 5
    # conv_input_shape = [3, 32, 32]
    conv_input_shape = [3, 40, 40]

    enc = ConvolutionalEncoder(conv_input_shape, 200, 2, n_objs)
    h = torch.Tensor(1000, *conv_input_shape)

    x, _, __ = enc(h)
    print(x.shape)