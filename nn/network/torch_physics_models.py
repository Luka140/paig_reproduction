import torch
import torch.nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import sys
from nn.network.torch_base import BaseNet
import inspect

from nn.network.torch_base import BaseNet, OPTIMIZERS
from nn.network.cells import bouncing_ode_cell, spring_ode_cell, gravity_ode_cell
from nn.network.stn import stn
from nn.network.blocks import unet, shallow_unet, variable_from_network
from nn.utils.misc import log_metrics
from nn.utils.viz import gallery, gif
from nn.utils.math import sigmoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')

logger = logging.getLogger(__name__)

# Define the available optimizer options
OPTIMIZERS = {
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "sgd": optim.SGD
}

# Define the available cell types
CELLS = {
    "bouncing_ode_cell": bouncing_ode_cell,
    "spring_ode_cell": spring_ode_cell,
    "gravity_ode_cell": gravity_ode_cell,
    "lstm": torch.nn.LSTMCell
}

# total number of latent units for each datasets
# coord_units = num_objects*num_dimensions*2
COORD_UNITS = {
    "bouncing_balls": 8,
    "spring_color": 8,
    "spring_color_half": 8,
    "3bp_color": 12,
    "mnist_spring_color": 8
}

class PhysicsNet(BaseNet):
    def __init__(self,
                 task="",
                 recurrent_units=128,
                 lstm_layers=1,
                 cell_type="lstm",
                 seq_len=20,
                 input_steps=3,
                 pred_steps=5,
                 autoencoder_loss=0.0,
                 alt_vel=False,
                 color=False,
                 input_size=36*36,
                 encoder_type="conv_encoder",
                 decoder_type="conv_st_decoder"):

        super(PhysicsNet, self).__init__()

        assert task in COORD_UNITS
        self.task = task

        self.recurrent_units = recurrent_units
        self.lstm_layers = lstm_layers

        self.cell_type = cell_type
        self.cell = CELLS[self.cell_type]
        self.color = color
        self.conv_ch = 3 if color else 1
        self.input_size = input_size

        self.conv_input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch]
        self.input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch] # same as conv_input_shape, just here for backward compatibility

        self.encoder = {name: method for name, method in \
            inspect.getmembers(self, predicate=inspect.ismethod) if "encoder" in name
        }[encoder_type] 
        self.decoder = {name: method for name, method in \
            inspect.getmembers(self, predicate=inspect.ismethod) if "decoder" in name
        }[decoder_type]  

        self.output_shape = self.input_shape

        assert seq_len > input_steps + pred_steps
        assert input_steps >= 1
        assert pred_steps >= 1
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.extrap_steps = self.seq_len-self.input_steps-self.pred_steps

        self.alt_vel = alt_vel
        self.autoencoder_loss = autoencoder_loss

        self.coord_units = COORD_UNITS[self.task]
        self.n_objs = self.coord_units//4

        self.extra_valid_fns.append((self.visualize_sequence,[],{}))
        self.extra_test_fns.append((self.visualize_sequence,[],{}))

    def get_batch(self, batch_size, iterator):
        batch_x, _ = iterator.next_batch(batch_size)
        batch_len = batch_x.shape[1]
        feed_dict = {self.input: batch_x}
        return feed_dict, (batch_x, None)

    def compute_loss(self):
        recons_target = self.input[:,:self.input_steps+self.pred_steps]
        recons_loss = torch.square(recons_target-self.recons_out)
        

    def build_graph(self):
        pass

    def build_optimizer(self, base_lr, optimizer="rmsprop", anneal_lr=True):
        pass

    def conv_encoder(self, inp):
        pass

    def vel_encoder(self, inp):
        pass

    def conv_st_decoder(self, inp):
        pass

    def conv_feedforward(self):
        lstms = [torch.nn.LSTMCell(self.recurrent_units) for i in range(self.lstm_layers)]
        states = [lstm.zero_state(self.input.shape[0], dtype=torch.float32) for lstm in lstms]
        rollout_cell = self.cell(self.coord_units // 2)

        h = self.input[:, :self.input_steps+self.pred_steps].reshape(-1, *self.input_shape)
        enc_pos = self.encoder(h)

        recons_out = self.decoder(enc_pos)

        self.recons_out = recons_out.view(self.input.shape[0], self.input_steps + self.pred_steps, *self.input_shape)
        self.enc_pos = enc_pos.view(self.input.shape[0], self.input_steps + self.pred_steps, self.coord_units // 2)

        if self.input_steps > 1:
            vel = self.vel_encoder(self.enc_pos[:, :self.input_steps])
        else:
            vel = torch.zeros(self.input.shape[0], self.coord_units // 2)

        pos = self.enc_pos[:, self.input_steps - 1]
        output_seq = []
        pos_vel_seq = []
        pos_vel_seq.append(torch.cat([pos, vel], dim=1))

        # rollout ODE and decoder
        for t in range(self.pred_steps+self.extrap_steps):
            # Rollout
            pos, vel = rollout_cell(pos, vel)

            # Decoder
            out = self.decoder(pos)

            pos_vel_seq.append(torch.cat([pos, vel], axis=1))
            output_seq.append(out)

        self.network_vars = list(self.parameters)
        logger.info(self.network_vars)

        output_seq = torch.staack(output_seq)
        pos_vel_seq = torch.stack(pos_vel_seq)
        output_seq = output_seq.permute(1, 0, 2, 3, 4)
        pos_vel_seq = pos_vel_seq.permute(1, 0, 2)
        return output_seq


    def visualize_sequence(self):
        pass