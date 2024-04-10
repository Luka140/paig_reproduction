import inspect
import logging
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as pnn

from nn.network.base import OPTIMIZERS, BaseNetTorch
from nn.network.blocks import ConvolutionalEncoder, VelocityEncoder, VariableFromNetwork
from nn.network.cells import bouncing_ode_cell, spring_ode_cell, gravity_ode_cell
from nn.network.stn import stn
from nn.utils.viz import gallery, gif

plt.switch_backend('agg')

logger = logging.getLogger("tf")


CELLS = {
    "bouncing_ode_cell": bouncing_ode_cell,
    "spring_ode_cell": spring_ode_cell,
    "gravity_ode_cell": gravity_ode_cell,
    "lstm": pnn.LSTMCell
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


class PhysicsNet(BaseNetTorch):
    def __init__(self,
                 task="",
                 recurrent_units=128,
                 lstm_layers=1,
                 cell_type="",
                 seq_len=20,
                 input_steps=3,
                 pred_steps=5,
                 autoencoder_loss=0.0,
                 alt_vel=False,
                 color=False,
                 input_size=36*36,
                 encoder_type="",
                 decoder_type="conv_st_decoder",
                 device=torch.device("cpu")):

        super(PhysicsNet, self).__init__()
        self.device = device
        assert task in COORD_UNITS
        self.task = task

        # Only used when using black-box dynamics (baselines)
        self.recurrent_units = recurrent_units
        self.lstm_layers = lstm_layers

        self.cell_type = cell_type
        self.cell = CELLS[self.cell_type]
        self.color = color
        self.conv_ch = 3 if color else 1
        self.input_size = input_size

        self.conv_input_shape = [self.conv_ch]+[int(np.sqrt(input_size))]*2 # Swapped order of channels and img dimensions
        self.input_shape = [self.conv_ch] + [int(np.sqrt(input_size))]*2 # same as conv_input_shape, just here for backward compatibility

        # self.encoder = {name: method for name, method in \
        #     inspect.getmembers(self, predicate=inspect.ismethod) if "encoder" in name
        # }[encoder_type] 
        self.decoder = {name: method for name, method in \
            inspect.getmembers(self, predicate=inspect.ismethod) if "decoder" in name
        }[decoder_type]

        self.output_shape = self.conv_input_shape

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

        # TODO commented out to avoid visualize sequence
        self.extra_valid_fns.append((self.visualize_sequence,[],{}))
        self.extra_test_fns.append((self.visualize_sequence,[],{}))

        ############
        tmpl_size = self.conv_input_shape[1]//2
        self.var_net_content = VariableFromNetwork([self.n_objs, self.conv_ch, tmpl_size, tmpl_size])
        self.var_net_background = VariableFromNetwork([1,*self.input_shape])
        self.var_net_template = VariableFromNetwork([self.n_objs, 1, tmpl_size, tmpl_size])
        # self.dtype = torch.float32
        self.encoder = ConvolutionalEncoder(self.conv_input_shape, 200, 2, self.n_objs, self.device)
        self.velocity_encoder = VelocityEncoder(self.alt_vel, self.input_steps, self.n_objs, self.coord_units, self.device)

        test_inp_dim = 4
        self.test_network = pnn.Sequential(pnn.Linear(test_inp_dim, 200), pnn.ReLU(), pnn.Linear(200,200), pnn.ReLU(), pnn.Linear(200, 3*32*32), pnn.Sigmoid())

        lstm_hidden_dim = 100
        # self.lstm = pnn.LSTM((self.input_steps+self.pred_steps)*self.conv_ch * self.input_shape[1]**2, lstm_hidden_dim, batch_first=True, num_layers=self.lstm_layers, dropout=0.05)
        # self.lstm_linear = pnn.Linear(lstm_hidden_dim, np.prod(self.input_shape))
        # lstms = [tf.compat.v1.nn.rnn_cell.LSTMCell(self.recurrent_units) for i in range(self.lstm_layers)]

        # for decoder 
        self.log_sig = 1.

    def get_batch(self, batch_size, iterator):
        batch_x, _ = iterator.next_batch(batch_size)
        batch_len = batch_x.shape[1]
        feed_dict = {"input": batch_x}
        return feed_dict, (batch_x, None)

    def compute_loss(self):

        # loss = torch.nn.CrossEntropyLoss()

        # Compute reconstruction loss
        recons_target = self.input[:,:self.input_steps+self.pred_steps]
        # output = loss(self.recons_out, recons_target)


        recons_loss = torch.square(recons_target-self.recons_out)
        recons_loss = torch.sum(recons_loss, dim=[2,3,4])

        self.recons_loss = torch.mean(recons_loss)

        target = self.input[:,self.input_steps:]
        loss = torch.square(target-self.output)
        loss = torch.sum(loss, dim=[2,3,4])

        # Compute prediction losses. pred_loss is used for training, extrap_loss is used for evaluation
        self.pred_loss = torch.mean(loss[:,:self.pred_steps])
        self.extrap_loss = torch.mean(loss[:,self.pred_steps:])

        train_loss = self.pred_loss
        if self.autoencoder_loss > 0.0:
            train_loss += self.autoencoder_loss*self.recons_loss

        eval_losses = [self.pred_loss, self.extrap_loss, self.recons_loss]
        return train_loss, eval_losses

    # def build_graph(self):
    #     tf.compat.v1.disable_eager_execution()
    #     # self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.seq_len]+self.input_shape)
    #     # TODO Placeholder ====================================================================================
    #     batch_size = 100
    #     self.input = torch.randn(batch_size, self.seq_len, *self.input_shape)
    #     # TODO Placeholder ====================================================================================
    #     self.output = self.conv_feedforward(self.input)
    #
    #     self.train_loss, self.eval_losses = self.compute_loss()
    #     self.train_metrics["train_loss"] = self.train_loss
    #     self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
    #     self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
    #     self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
    #     self.loss = self.train_loss

    def build_optimizer(self, base_lr, optimizer="rmsprop", anneal_lr=True):
        # Uncomment lines below to have different learning rates for physics and vision components
        # TODO dit moet gefixt worden via base branch met nieuwe FLAGs
        # base_lr = 1e-2
        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        self.lr = base_lr
        self.optimizer = OPTIMIZERS[optimizer](self.parameters(), self.lr)
        #self.dyn_optimizer = OPTIMIZERS[optimizer](1e-3)


        # gvs = self.optimizer.compute_gradients(self.loss, var_list=tf.compat.v1.trainable_variables())
        # gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs if grad is not None]
        # self.train_op = self.optimizer.apply_gradients(gvs)

        # self.train_op = self.optimizer.apply_gradients([gv for gv in gvs if "cell" not in gv[1].name])
        # if len([gv for gv in gvs if "cell" in gv[1].name]) > 0:
        #     self.dyn_train_op = self.dyn_optimizer.apply_gradients([gv for gv in gvs if "cell" in gv[1].name])
        #     self.train_op = tf.group(self.train_op, self.dyn_train_op)

    def conv_st_decoder(self, inp):
        batch_size = inp.shape[0]
        tmpl_size = self.conv_input_shape[1]//2

        # This parameter can be played with.
        # Setting it to log(2.0) makes the attention window half the size, which might make
        # it easier for the model to discover objects in some cases.
        # I haven't found this to make a consistent difference though. 
        # logsigma = tf.compat.v1.get_variable("logsigma", shape=[], initializer=tf.compat.v1.constant_initializer(np.log(1.0)), trainable=True)
        logsigma = np.log(self.log_sig)
        sigma = np.exp(logsigma)
        
        #TODO i think this is supposed to be the background - whats up with the tile though - why +5?
        
        template = self.var_net_template()
        self.template = template
        template = torch.tile(template, [1,3,1,1])+5
        
        # Non background objects
        contents = self.var_net_content()
        self.contents = contents 
        contents = pnn.Sigmoid()(contents)
        joint = torch.concat([template, contents], dim=1)

        out_temp_cont = []
        for loc, join in zip(torch.split(inp, inp.shape[1]//self.n_objs, -1), torch.split(joint, joint.shape[0]//self.n_objs, 0)):
            theta0 = torch.tile(torch.tensor([sigma],device=self.device), [inp.shape[0]])
            theta1 = torch.tile(torch.tensor([0.0],device=self.device), [inp.shape[0]])
            theta2 = (self.conv_input_shape[1]/2-loc[:,0])/tmpl_size*sigma
            theta3 = torch.tile(torch.tensor([0.0],device=self.device), [inp.shape[0]])
            theta4 = torch.tile(torch.tensor([sigma],device=self.device), [inp.shape[0]])
            theta5 = (self.conv_input_shape[1]/2-loc[:,1])/tmpl_size*sigma
            theta = torch.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)
            out_join = stn(torch.tile(join, [inp.shape[0], 1, 1, 1]), theta, self.conv_input_shape[1:])
            out_temp_cont.append(torch.split(out_join, out_join.shape[1]//2, 1))

        background_content = self.var_net_background()
        self.background_content = pnn.Sigmoid()(background_content)
        background_content = torch.tile(self.background_content, [batch_size, 1, 1, 1])
        contents = [p[1] for p in out_temp_cont]
        contents.append(background_content)
        self.transf_contents = contents

        background_mask = torch.ones_like(out_temp_cont[0][0])
        masks = torch.stack([p[0]-5 for p in out_temp_cont]+[background_mask], dim=1)
        masks = pnn.Softmax(dim=1)(masks)
        masks = torch.unbind(masks, dim=1)
        self.transf_masks = masks

        out = sum([m*c for m, c in zip(masks, contents)])
        return out

    def forward(self, input):
        return self.conv_feedforward(input)

    def conv_feedforward(self, inp):
        self.input = inp
        # lstms = [tf.compat.v1.nn.rnn_cell.LSTMCell(self.recurrent_units) for i in range(self.lstm_layers)]
        # states = [lstm.zero_state(tf.shape(self.input)[0], dtype=tf.float32) for lstm in lstms]
        rollout_cell = self.cell(self.coord_units//2, self.coord_units//2)

        h = self.input[:,:self.input_steps+self.pred_steps].reshape(self.input.shape[0],self.input_steps+self.pred_steps,-1)
        # print(h.shape)
        # h, _ = self.lstm(h)
        # print(h.shape)
        # h = self.lstm_linear(h)
        # print(h.shape)
        # Encode all the input and train frames
        # sequence length and batch get flattened together in dim0
        h = torch.reshape(h, [-1]+self.input_shape)

        enc_pos, self.enc_masks, self.masked_objs = self.encoder(h)

        # decode the input and pred frames
        recons_out = self.decoder(enc_pos)
        # print(enc_pos.shape, recons_out.shape)
        # recons_out = self.test_network(enc_pos)
        # recons_out = torch.reshape(recons_out, (-1, self.input_size))

        self.recons_out = torch.reshape(recons_out, [self.input.shape[0], self.input_steps+self.pred_steps]+self.input_shape)
        self.enc_pos = torch.reshape(enc_pos, [self.input.shape[0], self.input_steps+self.pred_steps, self.coord_units//2])

        if self.input_steps > 1:
            vel = self.velocity_encoder(self.enc_pos[:,:self.input_steps])
        else:
            vel = torch.zeros([self.input.shape[0], self.coord_units//2])

        pos = self.enc_pos[:,self.input_steps-1]
        output_seq = []
        pos_vel_seq = []
        pos_vel_seq.append(torch.cat([pos, vel], dim=1))

        # rollout ODE and decoder
        for t in range(self.pred_steps+self.extrap_steps):
            # rollout
            pos, vel = rollout_cell(pos, vel)

            # decode
            out = self.decoder(pos)

            pos_vel_seq.append(torch.cat([pos, vel], dim=1))
            output_seq.append(out)

        output_seq = torch.stack(output_seq)
        pos_vel_seq = torch.stack(pos_vel_seq)
        output_seq = torch.permute(output_seq, (1,0,2,3,4))
        self.pos_vel_seq = torch.permute(pos_vel_seq, (1,0,2))
        return output_seq

    def visualize_sequence(self):
        batch_size = self.batch_size
        feed_dict, (batch_x, _) = self.get_batch(batch_size, self.test_iterator)
        fetches = [self.output, self.recons_out]
        if hasattr(self, 'pos_vel_seq'):
            fetches.append(self.pos_vel_seq)

        res = fetches
        output_seq = res[0].detach().cpu().numpy()
        recons_seq = res[1].detach().cpu().numpy()
        if hasattr(self, 'pos_vel_seq'):
            pos_vel_seq = res[2]
        output_seq = np.concatenate([batch_x[:,:self.input_steps], output_seq], axis=1)
        recons_seq = np.concatenate([recons_seq, np.zeros((batch_size, self.extrap_steps)+recons_seq.shape[2:])], axis=1)

        # Plot a grid with prediction sequences
        for i in range(batch_x.shape[0]):
            if hasattr(self, 'pos_vel_seq'):
               if i == 0 or i == 1:
                   logger.info(pos_vel_seq[i])

            to_concat = [output_seq[i],batch_x[i],recons_seq[i]]
            total_seq = np.concatenate(to_concat, axis=0) 

            total_seq = total_seq.reshape([total_seq.shape[0], *self.input_shape[1:], self.conv_ch])
                                           # self.input_shape[0],
                                           # self.input_shape[1], self.conv_ch])

            result = gallery(total_seq, ncols=batch_x.shape[1])

            norm = plt.Normalize(0.0, 1.0)

            figsize = (result.shape[1]//self.input_shape[1], result.shape[0]//self.input_shape[0])
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir, "example%d.jpg"%i))
            plt.close()
        # Make a gif from the sequences
        bordered_output_seq = 0.5*np.ones([batch_size, self.seq_len, 
                                          self.conv_input_shape[1]+2, self.conv_input_shape[2]+2, 3])
        bordered_batch_x = 0.5*np.ones([batch_size, self.seq_len, 
                                          self.conv_input_shape[1]+2, self.conv_input_shape[2]+2, 3])
        output_seq = output_seq.reshape([batch_size, self.seq_len]+self.input_shape[1:] + [self.conv_ch])
        batch_x = batch_x.reshape([batch_size, self.seq_len]+self.input_shape[1:] + [self.conv_ch])
        bordered_output_seq[:,:,1:-1,1:-1] = output_seq
        bordered_batch_x[:,:,1:-1,1:-1] = batch_x
        output_seq = bordered_output_seq
        batch_x = bordered_batch_x
        output_seq = np.concatenate(np.split(output_seq, batch_size, 0), axis=-2).squeeze()
        batch_x = np.concatenate(np.split(batch_x, batch_size, 0), axis=-2).squeeze()
        frames = np.concatenate([output_seq, batch_x], axis=1)

        gif(os.path.join(self.save_dir, "animation%d.gif"%i), 
            frames*255, fps=7, scale=3)

        # Save extra tensors for visualization
        fetches = {"contents": self.contents.cpu(),
                   "templates": self.template.cpu(),
                   "background_content": self.background_content.cpu(),
                   "transf_contents": [cont.cpu() for cont in self.transf_contents],
                   "transf_masks": [mask.cpu() for mask in self.transf_masks],
                   "enc_masks": self.enc_masks.cpu(),
                   "masked_objs": [masked_obj.cpu() for masked_obj in self.masked_objs]}
        results = fetches
        # results = self.sess.run(fetches, feed_dict=feed_dict)
        np.savez_compressed(os.path.join(self.save_dir, "extra_outputs.npz"), **results)
        contents = torch.transpose(results["contents"], 1, -1)
        templates = torch.transpose(results["templates"], 1, -1)
        contents = 1/(1+np.exp(-contents))
        templates = 1/(1+np.exp(-(templates-5)))
        if self.conv_ch == 1:
            contents = np.tile(contents, [1,1,1,3])
        templates = np.tile(templates, [1,1,1,3])
        total_seq = np.concatenate([contents, templates], axis=0)
        result = gallery(total_seq, ncols=self.n_objs)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "templates.jpg"))
        plt.close("all")
        # logger.info([(v.name, self.sess.run(v)) for v in tf.compat.v1.trainable_variables() if "ode_cell" in v.name or "sigma" in v.name])

