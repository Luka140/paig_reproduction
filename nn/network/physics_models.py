import inspect
import logging
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as pnn

from nn.network.base import BaseNet, OPTIMIZERS
from nn.network.blocks import UNet, ShallowUNet, variable_from_network, ConvolutionalEncoder, VelocityEncoder
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

class PhysicsNet(BaseNet):
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
                 encoder_type="conv_encoder",
                 decoder_type="conv_st_decoder"):

        super(PhysicsNet, self).__init__()

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

        self.extra_valid_fns.append((self.visualize_sequence,[],{}))
        self.extra_test_fns.append((self.visualize_sequence,[],{}))

        ############

        self.encoder = ConvolutionalEncoder(self.conv_input_shape, 200, 2, self.n_objs)
        # TODO check inputs to velocity encoder
        self.velocity_encoder = VelocityEncoder(self.alt_vel, self.input_steps, self.n_objs, self.coord_units)
        
        # for decoder 
        self.log_sig = 1.0

    def get_batch(self, batch_size, iterator):
        batch_x, _ = iterator.next_batch(batch_size)
        batch_len = batch_x.shape[1]
        feed_dict = {self.input: batch_x}
        return feed_dict, (batch_x, None)

    def compute_loss(self):

        # loss = torch.nn.CrossEntropyLoss()

        # Compute reconstruction loss
        recons_target = self.input[:,:self.input_steps+self.pred_steps]
        # output = loss(self.recons_out, recons_target)


        recons_loss = torch.square(recons_target-self.recons_out)
        #recons_ce_loss = -(recons_target*tf.log(self.recons_out+1e-7) + (1.0-recons_target)*tf.log(1.0-self.recons_out+1e-7))

        recons_loss = torch.sum(recons_loss, dim=[2,3,4])

        self.recons_loss = torch.mean(recons_loss)

        target = self.input[:,self.input_steps:]
        #ce_loss = -(target*tf.log(self.output+1e-7) + (1.0-target)*tf.log(1.0-self.output+1e-7))
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

    def build_graph(self):
        tf.compat.v1.disable_eager_execution()
        # self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.seq_len]+self.input_shape)
        # TODO Placeholder ====================================================================================
        batch_size = 100
        self.input = torch.randn(batch_size, self.seq_len, *self.input_shape)
        # TODO Placeholder ====================================================================================
        self.output = self.conv_feedforward()

        self.train_loss, self.eval_losses = self.compute_loss()
        self.train_metrics["train_loss"] = self.train_loss
        self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
        self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
        self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
        self.loss = self.train_loss

    def build_optimizer(self, base_lr, optimizer="rmsprop", anneal_lr=True):
        # Uncomment lines below to have different learning rates for physics and vision components
        # TODO dit moet gefixt worden via base branch met nieuwe FLAGs
        # base_lr = 1e-2
        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        # self.lr = tf.Variable(base_lr, trainable=False, name="base_lr")
        self.lr = base_lr
        self.optimizer = OPTIMIZERS[optimizer](self.parameters(), self.lr)
        #self.dyn_optimizer = OPTIMIZERS[optimizer](1e-3)


        # print(list(self.parameters()))
        self.loss.backward()
        gvs = self.optimizer.step()

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
        
        # TODO maybe swap the channel dim to spot 1 later in the torch.randn calls
        # template = variable_from_network([self.n_objs, tmpl_size, tmpl_size, 1])
        template = torch.randn([self.n_objs, 1, tmpl_size, tmpl_size])
        self.template = template
        template = torch.tile(template, [1,3,1,1])+5
        
        # Non background objects
        # contents = variable_from_network([self.n_objs, tmpl_size, tmpl_size, self.conv_ch])
        contents = torch.randn([self.n_objs, self.conv_ch, tmpl_size, tmpl_size])
        self.contents = contents 
        contents = pnn.Sigmoid()(contents)
        joint = torch.concat([template, contents], dim=1)

        out_temp_cont = []
        for loc, join in zip(torch.split(inp, inp.shape[1]//self.n_objs, -1), torch.split(joint, joint.shape[0]//self.n_objs, 0)):
            theta0 = torch.tile(torch.Tensor([sigma]), [inp.shape[0]])
            theta1 = torch.tile(torch.Tensor([0.0]), [inp.shape[0]])
            theta2 = (self.conv_input_shape[1]/2-loc[:,0])/tmpl_size*sigma
            theta3 = torch.tile(torch.Tensor([0.0]), [inp.shape[0]])
            theta4 = torch.tile(torch.Tensor([sigma]), [inp.shape[0]])
            theta5 = (self.conv_input_shape[1]/2-loc[:,1])/tmpl_size*sigma
            theta = torch.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)
            # print("conv_ch", self.conv_ch, "loc:", loc.shape, "join", join.shape)
            out_join = stn(torch.tile(join, [inp.shape[0], 1, 1, 1]), theta, self.conv_input_shape[1:])
            # print("outjoin shape", out_join.shape)
            out_temp_cont.append(torch.split(out_join, out_join.shape[1]//2, 1))

        # background_content = variable_from_network([1]+self.input_shape)
        background_content = torch.randn(1,*self.input_shape)
        self.background_content = pnn.Sigmoid()(background_content)
        background_content = torch.tile(self.background_content, [batch_size, 1, 1, 1])
        contents = [p[1] for p in out_temp_cont]
        contents.append(background_content)
        self.transf_contents = contents

        background_mask = torch.ones_like(out_temp_cont[0][0])
        # print("BG mask", background_mask.shape)
        masks = torch.stack([p[0]-5 for p in out_temp_cont]+[background_mask], dim=1)
        # print("masks", masks.shape)
        masks = pnn.Softmax(dim=1)(masks)
        masks = torch.unbind(masks, dim=1)
        self.transf_masks = masks

        out = sum([m*c for m, c in zip(masks, contents)])
        # print("out", out.shape)
        return out

    def conv_feedforward(self):

        # TODO: what do the lines below do? - this is still tensorflow code but somehow doesnt throw an error
        lstms = [tf.compat.v1.nn.rnn_cell.LSTMCell(self.recurrent_units) for i in range(self.lstm_layers)]
        states = [lstm.zero_state(tf.shape(self.input)[0], dtype=tf.float32) for lstm in lstms]
        rollout_cell = self.cell(self.coord_units//2, self.coord_units//2)

        # TODO: PLACEHOLDER
        # batch_size = 1000
        # self.input = torch.randn(batch_size, 12, 3, 32, 32)

        # Encode all the input and train frames
        # sequence length and batch get flattened together in dim0
        h = torch.reshape(self.input[:,:self.input_steps+self.pred_steps], [-1]+self.input_shape)
        # TODO placeholder atm
        # h = torch.randn([batch_]+self.input_shape)
        enc_pos, self.enc_masks, self.enc_objs = self.encoder(h)

        # decode the input and pred frames
        recons_out = self.decoder(enc_pos)

        # self.recons_out = tf.reshape(recons_out,
        #                              [tf.shape(self.input)[0], self.input_steps+self.pred_steps]+self.input_shape)
        self.recons_out = torch.reshape(recons_out, [self.input.shape[0], self.input_steps+self.pred_steps]+self.input_shape)
        # self.enc_pos = tf.reshape(enc_pos,
        #                           [tf.shape(self.input)[0], self.input_steps+self.pred_steps, self.coord_units//2])

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

        # current_scope = tf.compat.v1.get_default_graph().get_name_scope()
        # self.network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        #                                       scope=current_scope)
        # logger.info(self.network_vars)

        output_seq = torch.stack(output_seq)
        pos_vel_seq = torch.stack(pos_vel_seq)
        output_seq = torch.permute(output_seq, (1,0,2,3,4))
        self.pos_vel_seq = torch.permute(pos_vel_seq, (1,0,2))
        return output_seq

    def visualize_sequence(self):
        batch_size = 5

        feed_dict, (batch_x, _) = self.get_batch(batch_size, self.test_iterator)
        fetches = [self.output, self.recons_out]
        if hasattr(self, 'pos_vel_seq'):
            fetches.append(self.pos_vel_seq)

        res = fetches
        output_seq = res[0].detach().numpy()
        recons_seq = res[1].detach().numpy()
        if hasattr(self, 'pos_vel_seq'):
            pos_vel_seq = res[2]
        # print("\n\n\n", batch_x.shape, batch_x[:,:self.input_steps].shape, output_seq.shape)
        output_seq = np.concatenate([batch_x[:,:self.input_steps], output_seq], axis=1)
        recons_seq = np.concatenate([recons_seq, np.zeros((batch_size, self.extrap_steps)+recons_seq.shape[2:])], axis=1)

        # Plot a grid with prediction sequences
        for i in range(batch_x.shape[0]):
            #if hasattr(self, 'pos_vel_seq'):
            #    if i == 0 or i == 1:
            #        logger.info(pos_vel_seq[i])

            to_concat = [output_seq[i],batch_x[i],recons_seq[i]]
            total_seq = np.concatenate(to_concat, axis=0) 

            total_seq = total_seq.reshape([total_seq.shape[0], 
                                           self.input_shape[0], 
                                           self.input_shape[1], self.conv_ch])

            result = gallery(total_seq, ncols=batch_x.shape[1])

            norm = plt.Normalize(0.0, 1.0)

            figsize = (result.shape[1]//self.input_shape[1], result.shape[0]//self.input_shape[0])
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir, "example%d.jpg"%i))

        # Make a gif from the sequences
        bordered_output_seq = 0.5*np.ones([batch_size, self.seq_len, 
                                          self.conv_input_shape[0]+2, self.conv_input_shape[1]+2, 3])
        bordered_batch_x = 0.5*np.ones([batch_size, self.seq_len, 
                                          self.conv_input_shape[0]+2, self.conv_input_shape[1]+2, 3])
        output_seq = output_seq.reshape([batch_size, self.seq_len]+self.input_shape)
        batch_x = batch_x.reshape([batch_size, self.seq_len]+self.input_shape)
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
        fetches = {"contents": self.contents,
                   "templates": self.template,
                   "background_content": self.background_content,
                   "transf_contents": self.transf_contents,
                   "transf_masks": self.transf_masks,
                   "enc_masks": self.enc_masks,
                   "masked_objs": self.masked_objs}
        results = self.sess.run(fetches, feed_dict=feed_dict)
        np.savez_compressed(os.path.join(self.save_dir, "extra_outputs.npz"), **results)
        contents = results["contents"]
        templates = results["templates"]
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

        logger.info([(v.name, self.sess.run(v)) for v in tf.compat.v1.trainable_variables() if "ode_cell" in v.name or "sigma" in v.name])

