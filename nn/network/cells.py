import numpy as np
import torch 
import torch
import torch.nn as tnn
import tensorflow as tf


class ode_cell(tnn.RNNCell):
    def __init__(self, input_size, hidden_size):
        super(ode_cell, self).__init__(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    @property
    def state_size(self):
        return self.hidden_size, self.hidden_size

    def zero_state(self, batch_size, dtype):
        x_0 = torch.zeros(batch_size, self.hidden_size, dtype=dtype)
        v_0 = torch.zeros(batch_size, self.hidden_size, dtype=dtype)
        return x_0, v_0

class spring_ode_cell(ode_cell):
    """ Assumes there are 2 objects """

    def __init__(self, input_size, hidden_size):
        super(spring_ode_cell, self).__init__(input_size, hidden_size)
        self.dt = tnn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.k = tnn.Parameter(torch.tensor(np.log(1.0)), requires_grad=True)
        self.equil = tnn.Parameter(torch.tensor(np.log(1.0)), requires_grad=True)

    def forward(self, poss, vels):
        poss = torch.split(poss, 1, dim=1)
        vels = torch.split(vels, 1, dim=1)

        for i in range(5):
            norm = torch.sqrt(torch.abs(torch.sum((poss[0] - poss[1])**2, dim=-1, keepdim=True)))
            direction = (poss[0] - poss[1]) / (norm + 1e-4)
            F = torch.exp(self.k) * (norm - 2 * torch.exp(self.equil)) * direction
            vels = list(vels)
            vels[0] = vels[0] - self.dt / 5 * F
            vels[1] = vels[1] + self.dt / 5 * F
            vels = tuple(vels)

            poss = list(poss)
            poss[0] = poss[0] + self.dt / 5 * vels[0]
            poss[1] = poss[1] + self.dt / 5 * vels[1]
            poss = tuple(poss)

        poss = torch.cat(poss, dim=1)
        vels = torch.cat(vels, dim=1)
        return poss, vels

class bouncing_ode_cell(ode_cell):
    """ Assumes there are 2 objects """

    def __init__(self, input_size, hidden_size):
        super(bouncing_ode_cell, self).__init__(input_size, hidden_size)
        self.dt = tnn.Parameter(torch.tensor(0.3), requires_grad=False)

    def forward(self, poss, vels):
        poss = torch.split(poss, 1, dim=1)
        vels = torch.split(vels, 1, dim=1)

        for i in range(5):
            poss = list(poss)
            poss[0] = poss[0] + self.dt / 5 * vels[0]
            poss[1] = poss[1] + self.dt / 5 * vels[1]
            poss = tuple(poss)

            for j in range(2):
                # Compute wall collisions. Image boundaries are hard-coded.
                vels = list(vels)
                vels[j] = torch.where(poss[j] + 2 > 32, -vels[j], vels[j])
                vels[j] = torch.where(0.0 > poss[j] - 2, -vels[j], vels[j])
                poss = list(poss)
                poss[j] = torch.where(poss[j] + 2 > 32, 32 - (poss[j] + 2 - 32) - 2, poss[j])
                poss[j] = torch.where(0.0 > poss[j] - 2, -(poss[j] - 2) + 2, poss[j])
                vels = tuple(vels)
                poss = tuple(poss)

        poss = torch.cat(poss, dim=1)
        vels = torch.cat(vels, dim=1)
        return poss, vels

class gravity_ode_cell(ode_cell):
    """ Assumes there are 3 objects """

    def __init__(self, input_size, hidden_size):
        super(gravity_ode_cell, self).__init__(input_size, hidden_size)
        self.dt = tnn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.g = tnn.Parameter(torch.tensor(np.log(1.0)), requires_grad=True)
        self.m = tnn.Parameter(torch.tensor(np.log(1.0)), requires_grad=False)
        self.A = torch.exp(self.g) * torch.exp(2 * self.m)

    def forward(self, poss, vels):
        for i in range(5):
            vecs = [poss[:, 0:2] - poss[:, 2:4], poss[:, 2:4] - poss[:, 4:6], poss[:, 4:6] - poss[:, 0:2]]
            norms = [torch.sqrt(torch.clamp(torch.sum(vec ** 2, dim=-1, keepdim=True), min=1e-1, max=1e5)) for vec in vecs]
            F = [(vec / torch.pow(torch.clamp(norm, min=1, max=170), 3)) for vec, norm in zip(vecs, norms)]
            F = [F[0] - F[2], F[1] - F[0], F[2] - F[1]]
            F = [-self.A * f for f in F]
            F = torch.cat(F, dim=1)
            vels = vels + self.dt / 5 * F
            poss = poss + self.dt / 5 * vels
        return poss, vels


# import numpy as np
# import tensorflow as tf

# import os
# from importlib.metadata import version
# if int(version("tensorflow")[2:4]) >= 16:
#     import os;os.environ["TF_USE_LEGACY_KERAS"]="1"

# class ode_cell(tf.compat.v1.nn.rnn_cell.BasicRNNCell):
#     @property
#     def state_size(self):
#         return self._num_units, self._num_units

#     def zero_state(self, batch_size, dtype):
#         x_0 = tf.zeros([batch_size, self._num_units], dtype=dtype)
#         v_0 = tf.zeros([batch_size, self._num_units], dtype=dtype)
#         return x_0, v_0


# class bouncing_ode_cell(ode_cell):
#     """ Assumes there are 2 objects """

#     def build(self, inputs_shape):
#         print("Hey, I'm in bouncing_ode_cell, function build")
#         if inputs_shape[-1] is None:
#             raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
#                                 % str(inputs_shape))

#         input_depth = inputs_shape[-1]
#         h_depth = self._num_units
#         assert h_depth == input_depth

#         self.dt = self.add_variable("dt_x", shape=[], initializer=tf.compat.v1.constant_initializer(0.3), trainable=False)
#         self.built = True

#     def call(self, poss, vels):
#         print("Hey, I'm in bouncing_ode_cell, function call")
#         poss = tf.split(poss, 2, 1)
#         vels = tf.split(vels, 2, 1)
#         for i in range(5):
#             poss[0] = poss[0] + self.dt/5*vels[0]
#             poss[1] = poss[1] + self.dt/5*vels[1]

#             for j in range(2):
#                 # Compute wall collisions. Image boundaries are hard-coded.
#                 vels[j] = tf.compat.v1.where(tf.greater(poss[j]+2, 32), -vels[j], vels[j])
#                 vels[j] = tf.compat.v1.where(tf.greater(0.0, poss[j]-2), -vels[j], vels[j])
#                 poss[j] = tf.compat.v1.where(tf.greater(poss[j]+2, 32), 32-(poss[j]+2-32)-2, poss[j])  
#                 poss[j] = tf.compat.v1.where(tf.greater(0.0, poss[j]-2), -(poss[j]-2)+2, poss[j]) 

#         poss = tf.concat(poss, axis=1)
#         vels = tf.concat(vels, axis=1)
#         return poss, vels


# class spring_ode_cell(ode_cell):
#     """ Assumes there are 2 objects """

#     def build(self, inputs_shape):
#         if inputs_shape[-1] is None:
#           raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
#                            % str(inputs_shape))

#         input_depth = inputs_shape[-1]
#         h_depth = self._num_units
#         assert h_depth == input_depth

#         self.dt = self.add_variable("dt_x", shape=[], initializer=tf.compat.v1.constant_initializer(0.3), trainable=False)
#         self.k = self.add_variable("log_k", shape=[], initializer=tf.compat.v1.constant_initializer(np.log(1.0)), trainable=True)
#         self.equil = self.add_variable("log_l", shape=[], initializer=tf.compat.v1.constant_initializer(np.log(1.0)), trainable=True)
#         self.built = True

#     def call(self, poss, vels):
#         poss = tf.split(poss, 2, 1)
#         vels = tf.split(vels, 2, 1)
#         for i in range(5):
#             norm = tf.sqrt(tf.abs(tf.reduce_sum(tf.square(poss[0]-poss[1]), axis=-1, keepdims=True)))
#             direction = (poss[0]-poss[1])/(norm+1e-4)
#             F = tf.exp(self.k)*(norm-2*tf.exp(self.equil))*direction
#             vels[0] = vels[0] - self.dt/5*F
#             vels[1] = vels[1] + self.dt/5*F

#             poss[0] = poss[0] + self.dt/5*vels[0]
#             poss[1] = poss[1] + self.dt/5*vels[1]
 
#         poss = tf.concat(poss, axis=1)
#         vels = tf.concat(vels, axis=1)
#         return poss, vels


# class gravity_ode_cell(ode_cell):
#     """ Assumes there are 3 objects """

#     def build(self, inputs_shape):
#         if inputs_shape[-1] is None:
#           raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
#                            % str(inputs_shape))

#         input_depth = inputs_shape[-1]
#         h_depth = self._num_units
#         assert h_depth == input_depth

#         self.dt = self.add_variable("dt_x", shape=[], initializer=tf.compat.v1.constant_initializer(0.5), trainable=False)
#         self.g = self.add_variable("log_g", shape=[], initializer=tf.compat.v1.constant_initializer(np.log(1.0)), trainable=True)
#         self.m = self.add_variable("log_m", shape=[], initializer=tf.compat.v1.constant_initializer(np.log(1.0)), trainable=False)
#         self.A = tf.exp(self.g)*tf.exp(2*self.m)
#         self.built = True

#     def call(self, poss, vels):
#         for i in range(5):
#             vecs = [poss[:,0:2]-poss[:,2:4], poss[:,2:4]-poss[:,4:6], poss[:,4:6]-poss[:,0:2]]
#             norms = [tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(vec), axis=-1, keepdims=True), 1e-1, 1e5)) for vec in vecs]
#             F = [vec/tf.pow(tf.clip_by_value(norm, 1, 170), 3) for vec, norm in zip(vecs, norms)]
#             F = [F[0]-F[2], F[1]-F[0], F[2]-F[1]]
#             F = [-self.A*f for f in F]
#             F = tf.concat(F, axis=1)
#             vels = vels + self.dt/5*F
#             poss = poss + self.dt/5*vels
 
#         poss = tf.concat(poss, axis=1)
#         vels = tf.concat(vels, axis=1)
#         return poss, vels
