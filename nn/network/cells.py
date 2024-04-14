import numpy as np
import torch
import torch.nn as tnn


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
