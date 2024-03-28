import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nn.utils.misc import log_metrics, zipdir

logger = logging.getLogger("tf")
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
   "momentum": lambda lr: torch.optim.SGD(momentum=0.9, lr=lr),
    "sgd": torch.optim.SGD
}


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.train_metrics = {}
        self.eval_metrics = {}

        # Extra functions to be run at train/valid/test time
        self.extra_train_fns = []
        self.extra_valid_fns = []
        self.extra_test_fns = []

    def run_extra_fns(self, type):
        if type == "train":
            extra_fns = self.extra_train_fns
        elif type == "valid":
            extra_fns = self.extra_valid_fns
        else:
            extra_fns = self.extra_test_fns

        for fn, args, kwargs in extra_fns:
            fn(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, output, target):
        raise NotImplementedError

    def get_data_loaders(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def initialize_model(self):
        raise NotImplementedError

    def build_optimizer(self, base_lr, optimizer="adam"):
        self.base_lr = base_lr
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=base_lr)

    def add_train_logger(self):
        raise NotImplementedError

    def train(self, epochs, eval_every_n_epochs, save_every_n_epochs, print_interval):
        self.add_train_logger()
        step = 0

        for ep in range(1, epochs + 1):
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                results = {}  # Define your training metrics here
                self.run_extra_fns("train")

                if step % print_interval == 0:
                    # Log training metrics
                    pass
                step += 1

            # Validation
            if ep % eval_every_n_epochs == 0:
                valid_metrics_results = self.evaluate(self.valid_loader)
                # Log validation metrics

            # Save model
            if ep % save_every_n_epochs == 0:
                torch.save(self.state_dict(), f"model_epoch_{ep}.pth")

        # Testing
        test_metrics_results = self.evaluate(self.test_loader)
        # Log testing metrics

    def evaluate(self, data_loader):
        self.eval()
        eval_metrics_results = {}

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.forward(inputs)
                # Compute evaluation metrics
                # Aggregate results

        self.train()  # Set the model back to training mode
        return eval_metrics_results