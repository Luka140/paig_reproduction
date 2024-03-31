import os
import sys
import logging
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nn.utils.misc import log_metrics, zipdir

logger = logging.getLogger("tf")
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
   "momentum": lambda params, lr: torch.optim.SGD(params, momentum=0.9, lr=lr),
    "sgd": torch.optim.SGD
}


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.train_metrics = {}
        self.eval_metrics = {}

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

    def feedforward(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError
    
    def build_graph(self):
        raise NotImplementedError

    def get_data(self, data_iterators):
        self.train_iterator, self.valid_iterator, self.test_iterator = data_iterators

    def get_iterator(self, type):
        if type == "train":
            eval_iterator = self.train_iterator 
        elif type == "valid":
            eval_iterator = self.valid_iterator 
        elif type == "test":
            eval_iterator = self.test_iterator
        return eval_iterator
    
    def initialize_graph(self,
                         save_dir,
                         use_ckpt,
                         ckpt_dir=""):
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            if use_ckpt:
                restore = True
                if ckpt_dir:
                    restore_dir = ckpt_dir
                else:
                    restore_dir = save_dir
            else:
                logger.info("Folder exists, deleting...")
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
                restore = False
        else:
            os.makedirs(save_dir)
            if use_ckpt:
                restore = True
                restore_dir = ckpt_dir 
            else:
                restore = False
        restore = False # TODO: REMOVE THIS LINE LATER, SOMEHOW RESTORE IS SET TO TRUE CAUSING ERRORS
        if restore:
            self.load_state_dict(torch.load(os.path.join(restore_dir, "model.pth")))
        

    def build_optimizer(self, base_lr, optimizer="adam", anneal_lr=True):
        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        self.optimizer = OPTIMIZERS[optimizer](self.parameters(), lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def get_batch(self, batch_size, iterator):
        batch_x, batch_y = iterator.next_batch(batch_size)
        if batch_y is None:
            return batch_x, None
        else:
            return batch_x, batch_y

    def add_train_logger(self):
        log_path = os.path.join(self.save_dir, "log.txt")
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def train(self,
              epochs, 
              batch_size,
              save_every_n_epochs,
              eval_every_n_epochs,
              print_interval,
              debug=False):

        self.add_train_logger()
        zipdir(root_path, self.save_dir) 
        logger.info("\n".join(sys.argv))

        # Run validation once before starting training
        if not debug and epochs > 0:
            valid_metrics_results = self.eval(batch_size, type='valid')
            log_metrics(logger, "valid - epoch=%s"%0, valid_metrics_results)

        for ep in range(1, epochs+1):
            if self.anneal_lr:
                if ep == int(0.75*epochs):
                    self.scheduler.step()
            while self.train_iterator.epochs_completed < ep:
                batch_x, batch_y = self.get_batch(batch_size, self.train_iterator)
                self.optimizer.zero_grad()
                output = self.forward(batch_x)
                loss = self.compute_loss(output, batch_y)
                loss
    
    def eval(self,
         batch_size,
         type='valid'):

        eval_metrics_results = {k: [] for k in self.eval_metrics.keys()}
        eval_outputs = {"input": [], "output": []}

        eval_iterator = self.get_iterator(type)
        eval_iterator.reset_epoch()

        with torch.no_grad():
            while eval_iterator.get_epoch() < 1:
                if eval_iterator.X.shape[0] < 100:
                    batch_size = eval_iterator.X.shape[0]
                batch_x, batch_y = self.get_batch(batch_size, eval_iterator)
                output = self.forward(batch_x)

                # Assuming that self.eval_metrics is a dictionary containing evaluation metrics functions
                for k, metric_fn in self.eval_metrics.items():
                    eval_metrics_results[k].append(metric_fn(output, batch_y).item())

                eval_outputs["input"].append(batch_x)
                eval_outputs["output"].append(output)

        # Compute mean of evaluation metrics
        eval_metrics_results = {k: np.mean(v) for k, v in eval_metrics_results.items()}

        # Save evaluation outputs
        np.savez_compressed(os.path.join(self.save_dir, "outputs.npz"),
                            input=np.concatenate(eval_outputs["input"], axis=0),
                            output=np.concatenate(eval_outputs["output"], axis=0))

        self.run_extra_fns(type)

        return eval_metrics_results
    