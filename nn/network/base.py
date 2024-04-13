import os
import sys
import shutil
import logging
import numpy as np
import torch

from nn.utils.misc import log_metrics, zipdir
import torch

logger = logging.getLogger("tf")
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

OPTIMIZERS = {
    "adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
   "momentum": lambda params, lr: torch.optim.SGD(params, momentum=0.9, lr=lr),
    "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr)
}


class BaseNetTorch(torch.nn.Module):
    def __init__(self):
        super(BaseNetTorch, self).__init__()
        self.train_metrics = {}
        self.eval_metrics = {}

        # Extra functions to be ran at train/valid/test time
        # that can be defined by the children
        # Should have the format:
        #   self.extra_valid_fns = [
        #      (valid_fn1, args, kwargs),
        #       ...
        #   ]
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

    def conv_feedforward(self, inp: torch.Tensor):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def get_data(self, data_iterators):
        self.train_iterator, self.valid_iterator, self.test_iterator = data_iterators

    def get_batch(self, batch_size, iterator):
        batch_x, batch_y = iterator.next_batch(batch_size)
        if batch_y is None:
            feed_dict = {"input": batch_x}
        else:
            feed_dict = {"input": batch_x, "target": batch_y}
        return feed_dict, (batch_x, batch_y)

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

        if restore:
            print(f"Loading model from: {restore_dir+'/model.ckpt'}")
            self.load_state_dict(torch.load(os.path.join(restore_dir, "model.ckpt")))

    def get_iterator(self, type):
        if type == "train":
            eval_iterator = self.train_iterator
        elif type == "valid":
            eval_iterator = self.valid_iterator
        elif type == "test":
            eval_iterator = self.test_iterator
        return eval_iterator

    def add_train_logger(self):
        log_path = os.path.join(self.save_dir, "log.txt")
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def train_model(self,
              epochs,
              batch_size,
              save_every_n_epochs,
              eval_every_n_epochs,
              print_interval,
              debug=False):

        self.train()

        self.batch_size = batch_size
        self.add_train_logger()
        zipdir(root_path, self.save_dir)
        logger.info("\n".join(sys.argv))

        step = 0

        # Run validation once before starting training
        if not debug and epochs > 0:
            valid_metrics_results = self.eval_performance(batch_size, type='valid')
            log_metrics(logger, "valid - epoch=%s" % 0, valid_metrics_results)

        for ep in range(1, epochs + 1):
            if self.anneal_lr:
                if ep == int(0.75 * epochs):
                    self.lr = self.lr / 5
            while self.train_iterator.epochs_completed < ep:
                feed_dict, _ = self.get_batch(batch_size, self.train_iterator)
                # results, _ = self.sess.run(
                # [self.train_metrics, self.train_op], feed_dict=feed_dict)
                # torch.inference_mode(False)

                inp = torch.tensor(feed_dict["input"], requires_grad=True, device=self.device)
                result_sequence = self.forward(inp)
                # self.optimizer.zero_grad(set_to_none=True)
                self.train_loss, self.eval_losses = self.compute_loss()

                self.train_metrics["train_loss"] = self.train_loss
                self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
                self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
                self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
                self.loss = self.train_loss
                self.optimizer.zero_grad(set_to_none=True)
                self.loss.backward()
                self.optimizer.step()

                self.run_extra_fns("train") # DUnno what this does

                # self.optimizer.zero_grad(set_to_none=True)

                if step % print_interval == 0:
                    log_metrics(logger, "train - iter=%s" % step, self.train_metrics)
                step += 1

            if ep % eval_every_n_epochs == 0:
                print("eval running")
                valid_metrics_results = self.eval_performance(batch_size, type='valid')
                log_metrics(logger, "valid - epoch=%s" % ep, valid_metrics_results)

            if ep % save_every_n_epochs == 0:
                print("saving")
                torch.save(self.state_dict(), os.path.join(self.save_dir, "model.ckpt"))

        test_metrics_results = self.eval_performance(batch_size, type='test')
        log_metrics(logger, "test - epoch=%s" % epochs, test_metrics_results)

    def eval_performance(self,
             batch_size,
             type='valid'):

        self.eval()
        # torch.inference_mode(True)
        with torch.no_grad():
            # self.eval_metrics["train_loss"] = torch.Tensor([0])
            self.eval_metrics["eval_pred_loss"] = torch.tensor([0], device=self.device)
            self.eval_metrics["eval_extrap_loss"] = torch.tensor([0], device=self.device)
            self.eval_metrics["eval_recons_loss"] = torch.tensor([0], device=self.device)
            eval_metrics_results = {k: [] for k in self.eval_metrics.keys()}
            eval_outputs = {"input": [], "output": []}

            eval_iterator = self.get_iterator(type)
            eval_iterator.reset_epoch()

            while eval_iterator.get_epoch() < 1:
                if eval_iterator.X.shape[0] < 100:
                    batch_size = eval_iterator.X.shape[0]
                feed_dict, _ = self.get_batch(batch_size, eval_iterator)
                # fetches = {k: v for k, v in self.eval_metrics.items()}
                # fetches["output"] = self.output
                # fetches["input"] = self.input
                inp = torch.tensor(feed_dict["input"], requires_grad=False, device=self.device)
                self.output = self.conv_feedforward(inp)
                self.train_loss, self.eval_losses = self.compute_loss()
                self.train_metrics["train_loss"] = self.train_loss
                self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
                self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
                self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
                self.loss = self.train_loss
                # print("\n\n\n\n", self.)

                for k in self.eval_metrics.keys():
                    eval_metrics_results[k].append(self.eval_metrics[k])
                eval_outputs["input"].append(feed_dict["input"])
                eval_outputs["output"].append(self.eval_losses)
                # print("\n\n\n\n\neval_outputs:", eval_outputs["output"])

            eval_metrics_results = {k: np.mean([i.detach().cpu().numpy() for i in v], axis=0) for k, v in
                                    eval_metrics_results.items()}
            np.savez_compressed(os.path.join(self.save_dir, "outputs.npz"),
                                input=np.concatenate(eval_outputs["input"], axis=0),
                                output=np.array([[output_loss.detach().cpu().numpy() for output_loss in output] for output in eval_outputs["output"]]))

            self.run_extra_fns(type)

            return eval_metrics_results
