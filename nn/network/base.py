import os
import sys
import shutil
import logging
import numpy as np
import tensorflow as tf
import torch

from nn.utils.misc import log_metrics, zipdir

logger = logging.getLogger("tf")
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

OPTIMIZERS = {
    "adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
   "momentum": lambda params, lr: torch.optim.SGD(params, momentum=0.9, lr=lr),
    "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr)
}


class BaseNet(torch.nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
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

        self.sess = tf.compat.v1.Session()

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
        # self.saver = tf.compat.v1.train.Saver()
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

        # restore = False # TODO: REMOVE THIS LINE LATER, SOMEHOW RESTORE IS SET TO TRUE CAUSING ERRORS
        # if restore:
        #     # self.saver.restore(self.sess, os.path.join(restore_dir, "model.ckpt"))
        #     self.sess.run(self.lr.assign(self.base_lr))
        # else:
        #     self.sess.run(tf.compat.v1.global_variables_initializer())

    def build_optimizer(self, base_lr, optimizer="adam", anneal_lr=True):
        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        self.lr = tf.Variable(base_lr, trainable=False, name="base_lr")
        self.optimizer = OPTIMIZERS[optimizer](self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def get_batch(self, batch_size, iterator):
        batch_x, batch_y = iterator.next_batch(batch_size)
        if batch_y is None:
            feed_dict = {self.input:batch_x}
        else:
            feed_dict = {self.input:batch_x, self.target:batch_y}
        return feed_dict, (batch_x, batch_y)

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

        step = 0

        # Run validation once before starting training
        if not debug and epochs > 0:
            valid_metrics_results = self.eval(batch_size, type='valid')
            log_metrics(logger, "valid - epoch=%s"%0, valid_metrics_results)

        for ep in range(1, epochs+1):
            if self.anneal_lr:
                if ep == int(0.75*epochs):
                    self.lr = self.lr/5
            while self.train_iterator.epochs_completed < ep:
                feed_dict, _ = self.get_batch(batch_size, self.train_iterator)
                # results, _ = self.sess.run(
                    # [self.train_metrics, self.train_op], feed_dict=feed_dict)
                train_x = feed_dict.keys()

                result_sequence = self.forward(train_x)
                self.train_loss, self.eval_losses = self.compute_loss()
                self.train_metrics["train_loss"] = self.train_loss
                self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
                self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
                self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
                self.loss = self.train_loss

                self.run_extra_fns("train")

                if step % print_interval == 0:
                    log_metrics(logger, "train - iter=%s"%step, self.train_metrics)
                step += 1

            if ep % eval_every_n_epochs == 0:
                valid_metrics_results = self.eval(batch_size, type='valid')
                log_metrics(logger, "valid - epoch=%s"%ep, valid_metrics_results)

            if ep % save_every_n_epochs == 0:
                self.saver.save(self.sess, os.path.join(self.save_dir, "model.ckpt"))
            
        test_metrics_results = self.eval(batch_size, type='test')
        log_metrics(logger, "test - epoch=%s"%epochs, test_metrics_results)

    def eval(self,
             batch_size,
             type='valid'):

        eval_metrics_results = {k:[] for k in self.eval_metrics.keys()}
        eval_outputs = {"input":[], "output":[]}
        
        eval_iterator = self.get_iterator(type)
        eval_iterator.reset_epoch()
        
        while eval_iterator.get_epoch() < 1:
            if eval_iterator.X.shape[0] < 100:
                batch_size = eval_iterator.X.shape[0]
            feed_dict, _ = self.get_batch(batch_size, eval_iterator)
            fetches = {k:v for k, v in self.eval_metrics.items()}
            fetches["output"] = self.output
            fetches["input"] = self.input
            # TODO misschien is dit wel nodig? wie weet
            # results = self.sess.run(fetches, feed_dict=feed_dict)


            for k in self.eval_metrics.keys():
                eval_metrics_results[k].append(fetches[k])
            eval_outputs["input"].append(fetches["input"])
            eval_outputs["output"].append(fetches["output"])

        eval_metrics_results = {k:np.mean([i.detach().numpy() for i in v], axis=0) for k,v in eval_metrics_results.items()}
        np.savez_compressed(os.path.join(self.save_dir, "outputs.npz"), 
                            input=np.concatenate(eval_outputs["input"], axis=0),
                            output=np.concatenate([output.detach().numpy() for output in eval_outputs["output"]], axis=0))

        self.run_extra_fns(type)

        return eval_metrics_results


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
        # self.saver = tf.compat.v1.train.Saver()
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

        # restore = False # TODO: REMOVE THIS LINE LATER, SOMEHOW RESTORE IS SET TO TRUE CAUSING ERRORS
        # if restore:
        #     # self.saver.restore(self.sess, os.path.join(restore_dir, "model.ckpt"))
        #     self.sess.run(self.lr.assign(self.base_lr))
        # else:
        #     self.sess.run(tf.compat.v1.global_variables_initializer())

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

        step = 0

        # Run validation once before starting training
        if not debug and epochs > 0:
            valid_metrics_results = self.eval(batch_size, type='valid')
            log_metrics(logger, "valid - epoch=%s" % 0, valid_metrics_results)

        for ep in range(1, epochs + 1):
            if self.anneal_lr:
                if ep == int(0.75 * epochs):
                    self.lr = self.lr / 5
            while self.train_iterator.epochs_completed < ep:
                feed_dict, _ = self.get_batch(batch_size, self.train_iterator)
                # results, _ = self.sess.run(
                # [self.train_metrics, self.train_op], feed_dict=feed_dict)

                result_sequence = self.forward(torch.Tensor(feed_dict["input"]))
                self.train_loss, self.eval_losses = self.compute_loss()

                self.train_metrics["train_loss"] = self.train_loss
                self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
                self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
                self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
                self.loss = self.train_loss
                self.loss.backward()
                self.optimizer.step()

                self.run_extra_fns("train") # DUnno what this does

                self.optimizer.zero_grad()

                if step % print_interval == 0:
                    log_metrics(logger, "train - iter=%s" % step, self.train_metrics)
                step += 1

            if ep % eval_every_n_epochs == 0:
                valid_metrics_results = self.eval(batch_size, type='valid')
                log_metrics(logger, "valid - epoch=%s" % ep, valid_metrics_results)

            if ep % save_every_n_epochs == 0:
                self.saver.save(self.sess, os.path.join(self.save_dir, "model.ckpt"))

        test_metrics_results = self.eval(batch_size, type='test')
        log_metrics(logger, "test - epoch=%s" % epochs, test_metrics_results)

    def eval(self,
             batch_size,
             type='valid'):

        self.eval_metrics["train_loss"] = 0
        self.eval_metrics["eval_pred_loss"] = 0
        self.eval_metrics["eval_extrap_loss"] = 0
        self.eval_metrics["eval_recons_loss"] = 0
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
            # TODO misschien is dit wel nodig? wie weet
            # results = self.sess.run(fetches, feed_dict=feed_dict)
            inp = torch.Tensor(feed_dict["input"])
            print("\n\n\n\n", inp.shape)
            self.output = self.conv_feedforward(inp)
            self.train_loss, self.eval_losses = self.compute_loss()
            self.train_metrics["train_loss"] = self.train_loss
            self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
            self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
            self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
            self.loss = self.train_loss

            for k in self.eval_metrics.keys():
                eval_metrics_results[k].append(self.eval_metrics[k])
            eval_outputs["input"].append(feed_dict["input"])
            eval_outputs["output"].append(self.eval_losses)


        eval_metrics_results = {k: np.mean([i.detach().numpy() for i in v], axis=0) for k, v in
                                eval_metrics_results.items()}
        np.savez_compressed(os.path.join(self.save_dir, "outputs.npz"),
                            input=np.concatenate(eval_outputs["input"], axis=0),
                            output=np.concatenate([output.detach().numpy() for output in eval_outputs["output"]],
                                                  axis=0))

        self.run_extra_fns(type)

        return eval_metrics_results