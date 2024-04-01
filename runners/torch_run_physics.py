import os
import torch
import logging
from nn.network import physics_models
from nn.utils.misc import classes_in_module
from nn.datasets.iterators import get_iterators
import argparse

parser = argparse.ArgumentParser(description="PyTorch version of the TensorFlow script.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
parser.add_argument("--save_dir", type=str, default="", help="Directory to save checkpoint and logs")
parser.add_argument("--use_ckpt", action="store_true", help="Whether to start from scratch or start from checkpoint")
parser.add_argument("--ckpt_dir", type=str, default="", help="Checkpoint directory to use")
parser.add_argument("--base_lr", type=float, default=1e-3, help="Base learning rate")
parser.add_argument("--anneal_lr", action="store_false", help="Whether to anneal lr after 0.75 of total epochs")
parser.add_argument("--optimizer", type=str, default="rmsprop", help="Optimizer to use")
parser.add_argument("--save_every_n_epochs", type=int, default=5, help="Epochs between checkpoint saves")
parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="Epochs between validation run")
parser.add_argument("--print_interval", type=int, default=10, help="Print train metrics every n mini-batches")
parser.add_argument("--debug", action="store_true", help="If true, eval is not run before training")
parser.add_argument("--test_mode", action="store_true", help="If true, only run test set")

parser.add_argument("--task", type=str, default="", help="Type of task.")
parser.add_argument("--model", type=str, default="PhysicsNet", help="Model to use.")
parser.add_argument("--recurrent_units", type=int, default=100, help="Number of units for each lstm, if using black-box dynamics.")
parser.add_argument("--lstm_layers", type=int, default=1, help="Number of lstm cells to use, if using black-box dynamics")
parser.add_argument("--cell_type", type=str, default="", help="Type of pendulum to use.")
parser.add_argument("--encoder_type", type=str, default="conv_encoder", help="Type of encoder to use.")
parser.add_argument("--decoder_type", type=str, default="conv_st_decoder", help="Type of decoder to use.")
parser.add_argument("--autoencoder_loss", type=float, default=0.0, help="Autoencoder loss weighing.")
parser.add_argument("--alt_vel", action="store_true", help="Whether to use linear velocity computation.")
parser.add_argument("--color", action="store_true", help="Whether images are RGB or grayscale.")
parser.add_argument("--datapoints", type=int, default=0, help="How many datapoints from the dataset to use. Useful for measuring data efficiency. Default=0 uses all data.")

args = parser.parse_args()

logger = logging.getLogger("torch")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

model_classes = classes_in_module(physics_models)
Model = model_classes[args.model]

data_file, test_data_file, cell_type, seq_len, test_seq_len, input_steps, pred_steps, input_size = {
    "bouncing_balls": (
        "bouncing/color_bounce_vx8_vy8_sl12_r2.npz", 
        "bouncing/color_bounce_vx8_vy8_sl30_r2.npz", 
        "bouncing_ode_cell",
        12, 30, 4, 6, 32*32),
    "spring_color": (
        "spring_color/color_spring_vx8_vy8_sl12_r2_k4_e6.npz", 
        "spring_color/color_spring_vx8_vy8_sl30_r2_k4_e6.npz",
        "spring_ode_cell",
        12, 30, 4, 6, 32*32),
    "spring_color_half": (
        "spring_color_half/color_spring_vx4_vy4_sl12_r2_k4_e6_halfpane.npz", 
        "spring_color_half/color_spring_vx4_vy4_sl30_r2_k4_e6_halfpane.npz", 
        "spring_ode_cell",
        12, 30, 4, 6, 32*32),
    "3bp_color": (
        "3bp_color/color_3bp_vx2_vy2_sl20_r2_g60_m1_dt05.npz", 
        "3bp_color/color_3bp_vx2_vy2_sl40_r2_g60_m1_dt05.npz", 
        "gravity_ode_cell",
        20, 40, 4, 12, 36*36),
    "mnist_spring_color": (
        "mnist_spring_color/color_mnist_spring_vx8_vy8_sl12_r2_k2_e12.npz", 
        "mnist_spring_color/color_mnist_spring_vx8_vy8_sl30_r2_k2_e12.npz", 
        "spring_ode_cell",
        12, 30, 3, 7, 64*64)
}[args.task]

if __name__ == "__main__":
    if not args.test_mode:
            network = Model(args.task, args.recurrent_units, args.lstm_layers, cell_type, 
                            seq_len, input_steps, pred_steps,
                            args.autoencoder_loss, args.alt_vel, args.color, 
                            input_size, args.encoder_type, args.decoder_type)

            network.build_graph()
            network.build_optimizer(args.base_lr, args.optimizer, args.anneal_lr)
            network.initialize_graph(args.save_dir, args.use_ckpt, args.ckpt_dir)

            data_iterators = get_iterators(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "../data/datasets/%s"%data_file), conv=True, datapoints=args.datapoints)
            network.get_data(data_iterators)
            network.train(args.epochs, args.batch_size, args.save_every_n_epochs, args.eval_every_n_epochs,
                        args.print_interval, args.debug)

            torch.cuda.empty_cache()

    network = Model(args.task, args.recurrent_units, args.lstm_layers, cell_type, 
                    test_seq_len, input_steps, pred_steps,
                args.autoencoder_loss, args.alt_vel, args.color, 
                input_size, args.encoder_type, args.decoder_type)

    network.build_graph()
    network.build_optimizer(args.base_lr, args.optimizer, args.anneal_lr)
    network.initialize_graph(args.save_dir, True, args.ckpt_dir)

    data_iterators = get_iterators(
                        os.path.join(
                            os.path.dirname(os.path.realpath(__file__)), 
                            "../data/datasets/%s" % test_data_file), conv=True, datapoints=args.datapoints)
    network.get_data(data_iterators)
    network.train(0, args.batch_size, args.save_every_n_epochs, args.eval_every_n_epochs,
                args.print_interval, args.debug)
  