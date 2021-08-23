import os
import logging
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils import data

from vqapc_model import GumbelAPCModel
import kaldiark
# from datasets import LibriSpeech


def main():
    parser = argparse.ArgumentParser()

    # RNN architecture config.
    parser.add_argument("--rnn_num_layers", default=4, type=int,
                        help="Number of layers for RNN.")
    parser.add_argument("--rnn_hidden_size", default=512, type=int,
                        help="Hidden size of RNN.")
    parser.add_argument("--rnn_dropout", default=0., type=float,
                        help="RNN dropout rate.")
    parser.add_argument("--rnn_residual", action="store_true",
                        help="Apply residual connections if true.")

    # VQ layer config.
    parser.add_argument("--codebook_size", default=128, type=int,
                        help="Codebook size; all VQ layers will use the same \
                        value.")
    parser.add_argument("--code_dim", default=512, type=int,
                        help="Size of each code.")
    parser.add_argument("--gumbel_temperature", default=0.5, type=float,
                        help="Gumbel-Softmax temperature.")
    parser.add_argument("--vq_hidden_size", default=-1, type=int,
                        help="Hidden size for the VQ layer.")
    # parser.add_argument("--apply_VQ", default="[0, 0, 0, 1]", nargs="+",
    #                     help="Quantize layer output if 1. E.g., [1, 0, 1] will \
    #                     apply VQ to the output of the first and third layers.")

    # Optimization config.
    parser.add_argument("--optimizer", default="adam", choices=["adam"],
                        help="Just use adam.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Mini-batch size.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="Learning rate.")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--n_future", default=2, type=int,
                        help="Given x_1, ..., x_t, predict x_{t + n_future}.")
    parser.add_argument("--clip_thresh", default=1., type=float,
                        help="Threshold for gradient clipping.")

  # Data config.
  # parser.add_argument("--librispeech_home",
  #                     default="./librispeech_data/preprocessed", type=str,
  #                     help="Path to the LibriSpeech home directory.")
  # parser.add_argument("--train_partition", nargs="+", required=True,
  #                     help="Partition(s) to be used for training.")
  # parser.add_argument("--train_sampling", default=1., type=float,
  #                     help="Ratio to sample for actual training.")
  # parser.add_argument("--val_partition", nargs="+", required=True,
  #                     help="Partition(s) to be used for validation.")
  # parser.add_argument("--val_sampling", default=1., type=float,
  #                     help="Ratio to sample for actual validation.")

  # Misc config.
    parser.add_argument("--feature_dim", default=40, type=int,
                        help="Dimension of input feature.")
    parser.add_argument("--load_data_workers", default=8, type=int,
                        help="Number of parallel data loaders.")
    parser.add_argument("--exp_name", default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument("--store_path", type=str,
                        default="./logs",
                        help="Where to save the trained models and logs.")

    config = parser.parse_args()

## Log info use other
  # # Create the directory to dump exp logs and models.
  # model_dir = os.path.join(config.store_path, config.exp_name + '.dir')
  # os.makedirs(config.store_path, exist_ok=True)
  # os.makedirs(model_dir, exist_ok=True)
  #
  # logging.basicConfig(
  #   level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
  #   filename=os.path.join(model_dir, config.exp_name), filemode='w')
  #
  # # Define a new Handler to log to console as well.
  # console = logging.StreamHandler()
  # console.setLevel(logging.INFO)
  # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  # console.setFormatter(formatter)
  # logging.getLogger('').addHandler(console)
  #
  # logging.info('Model Parameters:')
  # logging.info('RNN Depth: %d' % (config.rnn_num_layers))
  # logging.info('RNN Hidden Dim: %d' % (config.rnn_hidden_size))
  # logging.info('RNN Dropout: %f' % (config.rnn_dropout))
  # logging.info('RNN Residual Connections: %s' % (config.rnn_residual))
  # logging.info('VQ Codebook Size: %d' % (config.codebook_size))
  # logging.info('VQ Codebook Dim: %d' % (config.code_dim))
  # logging.info('VQ Gumbel Temperature: %f' % (config.gumbel_temperature))
  # logging.info('VQ Hidden Dim: %d' % (config.vq_hidden_size))
  # apply_VQ = [int(q) > 0 for q in config.apply_VQ]
  # logging.info('VQ Apply: %s' % (apply_VQ))
  # logging.info('Optimizer: %s' % (config.optimizer))
  # logging.info('Batch Size: %d' % (config.batch_size))
  # logging.info('Learning Rate: %f' % (config.learning_rate))
  # logging.info('Future (n): %d' % (config.n_future))
  # logging.info('Gradient Clip Threshold: %f' % (config.clip_thresh))
  # logging.info('Training Data: %s' % (config.train_partition))
  # logging.info('Training Ratio: %f' % (config.train_sampling))
  # logging.info('Validation Data: %s' % (config.val_partition))
  # logging.info('Validation Ratio: %f' % (config.val_sampling))
  # logging.info('Number of GPUs Used: %d' % (torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #
    model = GumbelAPCModel(input_size=config.feature_dim,
                           hidden_size=config.rnn_hidden_size,
                           num_layers=config.rnn_num_layers,
                           dropout=config.rnn_dropout,
                           residual=config.rnn_residual,
                           codebook_size=config.codebook_size,
                           code_dim=config.code_dim,
                           gumbel_temperature=config.gumbel_temperature,
                           vq_hidden_size=config.vq_hidden_size,
                           apply_VQ=[1]).to(device)

    # model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate decay schedule
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[config.epochs // 2,
                                                                config.epochs // 4 * 3], gamma=0.1)
    # print(model)
    # Need prefix `module` before state_dict() when using nn.DataParallel. i.e. model.module.state_dict()
    torch.save(model.state_dict(),
            './models/vqapc/Epoch_0_{:s}.pth.tar'.format(config.exp_name))

    # Decide the file path under different environment
    # Python do not have switch case, use if else instead
    TYPE = 0
    if TYPE == 1:  # under Ubbuntu test environment
        TRAIN_SCP_PATH = './data/si284-0.9-train.fbank.scp'
        DEV_SCP_PATH = './data/si284-0.9-train.fbank.scp'
        UTT_RELATIVE_PATH = './data'  # relative path of ark file under Ubuntu environment
        C = 24  # cutting position to divide the list
    else:
        TRAIN_SCP_PATH = '../remote/data/wsj/extra/si284-0.9-train.fbank.scp'
        DEV_SCP_PATH = '../remote/data/wsj/extra/si284-0.9-dev.fbank.scp'
        UTT_RELATIVE_PATH = '../remote/data'
        C = 14

    global_step = 0
    train_loss_total = []
    val_loss_total = []
    for epoch_i in range(config.epochs):

        ####################
        ##### Training #####
        ####################
        model.train()
        train_losses = []
        # Use the total scp files
        # Read data index from the total scp file
        with open(TRAIN_SCP_PATH, 'rb') as scp_file:
            lines = scp_file.readlines()
        for line in lines[:100]:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][C:]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

            # According to the file name and pointer to get the matrix
            with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

            utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
            seq_len = utt_mat.shape[1]
            utt_mat = torch.Tensor(utt_mat).to(device)  # change data to tensor

            predicted_BxLxM, _, _ = model(utt_mat[:, :-config.n_future, :],
                                                  seq_len - config.n_future, testing=False)
            optimizer.zero_grad()  # clear gradients for this training step
            train_loss = criterion(predicted_BxLxM,
                                            utt_mat[:, config.n_future:, :])
            train_losses.append(train_loss.item())
            train_loss.backward()  # back-prop
            optimizer.step()  # gradients

        train_loss_total.append(np.mean(train_losses))

        global_step += 1

        ######################
        ##### Validation #####
        ######################
        model.eval()
        val_losses = []
        with torch.set_grad_enabled(False):
            with open(DEV_SCP_PATH, 'rb') as scp_file:
                lines = scp_file.readlines()
            for line in lines[:100]:
                temp = str(line).split()[1]
                file_loc = temp.split(':')[0][C:]
                pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

                # According to the file name and pointer to get the matrix
                with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
                    ark_file.seek(int(pointer))
                    utt_mat = kaldiark.parse_feat_matrix(ark_file)

                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                seq_len = utt_mat.shape[1]
                utt_mat = torch.Tensor(utt_mat).to(device)  # change data to tensor

                val_predicted_BxLxM, _, _ = model(
                                            utt_mat[:, :-config.n_future, :],
                                            seq_len - config.n_future, testing=True)

                val_loss = criterion(val_predicted_BxLxM,
                                     utt_mat[:, config.n_future:, :])
                val_losses.append(val_loss.item())
        val_loss_total.append(np.mean(val_losses))


        min_valid_loss = np.min(val_loss_total)

        # save the net
        if ((epoch_i + 1) % 10 == 0):
            torch.save(model.state_dict(),
                    './models/vqapc/Epoch_{:d}_{:s}.pth.tar'.format((epoch_i + 1), config.exp_name))
        # Log
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((epoch_i + 1), config.epochs,
                                                                  train_loss_total[-1],
                                                                  val_loss_total[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'])

        mult_step_scheduler.step()  # Update lr -- can remove
        print(log_string)  # 打印日志


if __name__ == '__main__':
  main()
