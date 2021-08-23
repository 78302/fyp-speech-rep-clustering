import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob

import argparse
# Deal with the use input parameters
# ideal input: epoch, learning-rate, K, input-size, hidden-size, train-scp-path, dev-scp-path, layers
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--learning_rate', '-lr', help='Learning rate, not required', type=float, default=0.001)
parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=1)
parser.add_argument('--gap', '-k', help='Position in origin where the first prediction correspond to, not required', type=int,  default=2)
parser.add_argument('--input_size', '-is', help='Input dimension, not required', type=float, default=40)
parser.add_argument('--hidden_size', '-hs', help='Hidden vector dimension, not required', type=float, default=512)
parser.add_argument('--layers', '-l', help='Number of LSTM layers used, not required', type=int, default=4)
parser.add_argument('--type', '-t', help='Ubuntu type or mlp type, not required default is Ubuntu type', type=int,  default=1)
parser.add_argument('--order', '-o', help='The order of the epoch, not required', type=int,  default=1)

# parser.add_argument('--train_scp_path', '-tsp', help='Path of the input training scp file, not required', default='./data/raw_fbank_train_si284.scp')
# parser.add_argument('--dev_scp_path', '-dsp', help='Path of the input validation scp file, not required', default='./data/raw_fbank_dev.scp')
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name
LEARNING_RATE = args.learning_rate
EPOCH = args.epoch
K = args.gap
INPUT_SIZE = args.input_size
HIDDEN_SIZE = args.hidden_size
LAYERS = args.layers
TYPE = args.type
ORDER = args.order

# Decide the file path under different environment
# Python do not have switch case, use if else instead
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #
rnn = toy_lstm(INPUT_SIZE=INPUT_SIZE, HIDDEN_SIZE=HIDDEN_SIZE, LAYERS=LAYERS).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all parameters
# Learning rate decay schedule
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)

# from functions import load_model
# if ORDER != 1:  # load the previous model
#     path = './pretrain_model/model/Epoch{:d}_{:s}.pth.tar'.format(ORDER-1, NAME)
#     # print(path)
#     rnn, optimizer = load_model(path, rnn, optimizer) # load the model
# # exit()

# Load loss:
# try:
#     losses = np.load(NAME + '_losses.npy')
# except:
#     losses = np.zeros((2, 100))



# optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all parameters
loss_func = nn.MSELoss()


# Train + Dev
train_loss = []
valid_loss = []
losses = np.zeros((2, EPOCH))
min_valid_loss = np.inf

import time
start = time.time()
tmp = start

for i in range(EPOCH):
    total_train_loss = []
    rnn.train()  # Training

    # Use the total scp files
    # Read data index from the total scp file
    with open(TRAIN_SCP_PATH, 'rb') as scp_file:
        lines = scp_file.readlines()
        for line in lines:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][C:]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
            # print(file_loc, temp.split(':')[0][14:])

            # According to the file name and pointer to get the matrix
            with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                utt_mat = torch.Tensor(utt_mat).to(device)  # change data to tensor

                output = rnn(utt_mat[:, :-K, :])

                #                 print(utt_mat.shape, output.shape)

                loss = loss_func(output, utt_mat[:, K:, :])  # compute the difference
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # back-prop
                optimizer.step()  # gradients
                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))
    # print('train complete!')

    total_valid_loss = []
    rnn.eval()  # Validation

    # Use one of scp files
    # Read data index from the total scp file
    with open(DEV_SCP_PATH, 'rb') as scp_file:
        # mlp file path: ./data/raw_fbank_dev.scp
        # test file path: ./data/raw_fbank_train_si284.2.scp
        # win file path: ./data/raw_fbank_train_si284.2.scp
        lines = scp_file.readlines()
        for line in lines:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][C:]
            # ark file path; keep [18:]
            # win file keep [28:]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

            # According to the file name and pointer to get the matrix
            with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                utt_mat = torch.Tensor(utt_mat).to(device)  # change data to tensor

                with torch.no_grad():
                    output = rnn(utt_mat[:, :-K, :])  # rnn output

                #                     print(utt_mat_mat.shape, output.shape)

                loss = loss_func(output, utt_mat[:, K:, :])
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
    # print('dev complete!')

#     if (valid_loss[-1] < min_valid_loss):
#         torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
#                     'valid_loss': valid_loss}, './LSTM.model')
#         min_valid_loss = valid_loss[-1]

    min_valid_loss = np.min(valid_loss)

    end = time.time()

    # save the net
    if ((i + 1) % 10 == 0):
        torch.save({'epoch': i + 1, 'state_dict': rnn.state_dict(), 'train_loss': train_loss,
                    'valid_loss': valid_loss, 'optimizer': optimizer.state_dict()},
                    './pretrain_model/model/Epoch{:d}_{:s}.pth.tar'.format(i+1, NAME))

    # Log
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}, time: {:0.7f}').format((i + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  valid_loss[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'],
                                                                  (end-tmp))
    tmp = end
    mult_step_scheduler.step()  # 学习率更新
    losses[0][i] = train_loss[-1]
    losses[1][i] = valid_loss[-1]
    print(log_string)  # 打印日志

print(end-start)


# Save the train loss into npy
np.save(NAME + '_losses.npy', losses)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load _losses
losses = np.load(NAME + '_losses.npy')

y_t = losses[0]
y_d = losses[1]
x = np.arange(1,EPOCH+1)
fig, ax = plt.subplots(figsize=(14,7))
ax.plot(x,y_t,'r',label='Train Loss')
ax.plot(x,y_d,'b',label='Dev Loss')

ax.set_title('Loss',fontsize=18)
ax.set_xlabel('epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
ax.set_ylabel('loss', fontsize='x-large',fontstyle='oblique')
ax.legend()

plt.savefig("{:s}_apc_loss.pdf".format(NAME))
