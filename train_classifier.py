import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
from classifier import classification_net
import glob

import argparse
# Deal with the use input parameters
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--learning_rate', '-lr', help='Learning rate, not required', type=float, default=0.001)
parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=1)
parser.add_argument('--gap', '-k', help='Position in origin where the first prediction correspond to, not required', type=int,  default=2)
parser.add_argument('--input_size', '-is', help='Input dimension, not required', type=int, default=40)
parser.add_argument('--hidden_size', '-hs', help='Hidden vector dimension, not required', type=int, default=512)
parser.add_argument('--output_size', '-os', help='Output dimension, not required', type=int, default=43)
parser.add_argument('--model_path', '-p', help='Path of the pre-trained model, not required', default=None)
parser.add_argument('--type', '-t', help='Ubuntu type or mlp type, not required default is Ubuntu type', type=int,  default=1)
parser.add_argument('--order', '-o', help='The order of the epoch, not required', type=int,  default=1)
args = parser.parse_args()

# Assign parameters
NAME = args.name
LEARNING_RATE = args.learning_rate
EPOCH = args.epoch
K = args.gap
INPUT_SIZE = args.input_size
HIDDEN_SIZE = args.hidden_size
OUTPUT_SIZE = args.output_size
PRETRAIN_PATH = args.model_path
TYPE = args.type
ORDER = args.order

# Decide the file path under different environment
# Python do not have switch case, use if else instead
if TYPE == 1:  # under Ubbuntu test environment
    TRAIN_UTT_SCP_PATH = './data/si284-0.9-train.fbank.scp'
    TRAIN_LABEL_SCP_PATH = './data/si284-0.9-train.bpali.scp'
    TRAIN_LABEL_PATH = './data/train-si284.bpali'
    DEV_UTT_SCP_PATH = './data/si284-0.9-train.fbank.scp'
    DEV_LABEL_SCP_PATH = './data/si284-0.9-train.bpali.scp'
    DEV_LABEL_PATH = './data/train-si284.bpali'
    UTT_RELATIVE_PATH = './data'  # relative path of ark file under Ubuntu environment
    C = 24  # cutting position to divide the list
else:
    TRAIN_UTT_SCP_PATH = '../remote/data/wsj/extra/si284-0.9-train.fbank.scp'
    TRAIN_LABEL_SCP_PATH = '../remote/data/wsj/extra/si284-0.9-train.bpali.scp'
    TRAIN_LABEL_PATH = '../remote/data/wsj/extra/train-si284.bpali'
    DEV_UTT_SCP_PATH = '../remote/data/wsj/extra/si284-0.9-dev.fbank.scp'
    DEV_LABEL_SCP_PATH = '../remote/data/wsj/extra/si284-0.9-dev.bpali.scp'
    DEV_LABEL_PATH = '../remote/data/wsj/extra/si284-0.9-dev.bpali'
    UTT_RELATIVE_PATH = '../remote/data'
    C = 14

with open('./data/phones-unk.txt', 'r') as ph_file:
    standard = ph_file.read().splitlines()
    # print(standard)


# # Load loss and acc:
# try:
#     e_results = np.load(NAME + '_results.npy')
# except:
#     e_results = np.zeros((4, 20))  # train loss, val loss, train acc, val acc | 20 epochs


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #
# classifier use representation dimension as input size -- i.e. HIDDEN_SIZE
classifier = classification_net(INPUT_SIZE=INPUT_SIZE, HIDDEN_SIZE=HIDDEN_SIZE, OUTPUT_SIZE=OUTPUT_SIZE, PRETRAIN_PATH=PRETRAIN_PATH).to(DEVICE)
# print(classifier)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LEARNING_RATE)  # optimize all require grad's parameters
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)
# from functions import load_model
# if ORDER != 1:  # load the previous model
#     path = './model_classifier/Epoch{:d}_{:s}.pth.tar'.format(ORDER-1, NAME)
#     classifier, optimizer = load_model(path, classifier, optimizer) # load the model


# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LEARNING_RATE)  # optimize all require grad's parameters
loss_func = torch.nn.CrossEntropyLoss()

train_bpali_file = open(TRAIN_LABEL_PATH, 'rb')
train_fbank_scp = open(TRAIN_UTT_SCP_PATH, 'rb')
dev_bpali_file = open(DEV_LABEL_PATH, 'rb')
dev_fbank_scp = open(DEV_UTT_SCP_PATH, 'rb')
train_fbank_lines = train_fbank_scp.readlines()
dev_fbank_lines = dev_fbank_scp.readlines()

# Train + Dev
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
e_results = np.zeros((4, 20))
min_valid_loss = np.inf

import time
start = time.time()
tmp = start

for i in range(EPOCH):
    total_train_loss = []
    classifier.train()  # Training

    train_correct = 0
    train_total = 0
    with open(TRAIN_LABEL_SCP_PATH, 'rb') as scp_file:
        bpali_lines = scp_file.readlines()
        for idx,line in enumerate(bpali_lines):  # [:K]
            # Find the label
            temp = str(line).split()[1]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the label
            # Linux only has \n
            train_bpali_file.seek(int(pointer))
            transcript = train_bpali_file.readline()
            labels = str(transcript)[2:-3].split()
            # Change the label into numbers
            targets = [standard.index(l) for l in labels]
            targets = torch.tensor(targets, dtype=torch.long).to(DEVICE)
            # print(labels[:5], targets[:5])

            # Find the utterance
            utt_line = train_fbank_lines[idx]
            temp = str(utt_line).split()[1]
            utt_file_loc = temp.split(':')[0][C:]  # ubuntu [24:] mlp [14:]
            utt_pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
            # print(temp, utt_file_loc)

            with open(UTT_RELATIVE_PATH + utt_file_loc, 'rb') as ark_file:
                # mlp use: '../remote/data' + utt_file_loc
                # ubuntu use: './data' + utt_file_loc
                ark_file.seek(int(utt_pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)
                # print(utt_mat.shape )
                utt_mat = torch.Tensor(utt_mat).to(DEVICE)   # change data to tensor
                # utt_mat = torch.unsqueeze(utt_mat, 0)
                # Use batch of utturance instead of single

                output = classifier(utt_mat)
                # print(output.shape)
                loss = loss_func(output, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Counts the correst preds and total numbers
                train_total += targets.shape[0]

                num_corrects = torch.argmax(output, dim=1)
                train_correct += int(targets.eq(torch.argmax(output, dim=1)).sum())
                # print(train_total, train_correct)


                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))
    train_acc.append(train_correct/train_total)


    total_valid_loss = []
    classifier.eval()  # Validation

    dev_correct = 0
    dev_total = 0
    with open(DEV_LABEL_SCP_PATH, 'rb') as scp_file:
        bpali_lines = scp_file.readlines()
        for didx,line in enumerate(bpali_lines):  # [:K//2]
            # Find the label
            temp = str(line).split()[1]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the label
            # Linux only has \n
            dev_bpali_file.seek(int(pointer))
            transcript = dev_bpali_file.readline()
            labels = str(transcript)[2:-3].split()
            # Change the label into numbers
            targets = [standard.index(l) for l in labels]
            targets = torch.tensor(targets, dtype=torch.long).to(DEVICE)
            # print(pointer, len(labels))

            # Find the utterance
            utt_line = dev_fbank_lines[didx]
            temp = str(utt_line).split()[1]
            utt_file_loc = temp.split(':')[0][C:]  # ark file path; keep [14:], test ues 24
            utt_pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
            # print(didxutt_file_loc, utt_pointer)
            # According to the file name and pointer to get the matrix
            with open(UTT_RELATIVE_PATH + utt_file_loc, 'rb') as ark_file:
                ark_file.seek(int(utt_pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)
                # print(utt_mat.shape)
                utt_mat = torch.Tensor(utt_mat).to(DEVICE)   # change data to tensor

                with torch.no_grad():
                    output = classifier(utt_mat)

                loss = loss_func(output, targets)

                # Counts the correst preds and total numbers
                dev_total += targets.shape[0]
                dev_correct += int(targets.eq(torch.argmax(output, dim=1)).sum())

                    # loss=loss_func(output,y)
                total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
    valid_acc.append(dev_correct/dev_total)

    # save the net

    min_valid_loss = np.min(valid_loss)
    end = time.time()

    if ((i + 1) % 1 == 0):
        torch.save({'epoch': i + 1, 'state_dict': classifier.state_dict(), 'train_loss': train_loss,
                    'valid_loss': valid_loss, 'optimizer': optimizer.state_dict()},
                    './model_classifier/Epoch{:d}_{:s}.pth.tar'.format(i+1, NAME))


    # Log
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, train_acc: {:0.6f}, valid_loss: {:0.6f}, '
                  'valid_acc: {:0.6f}, best_valid_loss: {:0.6f}, lr: {:0.7f}, time: {:0.7f}').format((i + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  train_acc[-1],
                                                                  valid_loss[-1],
                                                                  valid_acc[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'],
                                                                  (end - tmp))
    tmp = end
    mult_step_scheduler.step()  # 学习率更新
    e_results[0][i] = train_loss[-1]
    e_results[1][i] = valid_loss[-1]
    e_results[2][i] = 1 - train_acc[-1]
    e_results[3][i] = 1 - valid_acc[-1]

    print(log_string)  # 打印日志

print(end-start)

train_bpali_file.close()
train_fbank_scp.close()
dev_bpali_file.close()
dev_fbank_scp.close()

# Save the train loss into npy
np.save(NAME + '_results.npy', e_results)
