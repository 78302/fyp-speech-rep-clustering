# Implementation of K-means
import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
from vqapc import toy_vqapc
import glob
import matplotlib.pyplot as plt
from functions import load_model, assign_cluster, pretrain_representations, closest_centroid
import random

# Load kmeans paras
import argparse
# Deal with the use input parameters
# ideal input: name, epoch, K, pretrain_model_path, hidden-size, train-scp-path, dev-scp-path, layers
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=1)
parser.add_argument('--type', '-t', help='Ubuntu type or mlp type, not required default is Ubuntu type', type=int,  default=1)
parser.add_argument('--cluster_number', '-k', help='The largest number of clusters, not required', type=int,  default=10)
parser.add_argument('--model_path', '-p', help='Path of the pre-trained model, not required', default=None)  # model path: './pretrain_model/model/Epoch50.pth.tar'
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name
EPOCH = args.epoch
TYPE = args.type
K = args.cluster_number
PRETRAIN_PATH = args.model_path


# Decide the file path under different environment
# Python do not have switch case, use if else instead
if TYPE == 1:  # under Ubbuntu test environment
    SCP_FILE = './data/si284-0.9-train.fbank.scp'  # scp file path under Ubuntu environment
    UTT_RELATIVE_PATH = './data'  # relative path of ark file under Ubuntu environment
    C = 24  # cutting position to divide the list
else:
    SCP_FILE = '../remote/data/wsj/extra/si284-0.9-train.fbank.scp'
    UTT_RELATIVE_PATH = '../remote/data'
    C = 14


np.random.seed(100)
random.seed(100)

import time
start_time = time.time()
tmp = start_time

epochs = 0
temp = None
start = True
k=K

# Load model:
if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
    model = pretrain_representations(PRETRAIN_PATH)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load ceters and loss
import csv

for e in range(EPOCH):

    error = 0
    # assign_time = 0
    # record_time = 0
    d_centers = np.zeros(k)
    if PRETRAIN_PATH:
        n_centers = np.zeros((k, 512))
    else:
        n_centers = np.zeros((k, 40))

    with open(SCP_FILE, 'rb') as scp_file:
        lines = scp_file.readlines()
        random.shuffle(lines)
        # for utterance in the file
        for line in lines[:1000]:  # use 2 for test
            tempt = str(line).split()[1]
            file_loc = tempt.split(':')[0][C:]
            pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

            with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

            # end = time.time()
            # # print('transfer to rep: {:0.7f}'.format(end-tmp))
            # tmp = end
            # Use pretrain model to get representations
            if PRETRAIN_PATH:
                utt_mat = torch.Tensor(utt_mat).to(DEVICE)
                utt_mat = torch.unsqueeze(utt_mat, 0)
                rep = model(utt_mat)
                # print(rep)
                rep = rep[0].cpu()
                utt_mat = rep.detach().numpy()[0]

            # end = time.time()
            # print('transfer to rep: {:0.7f}'.format(end-tmp))
            # tmp = end

            # Init centers:
            # randomly pick k data from data set as centers
            if start:
                # if k <= utt_mat.shape[0]:
                #     centers = np.array(random.sample(list(utt_mat), k))  # k=4
                # else:
                u_max = np.max(utt_mat,axis=0)
                u_min = np.min(utt_mat,axis=0)
                # print(u_max.shape)

                centers = np.random.rand(k, utt_mat.shape[1])
                centers = (u_max - u_min) * centers + u_min

                start = False
                print(centers.shape)

            # Assign centers to the utterance
            assigns, errors = closest_centroid(utt_mat, centers)
            # end = time.time()
            # print('assign: {:0.7f}'.format(end-tmp))
            # tmp = end
            error += np.sum(errors)
            # print(error)

            # Record number of frames and f information
            for i in range(utt_mat.shape[0]):
                c = int(assigns[i])
                # n_centers[c] = d_centers[c]/(d_centers[c]+1) * n_centers[c] + 1/(d_centers[c]+1) * utt_mat[i]
                n_centers[c] += utt_mat[i]
                d_centers[c] += 1
            # end = time.time()
            # print('record: {:0.7f}'.format(end-tmp))
            # tmp = end

        # print(n_centers.shape)
        # print(d_centers)

        # Update Centers
        for c in range(k):
            if d_centers[c] > 0:
                centers[c] = n_centers[c] / d_centers[c]

        end = time.time()
        error = error / np.sum(d_centers)
        print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), error, (end-tmp)))
        tmp = end

        # write to the csv file for each epoch
        with open(NAME + '_result.csv', 'a+') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow([k, epochs+1, error])
        epochs += 1
print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, error, (end-start_time)))
