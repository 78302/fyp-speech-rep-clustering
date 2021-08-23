import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
from vqapc import toy_vqapc
import glob
import matplotlib.pyplot as plt


# Model loading
def load_model(path, model, optimizer):
    checkpoint = torch.load(path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def eu_distance(a, b):
    """
    Calculate the Euclidean Distance between two point
    :param a: Starting point
    :param b: End Point
    :return: Euclidean Distance
    """
    # dist = np.sqrt(np.sum(np.square(a - b))
    # np.linalg.norm(a - b)  # numpy function can replace the above
    return np.linalg.norm(a - b)


def assign_cluster(datapoint, centers):
    """
    Assign the given datapoint to a cluster center
    :param datapoint: a data vector
    :return: c_index: The assigned cluster index
    :return: dist: The clustering error (L2 distance)
    """
    dists = np.array([eu_distance(c, datapoint) for c in centers])
    c_index = np.argmin(dists)
    dist = np.min(dists)
    return c_index, dist

def closest_centroid(points, centroids):
	distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2)) # 计算欧氏距离
	return np.argmin(distances, axis=0), np.min(distances, axis=0)


def pretrain_representations(pretrain_path):
    """
    Get the representation through a pretrain model
    :param    pretrain_path: pretrain model file path
    :return   model        : pretrained model
    """

    # Import the pretrain model
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(DEVICE)

    rnn_pretrain_model = toy_vqapc(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4).to(DEVICE) # toy_lstm
    optimizer_pretrain = torch.optim.Adam(rnn_pretrain_model.parameters(), lr=0.001)  # just attach but no use
    rnn_pretrain_model, optimizer_pretrain = load_model(pretrain_path, rnn_pretrain_model, optimizer_pretrain) # load the model
    pre_train_model = nn.Sequential(*list(rnn_pretrain_model.children())[:-1]).to(DEVICE)  # only take the LSTM part
    pre_train_model.eval()
#     print(pre_train_model)
    return pre_train_model




if __name__ == '__main__':

    a = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])

    b = np.array([[2, 4], [2, 4], [1, 2]])

    print(closest_centroid(a, b))

    # SCP_FILE = './data/si284-0.9-train.fbank.scp'  # scp file path under Ubuntu environment
    # UTT_RELATIVE_PATH = './data/'  # relative path of ark file under Ubuntu environment
    # C = 24  # cutting position to divide the list
    #
    # # Read the SCP file
    # with open(SCP_FILE, 'rb') as scp_file:
    #     lines = scp_file.readlines()
    #     for line in lines[:1]:  # remove [:K]
    #         tempt = str(line).split()[1]
    #         file_loc = tempt.split(':')[0][C:]
    #         pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
    #
    #         # Read the ark file to get utterance
    #         with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
    #             ark_file.seek(int(pointer))
    #             utt_mat = kaldiark.parse_feat_matrix(ark_file)













    # from classifier import classification_net
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #
    #
    # classifier = classification_net(INPUT_SIZE=512, HIDDEN_SIZE=512, OUTPUT_SIZE=43, PRETRAIN_PATH='./pretrain_model/model/Epoch50.pth.tar').to(DEVICE)
    # print(classifier)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=0.001)  # optimize all require grad's parameters
    #
    # path = './model_classifier/Epoch1_test-up-lr0001.pth.tar'
    # classifier, optimizer = load_model(path, classifier, optimizer) # load the model

    # rnn_pretrain_model = toy_lstm(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4)
    # optimizer_pretrain = torch.optim.Adam(rnn_pretrain_model.parameters(), lr=0.001)  # just attach but no use
    # rnn_pretrain_model, optimizer_pretrain = load_model('./model/Epoch1.pth.tar', rnn_pretrain_model, optimizer_pretrain) # load the model
    # pre_train_model = nn.Sequential(*list(rnn_pretrain_model.children())[:1])  # only take the LSTM part
    # print(pre_train_model)


    # Test the k-means
    # import random
    #
    #
    # temp = None
    # start = True
    # epochs = 0
    #
    # for e in range(10):
    #
    #     epoch_error = 0
    #
    #     # Read the SCP file
    #     with open('./data/raw_fbank_train_si284.1.scp', 'rb') as scp_file:
    #         # mlp use '../remote/data/wsj/fbank/' replace '/data/'
    #         lines = scp_file.readlines()
    #         for line in lines[:5]:
    #
    #             tempt = str(line).split()[1]
    #             file_loc = tempt.split(':')[0][28:]  # mlp keep [18:]
    #             pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
    #
    #             # Read the ark file to get utterance
    #             with open('./data' + file_loc, 'rb') as ark_file:
    #                 # use '../remote/data' + file_loc replace './data/' + file_loc
    #                 ark_file.seek(int(pointer))
    #                 utt_mat = kaldiark.parse_feat_matrix(ark_file)
    #
    #                 # Use model to get representations
    #                 path = './pretrain_model/model/Epoch50.pth.tar'
    #                 utt_mat = pretrain_representations(path, utt_mat)
    #
    #                 # Init centers: randomly pick k data from data set as centers
    #                 if start:
    #                     centers = np.array(random.sample(list(utt_mat), 4))  # k=4
    #                     start = False
    #                     print(centers.shape)
    #
    #                 # Assign data to clusters
    #                 assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
    #
    #                 # Update centers
    #                 for c_index in range(4):  # k=4
    #                     data_in_c = np.array([utt_mat[i] for i in range(utt_mat.shape[0]) if assigns[i][0] == c_index])
    #                     centers[c_index] = np.mean(data_in_c, axis=0)
    #
    #                 # Calculate the clustering loss
    #                 assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
    #                 epoch_error += np.sum(assigns, axis=0)[1]
    #             # temp = assigns  # store the old assigns
    #
    #     print("Epoch {:d} error: {:0.7f}".format((epochs+1), epoch_error))
    #     epochs += 1
