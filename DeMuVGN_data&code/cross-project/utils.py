import argparse
import csv
import os

import scipy.sparse as sp
import numpy as np
import torch
import ipdb
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform



def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=66, help='隐藏层的大小，小型网络一般在64， 128， 256')
    parser.add_argument('--graph_direction', type=str, default='all', help='边的方向')
    parser.add_argument('--message_function', type=str, default='no_edge',
                        help='message_function传递函数')  # 分为edge_mm、edge_network、edge_pair、no_edge
    parser.add_argument('--graph_hops', type=int, default=2, help='图神经网络的跳数')
    parser.add_argument('--word_dropout', type=float, default=0.5, help='丢弃词向量，一般在0~1之间')
    parser.add_argument('--portion', type=float, default=0.5, help='采样比例')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--hops', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_nums', type=int, default=3000, help='number of batches per epoch')

    parser.add_argument('--imbalance', action='store_true', default=True)
    parser.add_argument('--needSmote', type=str, default='on',
                        choices=['on', 'off'])
    parser.add_argument('--setting', type=str, default='smote',
                        choices=['no', 'upsampling', 'smote', 'reweight', 'embed_up', 'recon', 'newG_cls',
                                 'recon_newG'])
    # upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes;
    # embed_up:
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

    parser.add_argument('--opt_new_G', action='store_true',
                        default=False)  # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--up_scale', type=float, default=2)
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--rec_weight', type=float, default=0.000001)
    parser.add_argument('--model', type=str, default='BiGGNN',
                        choices=['sage', 'gcn', 'GAT', 'BiGGNN'])

    return parser

def weight_norm(arr):
    # 获取第三列
    third_column = arr[:, 2]

    # 找到第三列的最大值
    max_value = np.max(third_column)

    # 对第三列所有数除以最大值
    normalized_third_column = third_column / max_value

    # 更新原始数组的第三列
    arr[:, 2] = normalized_third_column

    return arr


def load_data(file, name, choose):
    """Load citation network dataset (cora only for now)"""
    # print('Loading dataset...')
    # colNorm("{}{}.csv".format(file, name))
    with open("{}{}.csv".format(file, name), 'r') as a:
        reader = csv.reader(a)
        next(reader)  # 跳过首行
        idx_features_labels = []
        for i in reader:
            row = []
            for x in i:
                row.append(x)
            idx_features_labels.append(row)

    idx_features_labels = np.array(idx_features_labels)
    weight_file = open('{}{}.txt'.format(file, choose), 'r')
    weight_data = np.genfromtxt(weight_file, delimiter=',', dtype=np.float)
    weight_data = weight_norm(weight_data)

    features = sp.csr_matrix(idx_features_labels[:, 2:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1].astype(float).astype(int))  # onehot编码

    n = idx_features_labels.shape[0]  # 结点个数

    # build graph，构建邻接矩阵
    matrix = np.zeros((n, n))  # 初始化邻接矩阵
    matrix_in = np.zeros((n, n))
    matrix_out = np.zeros((n, n))
    for i in range(len(weight_data)):
        x = int(weight_data[i][0])
        y = int(weight_data[i][1])
        weight = float(weight_data[i][2])
        matrix[x - 1][y - 1] += weight
        matrix[y - 1][x - 1] += weight
        matrix_in[x - 1][y - 1] += weight
        matrix_out[y - 1][x - 1] += weight
    adj = sp.coo_matrix(matrix)

    idx = list(range(n))
    random.shuffle(idx)

    # 随机打乱索引列表
    random.shuffle(idx)

    # 计算各个划分的长度
    test_length = int(n * 0.15)
    val_length = int(n * 0.15)

    # 进行随机划分
    idx_test = idx[:test_length]
    idx_val = idx[test_length:test_length + val_length]
    idx_train = idx[test_length + val_length:]

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    matrix_in = torch.FloatTensor(matrix_in)
    matrix_out = torch.FloatTensor(matrix_out)

    return adj, features, labels, idx_train, idx_val, idx_test, matrix_in, matrix_out


def load_cp_data(file, name, choose, mood='test', portion=0.5):
    # csv转为txt
    """Load citation network dataset (cora only for now)"""
    # print('Loading dataset...')
    # colNorm("{}{}.csv".format(file, name))
    with open("{}{}.csv".format(file, name), 'r') as a:
        reader = csv.reader(a)
        next(reader)  # 跳过首行
        idx_features_labels = []
        for i in reader:
            row = []
            for x in i:
                row.append(x)
            idx_features_labels.append(row)

    idx_features_labels = np.array(idx_features_labels)
    weight_file = open('{}{}.txt'.format(file, choose), 'r')
    weight_data = np.genfromtxt(weight_file, delimiter=',', dtype=np.int32)
    weight_data = weight_norm(weight_data)

    features = sp.csr_matrix(idx_features_labels[:, 2:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1].astype(float).astype(int))  # onehot编码

    n = idx_features_labels.shape[0]  # 结点个数

    # build graph，构建邻接矩阵
    matrix = np.zeros((n, n))  # 初始化邻接矩阵
    matrix_in = np.zeros((n, n))
    matrix_out = np.zeros((n, n))
    for i in range(len(weight_data)):
        x = int(weight_data[i][0])
        y = int(weight_data[i][1])
        weight = float(weight_data[i][2])
        matrix[x - 1][y - 1] += weight
        matrix[y - 1][x - 1] += weight
        matrix_in[x - 1][y - 1] += weight
        matrix_out[y - 1][x - 1] += weight
    adj = sp.coo_matrix(matrix)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(adj + sp.eye(adj.shape[0]))  # 对A+I归一化

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    matrix_in = torch.FloatTensor(matrix_in)
    matrix_out = torch.FloatTensor(matrix_out)
    im_class_num = 1
    idx_train = torch.tensor(range(labels.shape[0]))
    if mood == 'train':
        adj, matrix_in, matrix_out, features, labels, idx_train = src_smote(adj, matrix_in, matrix_out,
                                                                                        features, labels,
                                                                                        idx_train,
                                                                                        portion=portion,
                                                                                        im_class_num=im_class_num)

    return adj, features, labels, matrix_in, matrix_out


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# 列归一化
def get_row_min_max(df, row):
    x_max = int(df.loc[row:row, "v_num":"RealBugCount"].max(axis=1))
    x_min = int(df.loc[row:row, "v_num":"RealBugCount"].min(axis=1))
    return x_min, x_max


def colNorm(filename):
    df = pd.read_csv(filename)
    for i in range(len(df)):
        if i % 10 == 0:
            print(round(i * 100 / len(df), 2), "%")
        x_min, x_max = get_row_min_max(df, i)
        df.loc[i:i, "CountDeclMethodPrivate":"RealBugCount"] = df.loc[i:i, "CountDeclMethodPrivate":"RealBugCount"].sub(
            x_min) / (x_max - x_min)
    df.to_csv(filename, index=False)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)  # 构造
    return mx


def accuracy(output, labels):
    # print("***",output,labels,"***")
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# smote过采样
def oversample(features, edges):
    # Convert the characteristic values to a numeric data type
    data = features[:, 2:-1].astype(float)

    # Separate the labels from the rest of the features
    labels = features[:, -1].astype(float).astype(int)

    # Use SMOTE to oversample the minority class
    smote = SMOTE()
    data_resampled, labels_resampled = smote.fit_resample(data, labels)

    # Determine the number of new nodes created by SMOTE
    num_new_nodes = len(data_resampled) - len(data)

    # Create a matrix to hold the new nodes
    new_nodes = np.empty((num_new_nodes, features.shape[1]))

    # Populate the matrix with the new nodes
    for i in range(num_new_nodes):
        # Set the index to be one greater than the highest existing index
        new_nodes[i, 0] = int(np.max(features[:, 0].astype(int)) + 1 + i)
        # Set the name to "new"
        new_nodes[i, 1] = 0
        # Set the characteristic values to the values generated by SMOTE
        new_nodes[i, 2:-1] = data_resampled[len(data) + i]
        # Set the label to the label generated by SMOTE
        new_nodes[i, -1] = labels_resampled[len(data) + i]

    # Concatenate the new nodes with the original features matrix
    oversampled_features = np.concatenate((features, new_nodes))

    # Add edges from the new nodes to the original nodes
    num_original_nodes = len(data)
    for i in range(num_new_nodes):
        # Add an edge from the new node to a randomly chosen original node
        new_edge = [int(new_nodes[i, 0]), int(np.random.choice(num_original_nodes))]
        edges = np.concatenate((edges, [new_edge]))

    return oversampled_features, edges




def split_arti(labels, c_train_num):
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    # cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)
    c_num_mat[:, 1] = 25
    c_num_mat[:, 2] = 55

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i, 0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i] + 25]
        test_idx = test_idx + c_idx[c_train_num[i] + 25:c_train_num[i] + 80]

    random.shuffle(train_idx)

    # ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    # c_num_mat = torch.LongTensor(c_num_mat)

    return train_idx, val_idx, test_idx, c_num_mat


def split_genuine(labels):
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    # cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num < 4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            c_num_mat[i, 0] = int(c_num / 4)
            c_num_mat[i, 1] = int(c_num / 4)
            c_num_mat[i, 2] = int(c_num / 2)

        train_idx = train_idx + c_idx[:c_num_mat[i, 0]]

        val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = test_idx + c_idx[
                              c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    random.shuffle(train_idx)

    # ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    # c_num_mat = torch.LongTensor(c_num_mat)

    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item() + 1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            # ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i, j, edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    # print class-wise performance
    '''
    for i in range(labels.max()+1):

        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)

        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    # ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro',
                                  multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:, 1].detach(), average='macro')

    macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
    print(str(pre) + ' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score, macro_F))

    return


def arrray_to_sparse(adj):
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    adj = torch.sparse.FloatTensor(
        torch.LongTensor(np.vstack((adj.row, adj.col))),
        torch.FloatTensor(adj.data),
        torch.Size(adj.shape)
    )
    return adj


def src_upsample(adj, adj_in, adj_out, features, labels, idx_train, portion=1.0, im_class_num=1):
    c_largest = labels.max().item()  # 最大类号
    adj_back = adj.to_dense()
    adj_in = arrray_to_sparse(adj_in)
    adj_back_in = adj_in.to_dense()
    adj_out = arrray_to_sparse(adj_out)
    adj_back_out = adj_out.to_dense()

    chosen = None

    # ipdb.set_trace()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]  # label=1时的索引号
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

            num = int(new_chosen.shape[0] * portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    new_adj_in = adj_back_in.new(torch.Size((adj_back_in.shape[0] + add_num, adj_back_in.shape[0] + add_num)))
    new_adj_in[:adj_back_in.shape[0], :adj_back_in.shape[0]] = adj_back_in[:, :]
    new_adj_in[adj_back_in.shape[0]:, :adj_back_in.shape[0]] = adj_back_in[chosen, :]
    new_adj_in[:adj_back_in.shape[0], adj_back_in.shape[0]:] = adj_back_in[:, chosen]
    new_adj_in[adj_back_in.shape[0]:, adj_back_in.shape[0]:] = adj_back_in[chosen, :][:, chosen]

    new_adj_out = adj_back_out.new(torch.Size((adj_back_out.shape[0] + add_num, adj_back_out.shape[0] + add_num)))
    new_adj_out[:adj_back_out.shape[0], :adj_back_out.shape[0]] = adj_back_out[:, :]
    new_adj_out[adj_back_out.shape[0]:, :adj_back_out.shape[0]] = adj_back_out[chosen, :]
    new_adj_out[:adj_back_out.shape[0], adj_back_out.shape[0]:] = adj_back_out[:, chosen]
    new_adj_out[adj_back_out.shape[0]:, adj_back_out.shape[0]:] = adj_back_out[chosen, :][:, chosen]

    # ipdb.set_trace()
    features_append = deepcopy(features[chosen, :])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()
    adj_in = new_adj_in.to_sparse()
    adj_out = new_adj_out.to_sparse()

    return adj, adj_in, adj_out, features, labels, idx_train


def src_smote(adj, adj_in, adj_out, features, labels, idx_train, portion=1.0, im_class_num=1):
    c_largest = labels.max().item()  # 标签最大值

    # 将邻接矩阵转换为稠密表示
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    adj_back_in = adj_in
    adj_back_out = adj_out

    # 计算平均样本数量
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for i in range(im_class_num):
        # 获取标签最多的类别的样本的索引
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        if portion == 0:  # refers to even distribution
            # 控制重采样的比例为平均样本数量除以样本数量
            c_portion = int(avg_number / new_chosen.shape[0])
            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion

        else:
            # 控制重采样的比例为给定的portion
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            # 获取被选中样本的特征向量并计算距离矩阵
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            # 根据距离矩阵找到每个样本的最近邻居索引
            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()  # 随机插值比例

            # 在样本和其最近邻居之间进行插值
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        new_chosen = new_chosen[:num]

        if num > 0:
            # 获取对应数量的样本特征向量并计算距离矩阵
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)  # 计算样本之间的距离矩阵

            idx_neighbor = distance.argmin(axis=-1)  # 根据距离矩阵找到每个样本的最近邻居索引

            interp_place = random.random()
            # 在样本和其最近邻居之间进行插值
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            # new_features = torch.cat((new_features, embed),0)

    # 扩展邻接矩阵的大小并将新采样的样本与原邻接矩阵的连接关系添加到新的邻接矩阵中
    # 没有生成新的连接关系，而是用的以前的连接关系进行扩展
    add_num = chosen.shape[0]  # 新采样样本的数量
    new_adj = adj_back.new(
        torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num)))  # 创建一个新的邻接矩阵，大小为原邻接矩阵大小加上新采样样本的数量
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]  # 将原邻接矩阵的值复制到新邻接矩阵的相应位置

    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]  # 将新采样样本与原邻接矩阵的连接关系添加到新邻接矩阵的上半部分
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]  # 将新采样样本与原邻接矩阵的连接关系添加到新邻接矩阵的左半部分
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]  # 将新采样样本与原邻接矩阵的连接关系添加到新邻接矩阵的右下角

    new_adj_in = adj_back_in.new(torch.Size((adj_back_in.shape[0] + add_num, adj_back_in.shape[0] + add_num)))
    new_adj_in[:adj_back_in.shape[0], :adj_back_in.shape[0]] = adj_back_in[:, :]
    new_adj_in[adj_back_in.shape[0]:, :adj_back_in.shape[0]] = adj_back_in[chosen, :]
    new_adj_in[:adj_back_in.shape[0], adj_back_in.shape[0]:] = adj_back_in[:, chosen]
    new_adj_in[adj_back_in.shape[0]:, adj_back_in.shape[0]:] = adj_back_in[chosen, :][:, chosen]

    new_adj_out = adj_back_out.new(torch.Size((adj_back_out.shape[0] + add_num, adj_back_out.shape[0] + add_num)))
    new_adj_out[:adj_back_out.shape[0], :adj_back_out.shape[0]] = adj_back_out[:, :]
    new_adj_out[adj_back_out.shape[0]:, :adj_back_out.shape[0]] = adj_back_out[chosen, :]
    new_adj_out[:adj_back_out.shape[0], adj_back_out.shape[0]:] = adj_back_out[:, chosen]
    new_adj_out[adj_back_out.shape[0]:, adj_back_out.shape[0]:] = adj_back_out[chosen, :][:, chosen]

    # ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    # 将新采样的样本特征和标签添加到原特征和标签中，并更新训练节点索引
    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    # 将扩展后的邻接矩阵转换为稀疏表示
    adj = new_adj.to_sparse()
    adj_in = new_adj_in.to_sparse()
    adj_out = new_adj_out.to_sparse()

    return adj, adj_in, adj_out, features, labels, idx_train


def recon_upsample(embed, labels, idx_train, adj=None, adj_out=None, portion=1.0, im_class_num=2):
    c_largest = labels.max().item()  # 最大类别
    avg_number = int(idx_train.shape[0] / (c_largest + 1))  # 每个类别平均数
    # ipdb.set_trace()
    adj_new = None
    adj_new_out = None

    # 决定增加哪几类的样本
    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]  # 找出类型为0/1的索引
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        # 决定增加的样本的数量
        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)
            # if adj_out is not None:
            #     if adj_new_out is None:
            #         adj_new_out = adj_out.new(torch.clamp_(adj_out[chosen,:] + adj_out[idx_neighbor,:], min=0.0, max = 1.0))
            #     else:
            #         temp = adj_out.new(torch.clamp_(adj_out[chosen,:] + adj_out[idx_neighbor,:], min=0.0, max = 1.0))
            #         adj_new_out = torch.cat((adj_new_out, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        # 创建一个新的矩阵，并将原来的邻接矩阵复制到左上角
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        # 将 adj_new 的转置复制到 new_adj 的右下角
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]
        # if adj_out is not None:
        #     add_num = adj_new_out.shape[0]
        #     # 创建一个新的矩阵，并将原来的邻接矩阵复制到左上角
        #     new_adj_out = adj_out.new(torch.Size((adj_out.shape[0] + add_num, adj_out.shape[0] + add_num))).fill_(0.0)
        #     new_adj_out[:adj_out.shape[0], :adj_out.shape[0]] = adj_out[:, :]
        #     # 将 adj_new 的转置复制到 new_adj 的右下角
        #     new_adj_out[adj_out.shape[0]:, :adj_out.shape[0]] = adj_new_out[:, :]
        #     new_adj_out[:adj_out.shape[0], adj_out.shape[0]:] = torch.transpose(adj_new_out, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt, adj_mask=None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2

    neg_weight = edge_num / (total_num - edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=30, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss