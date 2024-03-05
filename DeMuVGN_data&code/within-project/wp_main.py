import csv
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import models
import utils
from utils import load_data

# biggnn
parser = utils.get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)


def train(epoch, idx_train, adj_in, adj_out):
    encoder.train()
    classifier.train()
    optimizer_cls.zero_grad()
    optimizer_en.zero_grad()

    # 生成嵌入
    edge_vec = []
    if args.needSmote == 'on':
        embed = encoder(features, edge_vec, adj_in.to_dense(), adj_out.to_dense(), args)
    else:
        embed = encoder(features, edge_vec, adj_in, adj_out, args)

    embed = embed[0, :, :]
    output = classifier(embed, adj)  # 分类
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

    acc_train = utils.accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])
    global min_loss  # min_loss 定义为全局变量
    if loss_val < min_loss:
        min_loss = loss_val
        # print("save model, min_loss=", min_loss.item())
        all_states = {"encoder": encoder.state_dict(), "class": classifier.state_dict()}
        torch.save(all_states, save_path + "model.pth")
    prediction_add_to_df(output.argsort()[:, :2][:, 1], train_df)
    prediction_add_to_df(output.argsort()[:, :2][:, 1], val_df)

    optimizer_cls.step()
    optimizer_en.step()

    # print('Epoch: {:05d}'.format(epoch + 1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def test(adj_in, adj_out):
    encoder = models.BiGGNN(args).to(device)
    # if not os.path.exists(save_path + 'model.pth'):
    #     os.makedirs(save_path + 'model.pth')
    encoder.load_state_dict(torch.load(save_path + "model.pth")['encoder'])
    classifier = models.Classifier(nembed=args.hidden_size,
                                   nhid=args.hidden_size,
                                   nclass=2,
                                   dropout=args.dropout).to(device)
    classifier.load_state_dict(torch.load(save_path + "model.pth")['class'])
    encoder.eval()
    classifier.eval()
    edge_vec = []
    if args.needSmote == "on":
        features.to(device)
        adj_in = adj_in.to_dense().to(device)
        adj_out = adj_out.to_dense().to(device)
    else:
        features.to(device)
        adj_in = adj_in.to(device)
        adj_out = adj_out.to(device)
    embed = encoder(features, edge_vec, adj_in, adj_out, args).to(device)
    embed = embed[0, :, :]
    output = classifier(embed, adj)
    output_norm = F.softmax(output, dim=1)
    output_int = output.argsort()[:, :2]

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])
    auc = roc_auc_score(labels[idx_test], output_norm[idx_test][:, 1])
    if auc < 0.5: auc = 1 - auc
    recall = recall_score(labels[idx_test], output_int[idx_test][:, 1])
    brier = brier_score_loss(labels[idx_test], output_norm[idx_test][:, 1])
    # pop_RF = CE_score(labels_array[idx_test], output_int[idx_test][:, 1])
    pf = fp_rate(labels[idx_test], output_int[idx_test][:, 1])
    precision = precision_score(labels[idx_test], output_int[idx_test][:, 1], zero_division=1)

    prediction_add_to_df(output_int[:, 1], test_df)

    print_string = "Test set results:" + \
                   "loss= {:.4f}".format(loss_test.item()) + ", " + \
                   "accuracy= {:.4f}".format(acc_test.item()) + ", " + \
                   "auc= {:.4f}".format(auc.item()) + ", " + \
                   "recall= {:.4f}".format(recall.item()) + ", " + \
                   "brier= {:.4f}".format(brier.item()) + ", " + \
                   "fp={:.4f}".format(pf) + ", " + \
                   "precision={:.4f}".format(precision)

    result_metrics = loss_test.item(), acc_test.item(), auc, recall.item(), brier, pf, precision
    return result_metrics, embed, output_int[:, 1]


# 将预测值添加到df的最后一列
def prediction_add_to_df(output, df_data):
    # output应该为一维的tensor张量，df为需要添加的df格式的内容
    output_list = output.tolist()
    prediction_values = []
    for index in df_data.index:
        # 检查index是否在output_list中
        if index in range(len(output_list)):
            # 获取对应位置的output值，并添加到预测值列表中
            prediction_values.append(output_list[index])
        else:
            # 如果index不在output_list中，则将值设为null或其他缺失值标记
            prediction_values.append(None)
    # 将预测值列表添加为新列"prediction"
    df_data.loc[:, 'prediction'] = prediction_values
    # print(df_data)


def fp_rate(labels, output_int):
    # 初始化 TP、FN、FP 和 TN 的值为 0
    FP = 0
    TN = 0

    # 遍历每一行，统计 TP、FN、FP 和 TN 的值
    for i in range(len(output_int)):
        if output_int[i] == 1 and labels[i] == 0:
            FP += 1
        elif output_int[i] == 0 and labels[i] == 0:
            TN += 1

    # 计算 FP/(TN+FP)
    fp_rate = FP / (TN + FP)
    return fp_rate


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()

    torch.save(saved_content,
               'checkpoint/{}/{}_{}_{}_{}.pth'.format(args.dataset, args.setting, epoch, args.opt_new_G,
                                                      args.im_ratio))

    return


def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset, filename),
                                map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    # print("successfully loaded: " + filename)

    return


def add_attention(embed, embed2):
    embed_add = np.concatenate((embed[np.newaxis, :, :], embed2[np.newaxis, :, :]), axis=0)
    embed_add = torch.from_numpy(embed_add)
    q = k = v = embed_add
    alpha = torch.bmm(q, k.transpose(-1, -2))
    alpha = F.softmax(alpha, dim=-1)
    out = torch.bmm(alpha, v)
    embed, embed2 = np.split(out, 2, axis=0)
    embed_con = np.concatenate((embed, embed2), axis=2)
    embed_con = np.squeeze(embed_con)
    embed_con = torch.from_numpy(embed_con)
    return embed_con


def get_last_row(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1]
    return None


def write_output(file_load, filename, output):
    if type(output[0]) == float:
        loss_test = output[0]
        acc_test = output[1]
        auc = output[2]
        recall = output[3]
        brier = output[4]
        pf = output[5]
        precision = output[6]
    else:
        loss_test = output[0].item()
        acc_test = output[1].item()
        auc = output[2].item()
        recall = output[3].item()
        brier = output[4].item()
        pf = output[5].item()
        precision = output[6].item()

    if filename == 0:
        numbers = [loss_test, acc_test, auc, recall, brier, pf, precision]
    else:
        numbers = [filename, loss_test, acc_test, auc, recall, brier, pf, precision]
        # print(numbers)
    with open(file_load, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(numbers)


def copy_last_line(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip().split(',')
        return last_line[1:]  # 从第二列开始返回数据


def process_csv_files(folder_path):
    summary_data = []  # 用于保存每个文件的最后一行数据

    # 遍历文件夹中的CSV文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            last_line_data = copy_last_line(file_path)
            summary_data.append([filename] + last_line_data)

    # 将数据写入summary.csv文件
    with open(folder_path + "summary.csv", 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerows(summary_data)


def clear_folder(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    # for root, dirs, files in os.walk(folder_path):
    #     # 删除所有文件
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         os.remove(file_path)
    #     # 删除所有子文件夹
    #     for dir in dirs:
    #         dir_path = os.path.join(root, dir)
    #         os.remove(dir_path)


def norm(features):
    # 创建 StandardScaler 对象
    scaler = StandardScaler()
    # 将特征数据转换为 ndarray 格式
    features_ndarray = features.numpy()
    # 对特征进行归一化
    normalized_features_ndarray = scaler.fit_transform(features_ndarray)
    # 将归一化后的数据转换为 tensor
    normalized_features = torch.tensor(normalized_features_ndarray)
    return normalized_features


def writeFeatures(matrix, writeFile):
    matrix = matrix.cpu().detach().numpy()
    with open(writeFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        if matrix.ndim > 1:
            for row in matrix:
                writer.writerow(row)
        else:
            for row in matrix:
                writer.writerow([row])


if __name__ == '__main__':
    forward = "../datasets/"
    args.needSmote = 'on'
    files_and_names = [
        (forward + "activemq/activemq-5.0.0/", "activemq-5.0.0(1)"),
        (forward + "activemq/activemq-5.1.0/", "activemq-5.1.0(1)"),
        (forward + "activemq/activemq-5.2.0/", "activemq-5.2.0(1)"),
        (forward + "activemq/activemq-5.3.0/", "activemq-5.3.0(1)"),
        (forward + "activemq/activemq-5.8.0/", "activemq-5.8.0(1)"),
        (forward + "camel/camel-1.4.0/", "camel-1.4.0(1)"),
        (forward + "camel/camel-2.9.0/", "camel-2.9.0(1)"),
        (forward + "camel/camel-2.10.0/", "camel-2.10.0(1)"),
        (forward + "camel/camel-2.11.0/", "camel-2.11.0(1)"),
        (forward + "groovy/groovy-1.5.7/", "groovy-1.5.7(1)"),
        (forward + "hive/hive-0.9.0/", "hive-0.9.0(1)"),
        (forward + "jruby/jruby-1.1/", "jruby-1.1(1)"),
        (forward + "jruby/jruby-1.4.0/", "jruby-1.4.0(1)"),
        (forward + "jruby/jruby-1.5.0/", "jruby-1.5.0(1)"),
        (forward + "lucene/lucene-2.3.0/", "lucene-2.3.0(1)"),
        (forward + "lucene/lucene-2.9.0/", "lucene-2.9.0(1)")
    ]

    forward_path1 = forward + "score/wp/"

    for file, name in files_and_names:
        # 选文件
        for choose in ['file_dependencies', 'final_weight_edge', 'add_weight_edge']:
            if choose == 'file_dependencies':
                choName = 'CDG'
            elif choose == 'final_weight_edge':
                choName = 'DDG'
            else:
                choName = 'MPDG'
            print(name, choName)
            forward_path = forward_path1 + "/" + name + "/"
            save_path = forward_path + "/" + choName + "/"

            fileload2 = save_path + name + "_" + choName + "_summary" + ".csv"
            fileload3 = save_path + "final_summary.csv"
            df = pd.DataFrame(columns=['auc', 'recall', 'brier', 'pf', 'f1'])
            df2 = pd.DataFrame(columns=['file', 'auc', 'recall', 'brier', 'pf', 'f1'])

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if not os.path.exists(fileload2):
                df2.to_csv(fileload2, index=False, mode='w')
            if not os.path.exists(fileload3):
                df2.to_csv(fileload3, index=False, mode='w')

            # try:
            times = 50

            fileload = save_path + choName + "(-)" + name + ".csv"  # portion_new
            df_file = pd.read_csv(file + name + ".csv")

            auc_values = []
            recall_values = []
            brier_values = []
            pf_values = []
            precision_values = []
            for t in tqdm(range(times)):

                adj, features, labels, idx_train, idx_val, idx_test, adj_in, adj_out = load_data(file, name, choose)
                features = norm(features)
                train_df = df_file.iloc[idx_train]
                val_df = df_file.iloc[idx_val]
                test_df = df_file.iloc[idx_test]

                im_class_num = 1
                args.hidden_size = features.size(-1)
                if args.needSmote == "on":
                    print("SMOTE")
                    # todo: need to check
                    adj, adj_in, adj_out, features, labels, idx_train = utils.src_smote(adj, adj_in, adj_out, features,
                                                                                        labels, idx_train,
                                                                                        portion=args.up_scale,
                                                                                        im_class_num=im_class_num)

                encoder = models.BiGGNN(args)

                classifier = models.Classifier(nembed=args.hidden_size,
                                               nhid=args.hidden_size,
                                               nclass=labels.max().item() + 1,
                                               dropout=args.dropout)
                optimizer_cls = optim.Adam(classifier.parameters(),
                                           lr=args.lr, weight_decay=args.weight_decay)

                optimizer_en = optim.Adam(encoder.parameters(),
                                          lr=args.lr, weight_decay=args.weight_decay)

                if args.cuda:
                    encoder = encoder.to(device)
                    classifier = classifier.to(device)
                    features = features.to(device)
                    adj = adj.to(device)
                    labels = labels.to(device)
                    idx_train = idx_train.to(device)
                    idx_val = idx_val.to(device)
                    idx_test = idx_test.to(device)
                    adj_in = adj_in.to(device)
                    adj_out = adj_out.to(device)
                # Train model
                if args.load is not None:
                    load_model(args.load)
                sum_vector = None

                t_total = time.time()
                # 创建输出表格
                args.batch_size = 16
                min_loss = 10000
                # 训练/测试
                # arly_stopping = EarlyStopping(save_path)
                for epoch in range(args.epochs):
                    train(epoch, idx_train, adj_in, adj_out)
                result_metrics, out_features, output_int = test(adj_in, adj_out)
                loss, acc, auc, recall, brier, pf, precision = result_metrics
                auc_values.append(auc)
                recall_values.append(recall)
                brier_values.append(brier)
                pf_values.append(pf)
                precision_values.append(precision)

                writeFeaturesFile = save_path + name + "_" + choName + "_feat" + str(t) + ".csv"
                writePredictionFile = save_path + name + "_" + choName + "_pred" + str(t) + ".csv"
                writeFeatures(out_features, writeFeaturesFile)
                writeFeatures(output_int, writePredictionFile)

                if sum_vector is None:
                    sum_vector = result_metrics
                else:
                    sum_vector = tuple(
                        torch.add(sum_vector[i], result_metrics[i]) for i in range(len(result_metrics)))
            avg_auc = sum(auc_values) / len(auc_values)
            avg_recall = sum(recall_values) / len(recall_values)
            avg_brier = sum(brier_values) / len(brier_values)
            avg_pf = sum(pf_values) / len(pf_values)
            avg_precision = sum(precision_values) / len(precision_values)
            avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

            average_metrics = (avg_auc, avg_recall, avg_brier, avg_pf, avg_f1)

            # average_vector = tuple(value / times for value in sum_vector)
            # print("***average_vector***", average_vector)
            write_output(fileload3, choName + "(-)" + name, average_metrics)
