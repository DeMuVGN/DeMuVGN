# 这是服务器版的跨项目

import time
import random
from torch.optim import lr_scheduler

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import brier_score_loss, accuracy_score, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import models
import utils
from utils import load_cp_data

# biggnn
parser = utils.get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device:", device)

def train(idx_train, train_data, args, encoder, classifier, optimizer_cls, optimizer_en):
    encoder.train()
    classifier.train()
    optimizer_cls.zero_grad()
    optimizer_en.zero_grad()

    adj = train_data[0].to(device)
    features = train_data[1].to(device)
    labels = train_data[2].to(device)
    adj_in = train_data[3].to(device)
    adj_out = train_data[4].to(device)

    # 生成嵌入
    edge_vec = []
    if args.needSmote == 'on':
        embed = encoder(features, edge_vec, adj_in.to_dense(), adj_out.to_dense(), args, device).to(device)
    else:
        embed = encoder(features, edge_vec, adj_in, adj_out, args, device).to(device)

    embed = embed[0, :, :]
    output = classifier(embed, adj)  # 分类
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

    acc_train = utils.accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()

    optimizer_cls.step()
    optimizer_en.step()

    return output


def validate(idx_val, output, labels, encoder, classifier):
    encoder.eval()
    classifier.eval()

    output = output.to(device)
    labels = labels.to(device)
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])

    # global min_loss  # min_loss 定义为全局变量
    # if loss_val < min_loss:
    #     min_loss = loss_val
    #     # print("save model, min_loss=", min_loss.item())
    #     all_states = {"encoder": encoder.state_dict(), "class": classifier.state_dict()}
    #     torch.save(all_states, save_path + "model.pth")

    # train_df = prediction_add_to_df(output.argsort()[:, :2][:, 1], train_df)
    # val_df = prediction_add_to_df(output.argsort()[:, :2][:, 1], val_df)

    # return train_df, val_df, loss_val.item(), acc_val
    return acc_val, loss_val


def test(test_data, best_save_path):

    adj = test_data[0].to(device)
    features = test_data[1].to(device)
    labels = test_data[2].to(device)
    adj_in = test_data[3].to(device)
    adj_out = test_data[4].to(device)

    encoder_save = torch.load(os.path.join(best_save_path, 'best_encoder.pth'))
    classifier_save = torch.load(os.path.join(best_save_path, 'best_classifier.pth'))

    # 求此时的hidden_layer
    hidden_layers = []
    i = 0
    while f'layers.{i}.weight' in classifier_save:
        i = i + 1
    j = 1
    while j < i:
        hidden_layers.append(len(classifier_save[f'layers.{j}.weight'][0]))
        j = j+1

    args.hidden_size = len(encoder_save['linear_max.weight'])

    encoder = models.BiGGNN(args, device).to(device)

    classifier = models.Classifier(nembed=args.hidden_size,
                                   hidden_layers=hidden_layers,
                                   nclass=2,
                                   dropout=args.dropout).to(device)
    encoder.load_state_dict(torch.load(os.path.join(best_save_path, 'best_encoder.pth')))
    classifier.load_state_dict(torch.load(os.path.join(best_save_path, 'best_classifier.pth')))

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
    embed = encoder(features, edge_vec, adj_in, adj_out, args, device).to(device)
    embed = embed[0, :, :]
    output = classifier(embed, adj)
    output_norm = F.softmax(output, dim=1)
    output_int = output.argsort()[:, :2]

    loss_test = F.cross_entropy(output, labels)
    acc_test = utils.accuracy(output, labels)
    auc = roc_auc_score(labels, output_norm[:, 1])
    if auc < 0.5: auc = 1 - auc
    recall = recall_score(labels, output_int[:, 1])
    brier = brier_score_loss(labels, output_norm[:, 1])
    # pop_RF = CE_score(labels_array, output_int[:, 1])
    pf = fp_rate(labels, output_int[:, 1])
    precision = precision_score(labels, output_int[:, 1], zero_division=1)
    f1 = 2 * precision * recall / (precision + recall)

    # prediction_add_to_df(output_int[:, 1], test_df)

    print_string = "Test set results:" + \
                   "loss= {:.4f}".format(loss_test.item()) + ", " + \
                   "accuracy= {:.4f}".format(acc_test.item()) + ", " + \
                   "auc= {:.4f}".format(auc.item()) + ", " + \
                   "recall= {:.4f}".format(recall.item()) + ", " + \
                   "brier= {:.4f}".format(brier.item()) + ", " + \
                   "fp={:.4f}".format(pf) + ", " + \
                   "f1={:.4f}".format(f1)
    print(print_string)

    result_metrics = loss_test.item(), acc_test.item(), auc, recall.item(), brier, pf, f1
    return result_metrics, embed, output_int[:, 1]


def test2(test_data, encoder, classifier):
    encoder.load_state_dict(torch.load(save_path + 'model.pth'))

    adj = test_data[0]
    features = test_data[1]
    labels = test_data[2]
    adj_in = test_data[3]
    adj_out = test_data[4]

    if args.cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        adj_in = adj_in.cuda()
        adj_out = adj_out.cuda()

    encoder.eval()
    classifier.eval()

    edge_vec = []
    embed = encoder(features, edge_vec, adj_in, adj_out, args, device)
    embed = embed[0, :, :]

    # if claName in ['RF', 'NB', 'LR', 'SVM', 'XGBoost']:
    #     y_pred = classifier.predict(embed)
    #     y_prob = classifier.predict_proba(embed)
    #     auc = roc_auc_score(labels, y_prob[:, 1])
    #     recall = recall_score(labels, y_pred)
    #     brier = brier_score_loss(labels, y_prob[:, 1])
    #     pf = fp_rate(labels, y_pred)
    #     f1 = f1_score(labels, y_pred, zero_division=1)
    # else:
    output = classifier(embed, adj)
    output_norm = F.softmax(output, dim=1)
    output_int = output.argsort()[:, :2]

    auc = roc_auc_score(labels, output_norm[:, 1])
    recall = recall_score(labels, output_int[:, 1])
    brier = brier_score_loss(labels, output_norm[:, 1])
    pf = fp_rate(labels, output_int[:, 1])
    f1 = f1_score(labels, output_int[:, 1], zero_division=1)

    return auc, recall, brier, pf, f1


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

    print("successfully loaded: " + filename)

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


def clear_folder(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 删除所有文件
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        # 删除所有子文件夹
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)


def write_output(file_load, filename, output):
    # 由于我们现在只返回五个指标，我们将按照这个顺序接收它们
    auc = output[0]
    recall = output[1]
    brier = output[2]
    pf = output[3]
    f1 = output[4]

    if filename == 0:
        # 如果 filename 是 0，我们只写入指标值
        numbers = [auc, recall, brier, pf, f1]
    else:
        # 否则，我们在前面添加文件名
        numbers = [filename, auc, recall, brier, pf, f1]

    # 写入 CSV 文件
    with open(file_load, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(numbers)


# get file's lines_number
def get_row_count(file_path):
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count


if __name__ == '__main__':
    if device.type == 'cpu':
        forward = "../../datasets/"
    else:
        forward = "../../data/"
    args.needSmote = 'on'

    forward_path1 = os.path.join(forward, "score/cv_final/")
    times = 40
    train_list = [
        ("wicket/wicket-1.3.0Beta2/", "wicket-1.3.0-beta2(1)"),
        ("hbase/hbase-0.95.0/", "hbase-0.95.0(1)"),
        # ("activemq/activemq-5.0.0/", "activemq-5.0.0(1)"),
        # ("activemq/activemq-5.1.0/", "activemq-5.1.0(1)"),
        # ("activemq/activemq-5.2.0/", "activemq-5.2.0(1)"),
        # ("activemq/activemq-5.3.0/", "activemq-5.3.0(1)"),
        # ("camel/camel-1.4.0/", "camel-1.4.0(1)"),
        # ("camel/camel-2.9.0/", "camel-2.9.0(1)"),
        # ("camel/camel-2.10.0/", "camel-2.10.0(1)"),
        # ("jruby/jruby-1.1/", "jruby-1.1(1)"),
        # ("jruby/jruby-1.4.0/", "jruby-1.4.0(1)"),
        # ("lucene/lucene-2.3.0/", "lucene-2.3.0(1)"),
    ]
    test_list = [
        ("wicket/wicket-1.5.3/", "wicket-1.5.3(1)"),
        ("hbase/hbase-0.95.2/", "hbase-0.95.2(1)"),
        # ("activemq/activemq-5.1.0/", "activemq-5.1.0(1)"),
        # ("activemq/activemq-5.2.0/", "activemq-5.2.0(1)"),
        # ("activemq/activemq-5.3.0/", "activemq-5.3.0(1)"),
        # ("activemq/activemq-5.8.0/", "activemq-5.8.0(1)"),
        # ("camel/camel-2.9.0/", "camel-2.9.0(1)"),
        # ("camel/camel-2.10.0/", "camel-2.10.0(1)"),
        # ("camel/camel-2.11.0/", "camel-2.11.0(1)"),
        # ("jruby/jruby-1.4.0/", "jruby-1.4.0(1)"),
        # ("jruby/jruby-1.5.0/", "jruby-1.5.0(1)"),
        # ("lucene/lucene-2.9.0/", "lucene-2.9.0(1)")
    ]
    for choose in ['developer_edge']:
        if choose == 'file_edge':
            choName = 'CDG'
        elif choose == 'developer_edge':
            choName = 'DDG'
        else:
            choName = 'MPDG'

        num_combinations = 20  # 你想要尝试的参数组合数量

        param_dist = {
            'lr': [0.01, 0.001, 0.0001, 0.0005],
            'hidden_size': [16, 32, 64],
            'hops': [1, 2, 3, 4],
            'hidden_layers': [[32, 16]],
            'portion':[ 0.5]
            }
        for i in range(len(train_list)):
            train_file = train_list[i]
            test_file = test_list[i]

            train_file_path, train_file_name = train_file
            train_short_name = train_file_name.replace('(1)', '')
            test_file_path, test_file_name = test_file

            test_short_name = test_file_name.replace('(1)', '')
            print(train_short_name, '->', test_short_name)
            save_path = os.path.join(forward_path1, test_short_name)

            param_combinations = []  # 参数组合

            for _ in range(num_combinations):
                params = {k: v() if callable(v) else random.choice(v) for k, v in param_dist.items()}
                param_combinations.append(params)

            for t in tqdm(range(times)):
                best_acc = 0.0  # 跟踪最佳模型的准确率
                best_params = None  # 存储最佳参数组合
                best_model = None  # 存储最佳模型

                best_save_path = os.path.join(forward_path1, 'best_model', train_short_name)
                if not os.path.exists(best_save_path):
                    os.makedirs(best_save_path)

                for i, params in enumerate(param_combinations):
                    lr = params['lr']
                    hidden_size = params['hidden_size']
                    hops = params['hops']
                    hidden_layers = params['hidden_layers']
                    portion  = params['portion']

                    args.lr = lr
                    args.hidden_size = hidden_size
                    args.hops = hops
                    args.portion = portion

                    train_data, idx_train, idx_val = load_cp_data(forward + train_file_path, train_file_name,
                                                                  choose, mood='train', portion=args.portion)

                    args.batch_size = 1
                    encoder = models.BiGGNN(args, device)

                    classifier = models.Classifier(nembed=hidden_size,
                                                   hidden_layers=hidden_layers,
                                                   nclass=2,
                                                   dropout=args.dropout)

                    optimizer_cls = optim.Adam(classifier.parameters(),
                                               lr=args.lr, weight_decay=args.weight_decay)

                    optimizer_en = optim.Adam(encoder.parameters(),
                                              lr=args.lr, weight_decay=args.weight_decay)

                    if args.cuda:
                        encoder = encoder.to(device)
                        classifier = classifier.to(device)
                        idx_train = idx_train.to(device)
                        idx_val = idx_val.to(device)
                    # Train model
                    if args.load is not None:
                        load_model(args.load)


                    t_total = time.time()
                    # 创建输出表格
                    args.batch_size = 1
                    # min_loss = 10000
                    # 训练/测试
                    # early_stopping = EarlyStopping(save_path)
                    best_val_acc = 0

                    # 早停
                    best_loss_val = float('inf')  # 初始化最低验证损失
                    epochs_no_improve = 0  # 自从上次损失下降以来的epoch数
                    n_epochs_stop = 10  # 设定的早停阈值

                    args.epochs = 200
                    # 余弦动态lr
                    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_en, T_max=100)

                    for epoch in range(args.epochs):
                        output = train(idx_train, train_data, args, encoder, classifier, optimizer_cls, optimizer_en)
                        acc_val, loss_val = validate(idx_val, output, train_data[2], encoder, classifier)

                        if loss_val < best_loss_val:
                            best_loss_val = loss_val
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1

                        if acc_val > best_val_acc:
                            best_val_acc = acc_val

                        scheduler.step()

                    if best_val_acc > best_acc:
                        best_acc = best_val_acc
                        best_params = params
                        best_model = (encoder, classifier)  # 或者仅保存模型参数

                    best_encoder, best_classifier = best_model

                    with open(os.path.join(best_save_path, 'best_params.csv'), mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([train_short_name, best_params])

                    torch.save(best_encoder.state_dict(), os.path.join(best_save_path, 'best_encoder.pth'))
                    torch.save(best_classifier.state_dict(), os.path.join(best_save_path, 'best_classifier.pth'))

                sum_vector = None


                fileload2 = os.path.join(forward_path1, test_short_name + "_summary" + ".csv")
                fileload3 = os.path.join(forward_path1, "final_summary.csv")

                # 保存详细结果
                save_result_path = os.path.join(forward_path1, test_short_name)
                if not os.path.exists(save_result_path):
                    os.makedirs(save_result_path)
                print('详细结果保存：', save_result_path)

                file_path = os.path.join(save_result_path, train_short_name + '(-)' + test_short_name + '.csv')
                if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
                    with open(file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['auc', 'recall', 'brier', 'pf', 'f1'])

                df2 = pd.DataFrame(columns=['file', 'auc', 'recall', 'brier', 'pf', 'f1'])

                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                test_data = load_cp_data(forward + test_file_path, "/" + test_file_name, choose)

                result_metrics, out_features, output_int = test(test_data, best_save_path)
                loss, acc, auc, recall, brier, pf, f1 = result_metrics

                # 写入详细结果
                with open(save_result_path + '/' + train_short_name + '(-)' + test_short_name + '.csv', mode='a',
                          newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([auc, recall, brier, pf, f1])

