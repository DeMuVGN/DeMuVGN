import os
import csv
import numpy as np
from sklearn import metrics
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
from utils import EarlyStopping

# biggnn
parser = utils.get_parser()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

torch.cuda.device(1)
print(torch.cuda.is_available())

print("device:", device)


def train(epoch, encoder, train_data, test_data, classifier, idx_val):
    optimizer_en = optim.Adam(encoder.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    if claName == '0':
        optimizer_cls = optim.Adam(classifier.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        optimizer_cls.zero_grad()
        classifier.train()

    # t = time.time()
    encoder.train()

    optimizer_en.zero_grad()

    # for train_file in train_list:
    # print("train_file", train_file)
    adj = train_data[0]
    features = train_data[1]
    labels = train_data[2]
    adj_in = train_data[3]
    adj_out = train_data[4]

    adj_t = test_data[0]
    features_t = test_data[1]
    labels_t = test_data[2]
    adj_in_t = test_data[3]
    adj_out_t = test_data[4]

    if args.cuda:
        encoder = encoder.cuda()
        if claName == '0':
            classifier = classifier.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        adj_in = adj_in.cuda()
        adj_out = adj_out.cuda()
        features_t = features_t.cuda()
        adj_t = adj_t.cuda()
        labels_t = labels_t.cuda()
        adj_in_t = adj_in_t.cuda()
        adj_out_t = adj_out_t.cuda()

    adj_in = adj_in.to_dense()
    adj_out = adj_out.to_dense()
    # adj_in_t = adj_in_t.to_dense()
    # adj_out_t = adj_out_t.to_dense()
    # 生成嵌入
    edge_vec = []
    embed = encoder(features, edge_vec, adj_in, adj_out, args)[0, :, :]
    embed_t = encoder(features_t, edge_vec, adj_in_t, adj_out_t, args)[0, :, :]
    labels_new = labels
    global min_loss
    if claName in ['RF', 'NB', 'LR', 'SVM', 'XGBoost']:
        classifier.fit(embed, labels)
    if claName == '0':
        output = classifier(embed, adj)  # 分类

        if args.setting == 'reweight':
            weight = features.new((labels.max().item() + 1)).fill_(1)
            weight[-im_class_num:] = 1 + args.up_scale
            loss_train = F.cross_entropy(output, labels_new, weight=weight)
        else:
            loss_train = F.cross_entropy(output, labels_new)

        acc_train = utils.accuracy(output, labels_new)
        loss = loss_train
        loss.backward()

        output_t = classifier(embed_t, adj_t)
        loss_val = F.cross_entropy(output_t[idx_val], labels_t[idx_val])
        if loss_val < min_loss:
            min_loss = loss_val
            # print("save model", min_loss)
            torch.save(encoder.state_dict(), save_path + "model.pth")

        optimizer_cls.step()
    optimizer_en.step()
    # print('Epoch: {:05d}'.format(epoch + 1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'loss_rec: {:.4f}'.format(loss_rec.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def test(test_data, encoder, classifier, idx_test):
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
    embed = encoder(features, edge_vec, adj_in, adj_out, args)
    embed = embed[0, :, :]

    if claName in ['RF', 'NB', 'LR', 'SVM', 'XGBoost']:
        y_pred = classifier.predict(embed)
        y_prob = classifier.predict_proba(embed)
        auc = roc_auc_score(labels, y_prob[:, 1])
        recall = recall_score(labels, y_pred)
        brier = brier_score_loss(labels, y_prob[:, 1])
        pf = fp_rate(labels, y_pred)
        f1 = f1_score(labels, y_pred, zero_division=1)
    else:
        output = classifier(embed, adj)
        output_norm = F.softmax(output, dim=1)
        output_int = output.argsort()[:, :2]

        auc = roc_auc_score(labels[idx_test], output_norm[idx_test][:, 1])
        recall = recall_score(labels[idx_test], output_int[idx_test][:, 1])
        brier = brier_score_loss(labels[idx_test], output_norm[idx_test][:, 1])
        pf = fp_rate(labels[idx_test], output_int[idx_test][:, 1])
        f1 = f1_score(labels[idx_test], output_int[idx_test][:, 1], zero_division=1)

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
    forward = "../datasets/"
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
    for file_te, name_te in files_and_names:
        lines_num = get_row_count(file_te + name_te + ".csv")
        idx = list(range(lines_num))
        idx_val = idx[:int(lines_num * 0.5)]
        idx_test = idx[int(lines_num * 0.5):lines_num - 1]

        claName = '0'
        for choose in ['file_dependencies', 'final_weight_edge', 'add_weight_edge']:
            save_path = forward + 'score/cp_score' + choose + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            fileload2 = save_path + name_te + "_summary.csv"

            # 仅包括所需的列
            df = pd.DataFrame(columns=['auc', 'recall', 'brier', 'pf', 'f1'])
            df2 = pd.DataFrame(columns=['file', 'auc', 'recall', 'brier', 'pf', 'f1'])
            df2.to_csv(fileload2, index=False)

            train_list = [item for item in files_and_names if item[1].split("-")[0] not in file_te]

            im_class_num = 1
            args.hidden_size = 66
            times = 20
            encoder = models.BiGGNN(args)

            classifier = models.Classifier(nembed=args.hidden_size,
                                           nhid=args.hidden_size,
                                           nclass=2,
                                           dropout=args.dropout)
            early_stopping = EarlyStopping(save_path)
            for train_file in train_list:
                print("train_file:", train_file[1], ", test_file:", name_te, choose)
                fileload = save_path + train_file[1] + "-" + name_te + ".csv"
                df.to_csv(fileload, index=False)
                min_loss = 10000
                file_tr, name_tr = train_file
                train_data = load_cp_data(file_tr, name_tr, choose, mood='train', portion=args.portion)
                test_data = load_cp_data(file_te, "/" + name_te, choose)

                for t in tqdm(range(times)):
                    args.batch_size = 32
                    for epoch in range(args.epochs):
                        train(epoch, encoder, train_data, test_data, classifier, idx_val)

                        output2 = test(test_data, encoder, classifier, idx_test)
                        write_output(fileload2, train_file[1] + "-" + name_te, output2)
                        print("(", t, ")file write: ", fileload2)