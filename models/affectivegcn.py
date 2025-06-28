# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.bert import get_bert_output, get_bert_output1
import EP.get_sentence_line as gsl   # 获取数据集种每个句子的行号
import EP.statistic as s   # 获取每个句子的实体数量和实体
import EP.get_entpair_entdis
from models.inconsistency import Scaled_Dot_Product_Attention_pos, Scaled_Dot_Product_Attention_neg

l = [[[0.0 for i in range(600)]]]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 定义了一个简单的图卷积层
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # print("adj.shape:",adj.shape)
        # print("hidden.shape:",hidden.shape)
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class AFFGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        # 包括词嵌入层、双向LSTM层、多个图卷积层、全连接层等。
        super(AFFGCN, self).__init__()
        self.opt = opt
        # embedding_matrix参数是词嵌入矩阵，用于初始化词嵌入层的权重。
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc3 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc4 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc5 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc6 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.fc1 = nn.Linear(600, 300)
        self.text_embed_dropout = nn.Dropout(0.5)
        self.kg_pos = Scaled_Dot_Product_Attention_pos()
        self.kg_neg = Scaled_Dot_Product_Attention_neg()

    def forward(self, inputs, dataset_type):
        # 输入inputs解包为四个部分：datas（数据）、text_indices（文本索引）、adj（依赖图）、sentic_adj（情感图）。
        datas, text_indices, adj, sentic_adj = inputs
        # 在这里得到datas里每句话对应的行号
        lines = {}  # 存放datas的行号
        # `dataset_type`确定使用的数据集类型，然后根据数据集类型选择相应的句子行号字典，将数据对应到行号。
        if dataset_type == 'train':
            for data in datas:
                # 每个data（句子）是第几个（行号）
                lines[data] = gsl.train_sen_line[data]
        else:
            for data in datas:
                lines[data] = gsl.test_sen_line[data]
        # 计算文本长度，并将文本索引转换为词嵌入向量。
        # 对词嵌入向量进行Dropout处理。
        # 使用双向LSTM处理文本序列，得到文本的输出表示text_out。
        text_len = torch.sum(text_indices != 0, dim=-1)
        #print("#############datas#################")
        #print(datas)
        #print("#############text_indices############")
        #print(text_indices)
        #print("##############text_len################")
        #print(text_len)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        # text_out依赖图的节点嵌入
        text_out, (_, _) = self.text_lstm(text, text_len)

        # 使用BERT模型处理数据，得到增广列表`augu_list`。
        augu_list = get_bert_output(datas, dataset_type)
        # 调用get_bert_output1得到CLS
        # cls = get_bert_output1(datas)

        # 将文本输出表示和文本长度转换为CPU上的NumPy数组，并转换为Python列表。
        text_out_list = text_out.cpu().detach().numpy().tolist()
        text_len_list = text_len.cpu().detach().numpy().tolist()
        # 最长（句子长度+增广词数）
        # 计算文本输出和增广列表中每个样本的最大长度。
        max = 0
        augu_len = []
        for augu in augu_list:
            if augu == l:
                augu_len.append(0)
            else:
                augu_len.append(len(augu))

        for m in range(len(augu_len)):
            temp = text_len_list[m] + augu_len[m]
            if temp > max:
                max = temp
        # 补0，将lstm的文本输出向量扩展到最大长度，并用零填充。
        for p in range(len(text_out_list)):
            a = [0.0] * 600
            for q in range(max - len(text_out_list[p])):
                text_out_list[p].append(a)

        # 属性加进去，将增广列表中的词向量添加到文本输出向量中。
        for i in range(len(augu_list)):
            position = text_len_list[i]
            for j in range(augu_len[i]):
                text_out_list[i][position] = augu_list[i][j][0]
                position = position + 1
        # 情感图
        text_out1 = torch.tensor(text_out_list).to(device)

        # 情感图
        x = F.relu(self.gc2(text_out1, sentic_adj))
        x = F.relu(self.gc4(x, sentic_adj))
        x = F.relu(self.gc6(x, sentic_adj))

        # 依赖图
        y = F.relu(self.gc1(text_out, adj))
        y = F.relu(self.gc3(y, adj))

        # 计算依赖信息和文本输出的注意力权重，并使用注意力加权平均得到文本表示`y`。
        alpha_mat = torch.matmul(y, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        y = torch.matmul(alpha, text_out).squeeze(1).unsqueeze(0)

        # 计算情感信息和增强文本输出的注意力权重，并使用注意力加权平均得到情感表示x。
        alpha_mat1 = torch.matmul(x, text_out1.transpose(1, 2))
        alpha1 = F.softmax(alpha_mat1.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha1, text_out1).squeeze(1).unsqueeze(0)  # 1 batsize 600

        # 将情感表示和文本表示拼接起来，然后计算平均值，并通过全连接层`fc1`进行转换得到`temp`
        temp = torch.cat([x, y], dim=0)
        temp = torch.mean(temp, dim=0, keepdim=True).squeeze(0)
        temp = self.fc1(temp)  # 1 bs 300

        # 进入内容知识不一致模块，搞个函数放这
        # 根据数据集类型选择不同的实体数量字典和实体对的字典。
        if dataset_type == 'train':
            # dic的key是句子号，值是这个句子有几个实体
            dic = s.ent_num_per_train_sentence
            # 每个句子实体对的嵌入向量和距离（距离最大的3个）
            ep_emb = EP.get_entpair_entdis.train_line_epemb_dic
            ep_dis = EP.get_entpair_entdis.train_line_epdistance_dic
        else:
            dic = s.ent_num_per_test_sentence
            ep_emb = EP.get_entpair_entdis.test_line_epemb_dic
            ep_dis = EP.get_entpair_entdis.test_line_epdistance_dic

        # 调用`content_know_inconsistency`函数计算内容知识不一致模块的表示`k`。
        # 文本数据、两个图经过GCN后融合的信息、每个句子对应的号、每个句子的实体数量、每个句子实体对的嵌入向量、每个句子实体对的距离
        k = self.content_know_inconsistency(datas, temp, lines, dic, ep_emb, ep_dis)  # bs 1 600
        k = k.squeeze(1).unsqueeze(0).to(device)  # 1 bs 600

        # 进入最后的输出模块，融合前三部分的向量
        z = torch.cat([x, y, k], dim=0)
        z = torch.mean(z, dim=0, keepdim=True).squeeze(0)

        output = self.fc(z)
        return output

    # 文本数据、两个图经过GCN后融合的信息、每个句子对应的号、每个句子的实体数量、每个句子实体对的嵌入向量、每个句子实体对的距离
    def content_know_inconsistency(self, datas, xy_concat, lines, dic, line_epemb, line_epdis):
        # dic参数传每句话实体数量那个字典
        # 如果句子中没有实体或者只有一个实体
        # 如果句子中有实体对，进入内容知识不一致模块，搞个函数放这

        """
        首先遍历输入的数据 datas。
        对于每个数据，获取其所对应的行索引 line。
        如果该行中没有实体或者只有一个实体，则将内容知识设置为一个全零的张量。
        如果该行中存在实体对，则从 line_epemb 和 line_epdis 中获取实体对的嵌入向量和距离。
        使用这些嵌入向量和距离，调用 kg_pos 和 kg_neg 函数生成内容知识。
        将生成的内容知识添加到 f_kg 列表中。
        将所有的内容知识张量拼接起来，返回结果。
        """
        f_kg = []
        for data in datas:
            line = lines[data]
            if dic[str(line)] == 1 or dic[str(line)] == 0:
                kg_temp = torch.zeros(600).unsqueeze(0).unsqueeze(0)
            else:

                epemb = torch.cat(line_epemb[str(line)], 0).unsqueeze(0).to(torch.float)  # 1 n 300
                epdis = torch.tensor(line_epdis[str(line)]).to(torch.float)

                # 符号机制
                # Q, K, V, scale, kg_sim 实体对距离
                kg_pos_temp = self.kg_pos(xy_concat[0], epemb, epemb, 1 / math.sqrt(300), epdis)
                kg_neg_temp = self.kg_neg(xy_concat[0], epemb, epemb, 1 / math.sqrt(300), epdis)
                kg_temp = torch.cat([kg_pos_temp, kg_neg_temp], dim=2).cpu()  # 1 1 600
            f_kg.append(kg_temp)
        f = torch.cat(f_kg, 0)
        return f  # batsize 1 600

    # def content_know_inconsistency(self, datas, cls, lines, dic, line_epemb, line_epdis):
    #     # dic参数传每句话实体数量那个字典
    #     # 如果句子中没有实体或者只有一个实体
    #     # 如果句子中有实体对，进入内容知识不一致模块，搞个函数放这
    #     f_kg = []
    #     for data in datas:
    #         a_temp = []
    #         b_pos = []
    #         b_neg = []
    #         all_pos = 0
    #         all_neg = 0
    #         f_pos = torch.zeros(300)
    #         f_neg = torch.zeros(300)
    #         line = lines[data]
    #         if dic[str(line)] == 1 or dic[str(line)] == 0:
    #             f_kg.append(torch.zeros(600).cpu().detach().numpy().tolist())
    #         else:
    #             for epemb in line_epemb[str(line)]:
    #
    #                 temp = torch.matmul(cls[datas.index(data)].unsqueeze(0), epemb.to(torch.float).unsqueeze(0).transpose(1,0).to(device))
    #                 temp = temp / math.sqrt(300)
    #                 a_temp.append(temp)
    #             a_temp = torch.tensor(a_temp)
    #             print(a_temp.shape)
    #             a_pos = F.softmax(a_temp, dim=0).cpu().detach().numpy().tolist()
    #             a_neg = (-F.softmax(-a_temp, dim=0)).cpu().detach().numpy().tolist()
    #             for i in range(len(a_pos)):
    #                 all_pos = all_pos + a_pos[i] * line_epdis[str(line)][i]
    #                 all_neg = all_neg + a_neg[i] * line_epdis[str(line)][i]
    #             for i in range(len(a_pos)):
    #                 b_pos.append(a_pos[i] * line_epdis[str(line)][i] / all_pos)
    #                 b_neg.append(a_neg[i] * line_epdis[str(line)][i] / all_neg)
    #             for i in range(len(b_pos)):
    #                 f_pos = torch.add(f_pos.to(device), torch.as_tensor(b_pos[i]).to(device) * torch.as_tensor(
    #                     line_epemb[str(line)][i]).to(device))
    #                 f_neg = torch.add(f_neg.to(device), torch.as_tensor(b_neg[i]).to(device) * torch.as_tensor(
    #                     line_epemb[str(line)][i]).to(device))
    #             f_kg.append(torch.cat([f_pos, f_neg], dim=0).cpu().detach().numpy().tolist())
    #     return f_kg
