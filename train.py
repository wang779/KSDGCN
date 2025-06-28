# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
from tqdm import tqdm
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import DatesetReader

from models import AFFGCN

import numpy as np


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # 从.pkl文件读取数据（没有的话就从raw生成.pkl）
        dataset = DatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
        self.train_data_loader = BucketIterator(data=dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        self.model = opt.model_class(dataset.embedding_matrix, opt).to(opt.device)
        # 打印参数信息
        self._print_args()
        self.global_f1 = 0.
        # 如果GPU可用，则打印已分配的CUDA内存
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    # 打印模型训练参数的信息，包括可训练参数和不可训练参数的数量，以及所有训练参数的详细信息。
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # 重新初始化模型参数。根据参数的形状，使用指定的初始化器对参数进行重新初始化。
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # 训练模型。在每个epoch中，对训练数据进行迭代，计算损失并更新模型参数。
    # 同时，计算并打印训练过程中的准确率、损失以及测试集上的准确率、F1分数等指标。
    # 如果测试集上的F1分数提升，则保存模型。在连续若干次epoch内未出现F1分数提升时，触发提前停止训练的条件。
    def _train(self, criterion, optimizer):
        # 最大测试准确率、F1值、精确率、召回率
        max_test_acc = 0
        max_test_f1 = 0
        max_test_precision = 0
        max_test_recall = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(tqdm(self.train_data_loader)):
            
                # 每走一个batch，更新全局步数
                global_step += 1
                
                self.model.train()
                optimizer.zero_grad()
                # 将输入数据和标签移动到指定的设备上。
                # 将数据中除了context列的内容移到设备上
                '''
                return { \
                'context': batch_context, \
                'context_indices': torch.tensor(batch_context_indices), \
                'dependency_graph': torch.tensor(numpy.array(batch_dependency_graph)), \
                'sentic_graph': torch.tensor(numpy.array(batch_sentic_graph)), \
                'label': torch.tensor(batch_label),
            }
                '''
                inputs = [sample_batched[col].to(self.opt.device) if col != 'context' else sample_batched[col] for col
                          in self.opt.inputs_cols]
                targets = sample_batched['label'].to(self.opt.device)
                # 将输入数据传递给模型进行前向传播，得到输出。
                outputs = self.model(inputs, 'train')
                # 计算损失
                loss = criterion(outputs, targets)
                # 反向传播
                loss.backward()
                # 优化参数
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    # 统计训练过程中的正确预测数量和总样本数量。
                    # 计算训练准确率。
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    # 调用方法评估模型在测试集上的性能，得到测试准确率、F1值、精确率和召回率。
                    print("*******Testing********")
                    test_acc, test_f1, test_precision, test_recall, f1_weighted = self._evaluate_acc_f1()
                    if test_f1 > max_test_f1:
                        # 更新最佳测试准确率、召回率、精确率和F1值。
                        # - 设置增加标志为increase_flag为True，表示发生了性能提升。
                        max_test_acc = test_acc
                        max_test_recall = test_recall
                        max_test_precision = test_precision
                        increase_flag = True
                        max_test_f1 = test_f1
                        max_test_f1_weighted = f1_weighted
                        # 如果设置了保存模型，并且当前F1值超过了历史最佳F1值，则保存模型参数。
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            # torch.save(self.model.state_dict(), 'state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl')
                            # print('>>> best model saved.')
                    print("==============================epoch: ", epoch)
                    print("==============================i_batch: ", i_batch)
                    print("max test f1: ", max_test_f1, "max test acc: ", max_test_acc, "max test pre: ",
                          max_test_precision, "max test recall: ", max_test_recall, "f1_weighted: ", max_test_f1_weighted)
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc,
                                                                                                     test_acc, test_f1))
            if increase_flag == False:
            
                #  如果没有发生性能提升，则将未提升的次数加1，并判断是否达到早停的条件。
                #     - 如果达到早停条件，则提前结束训练。
                
                continue_not_increase += 1
                print("=================continue_not_increase: ", continue_not_increase)
                if continue_not_increase >= self.opt.estop:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        # 返回最佳测试准确率、F1值、精确率和召回率，用于记录训练结果。
        return max_test_acc, max_test_f1, max_test_precision, max_test_recall, max_test_f1_weighted

    # 评估模型在测试集上的准确率和F1分数。将模型切换到评估模式，对测试数据进行迭代，计算准确率、F1分数、
    # 宏平均精确率和宏平均召回率，并返回这些指标。
    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(self.test_data_loader, desc = "Test:")):
            
                #print("###Testing t_batch: ", t_batch)
                t_inputs = [t_sample_batched[col].to(opt.device) if col != 'context' else t_sample_batched[col] for col
                            in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(opt.device)
                t_outputs = self.model(t_inputs, 'test')

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                              average='macro')
        f1_weighted = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                              average='weighted')
        precision_macro = metrics.precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                                                  labels=[0, 1], average='macro')
        recall_macro = metrics.recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                                            average='macro')
        return test_acc, f1, precision_macro, recall_macro, f1_weighted

    # 在测试集上进行预测并将结果保存到文件中。
    # 将模型切换到评估模式，对测试数据进行迭代，预测每个样本的标签，并将真实标签和预测标签写入文件
    def predict(self):
        self.model.eval()
        # 初始化变量，用于存储所有测试样本的真实标签和模型预测输出
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                # 将输入数据和真实标签移动到指定的设备上。
                t_inputs = [t_sample_batched[col].to(opt.device) if col != 'context' else t_sample_batched[col] for col
                            in self.opt.inputs_cols]
                t_targets = t_sample_batched['label'].to(opt.device)
                # 使用模型进行推理，得到预测输出。
                t_outputs = self.model(t_inputs, 'test')
                # 将当前批次的真实标签和模型预测输出添加到之前的所有样本中。
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        # 打开一个文件，准备写入模型的预测结果。
        with open(self.opt.model_name + '_' + "pre.txt", 'w', encoding='utf-8') as fout:
            # 将真实标签和模型预测输出转换为NumPy数组，并将其移动到CPU上，然后转换为Python列表。
            t_targets_all = list(np.array(t_targets_all.cpu()))
            t_outputs_all = list(np.array(torch.argmax(t_outputs_all, -1).cpu()))

            # 将每个样本的真实标签和模型预测输出写入文件中，使用制表符分隔，并在每个样本之间换行。
            for x, y in zip(t_targets_all, t_outputs_all):
                fout.write(str(x) + '\t' + str(y) + '\n')

    # 执行模型训练和评估的整个流程。在该方法中，初始化损失函数和优化器，并迭代多次进行模型训练和评估。
    # 在每次训练和评估完成后，将结果写入日志文件，并计算平均的最大测试准确率和最大测试F1分数。
    def run(self, repeats=3):
        # Loss and Optimizer
        # 交叉熵损失函数来定义criterion，用于计算模型预测值和真实标签之间的损失。
        criterion = nn.CrossEntropyLoss()
        # 从模型参数中过滤出需要梯度更新的参数，并初始化优化器，使用了一些配置参数如学习率(`learning_rate`)和L2正则化系数(`l2reg`)。
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.txt', 'a', encoding='utf-8')
        f_out.write('****************选择算法：major******************\n')
        f_out.flush()

        # 初始化用于存储每次重复训练后的测试准确率、F1值、精确率和召回率的列表，并初始化平均测试准确率和F1值。
        test_acc = []
        test_f1 = []
        test_pre = []
        test_recall = []
        test_f1_weighted = []
        max_test_acc_avg = 0
        max_test_f1_avg = 0
        
        for i in range(repeats):
            # 开始执行重复次数的循环，以进行多次训练和评估。
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1))
            # 重置模型参数的初始状态。
            self._reset_params()
            # 调用`_train`方法进行模型训练，并返回测试集上的最大准确率、F1值、精确率和召回率。
            max_test_acc, max_test_f1, max_test_precision, max_test_recall, max_test_f1_weighted = self._train(criterion, optimizer)
            test_acc.append(max_test_acc)
            test_f1.append(max_test_f1)
            test_pre.append(max_test_precision)
            test_recall.append(max_test_recall)
            test_f1_weighted.append(max_test_f1_weighted)
            print('max_test_acc: {0}     max_test_f1: {1}   max_test_pre: {2}  max_test_recall: {3}  max_test_f1_weighted:{4}'.format(max_test_acc, max_test_f1, max_test_precision, max_test_recall, max_test_f1_weighted))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}   max_test_pre: {2}  max_test_recall: {3}  max_test_f1_weighted:{4}\n'.format(max_test_acc, max_test_f1, max_test_precision, max_test_recall, max_test_f1_weighted))
            f_out.flush()
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        for i, j, k, p in zip(test_acc, test_f1, test_pre, test_recall):
            print('max_test_acc: {0}     max_test_f1: {1}   max_test_pre: {2}  max_test_recall: {3}'.format(i, j, k, p))
        print(self.opt.log_info)
        self.predict()
        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='affgcn', type=str)
    parser.add_argument('--dataset', default='reddic', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    # 源代码默认batch 8
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--log_info', default="None", type=str)
    parser.add_argument('--estop', default=5, type=int)
    # 默认是1
    parser.add_argument('--repeat', default=5, type=int)
    parser.add_argument("--max_seq_len", default=-1, type=int)
    opt = parser.parse_args()

    model_classes = {
        'affgcn': AFFGCN,
    }
    input_colses = {
        'affgcn': ['context', 'context_indices',
                   'dependency_graph',
                   'sentic_graph'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    # opt.device = torch.device('cpu')
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("opt.device:", opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run(opt.repeat)
