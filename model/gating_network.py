import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


class MAGNN(nn.Module):
    def __init__(self, num_users, num_items, model_args, left_number, device):
        super(MAGNN, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        T = self.args.T
        dims = self.args.d
        heads = self.args.h
        units = self.args.m
        step = self.args.step

        self.train_left = left_number
        self.test_left = left_number + T

        # add gnn
        self.gnn = GNN(dims, L, T, step, device).to(device)

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims).to(device)
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)
        self.train_code_embedding = get_sinusoid_encoding_table(self.train_left, dims).to(device)
        # self.code_embedding = pos_encoding(L, dims).to(device)
        self.train_ones_production = torch.ones(1, self.train_left).to(device)
        self.test_code_embedding = get_sinusoid_encoding_table(self.test_left, dims).to(device)
        # self.code_embedding = pos_encoding(L, dims).to(device)
        self.test_ones_production = torch.ones(1, self.test_left).to(device)

        self.feature_gate_item = nn.Linear(dims, dims).to(device)
        self.feature_gate_user = nn.Linear(dims, dims).to(device)

        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(
            torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_user = Variable(torch.zeros(dims, L).type(
            torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(
            self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(
            self.instance_gate_user)

        self.short_W1 = Parameter(torch.Tensor(2 * dims, dims)).to(device)
        self.long_W1 = Parameter(torch.Tensor(dims, dims)).to(device)
        self.long_W2 = Parameter(torch.Tensor(dims, dims)).to(device)
        self.long_W3 = Parameter(torch.Tensor(dims, heads)).to(device)
        self.memory_K = Parameter(torch.Tensor(dims, units)).to(device)
        self.memory_V = Parameter(torch.Tensor(dims, units)).to(device)
        self.fusion_W1 = Parameter(torch.Tensor(dims, dims)).to(device)
        self.fusion_W2 = Parameter(torch.Tensor(dims, dims)).to(device)
        self.fusion_W3 = Parameter(torch.Tensor(dims, dims)).to(device)
        self.short_W1 = torch.nn.init.xavier_uniform_(self.short_W1)
        self.long_W1 = torch.nn.init.xavier_uniform_(self.long_W1)
        self.long_W2 = torch.nn.init.xavier_uniform_(self.long_W2)
        self.long_W3 = torch.nn.init.xavier_uniform_(self.long_W3)
        self.memory_K = torch.nn.init.xavier_uniform_(self.memory_K)
        self.memory_V = torch.nn.init.xavier_uniform_(self.memory_V)
        self.fusion_W1 = torch.nn.init.xavier_uniform_(self.fusion_W1)
        self.fusion_W2 = torch.nn.init.xavier_uniform_(self.fusion_W2)
        self.fusion_W3 = torch.nn.init.xavier_uniform_(self.fusion_W3)
        self.item_item_W = Parameter(torch.Tensor(dims, dims)).to(device)
        self.item_item_W = torch.nn.init.xavier_uniform_(self.item_item_W)

        self.attention_item1 = nn.Linear(dims, dims).to(device)
        self.attention_item2 = nn.Linear(dims, heads).to(device)

        self.W2 = nn.Embedding(num_items, dims, padding_idx=0).to(device)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=0).to(device)

        # weight initialization
        self.user_embeddings.weight.data.normal_(
            0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(
            0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, item_seq, item_left_seq, user_ids, items_to_predict, A, for_pred=False):
        item_embs = self.item_embeddings(item_seq)  # [4096,5,128]
        # print(item_embs.shape)
        item_left_embs = self.item_embeddings(item_left_seq)
        # print(item_left_embs.shape)
        # item_embs = self.gnn(A, item_embs)
        user_emb = self.user_embeddings(user_ids)  # [4096,128]

        if for_pred:
            short_embs = self.gnn(A, item_embs, True)  # [4096,5,128]
        else:
            short_embs = self.gnn(A, item_embs, False)  # [4096,5,128]

        short_arg_embs = torch.mean(short_embs, dim=1)  # [4096,128]

        # # 用户的短期兴趣
        # item_agg = torch.matmul(A, item_embs)  # nums_item * dims
        # short_hidden_cat = torch.cat(
        #     (item_agg, item_embs), dim=2)  # nums_item * 2 * dims [4096,5,128*2]
        # short_hidden = torch.matmul(
        #     short_hidden_cat, self.short_W1)  # dims * nums [4096,5,128]
        # short_embs = torch.tanh(short_hidden)  # nums_item * dims [4096,5,128]
        # # short_arg_embs = torch.mean(short_embs, dim=1)  # [4096,128]
        # # short_embs = torch.tanh(short_hidden)  # nums_item * dims [4096,5,128]
        # # short_arg_embs = torch.mean(short_embs, dim=1)  # [4096,128]
        # item_agg = torch.matmul(A, short_embs)  # nums_item * dims
        # short_hidden_cat = torch.cat(
        #     (item_agg, item_embs), dim=2)  # nums_item * 2 * dims [4096,5,128*2]
        # short_hidden = torch.matmul(
        #     short_hidden_cat, self.short_W1)  # dims * nums [4096,5,128]
        # short_embs = torch.tanh(short_hidden)  # nums_item * dims [4096,5,128]
       
        # 用户的长期兴趣
        if for_pred:
        	long_hidden = item_left_embs + self.test_code_embedding  # [4096,5,128]
        	long_user_hidden = torch.matmul(user_emb, self.long_W2).unsqueeze(2) * self.test_ones_production  # [4096,128,5] 
        else:       	
        	long_hidden = item_left_embs + self.train_code_embedding  # [4096,5,128]
        	long_user_hidden = torch.matmul(user_emb, self.long_W2).unsqueeze(2) * self.train_ones_production  # [4096,128,5]
        long_hidden_head = torch.tanh(torch.matmul(
            long_hidden, self.long_W1) + long_user_hidden.transpose(1, 2))  # [4096,5,128]
        long_hidden_head = torch.softmax(
            torch.matmul(long_hidden_head, self.long_W3), dim=2)  # [4096,5,20]

        # long_matrix = torch.tanh(long_hidden_head)
        matrix_z = torch.bmm(
            long_hidden.permute(0, 2, 1), long_hidden_head)  # [4096,128,20]
        long_query = torch.mean(torch.tanh(matrix_z), dim=2)  # [4096,128]

        # memory units
        memory_hidden = torch.softmax(torch.matmul(
            long_query, self.memory_K), dim=1)  # batch_size * units
        memory_hidden = torch.matmul(
            memory_hidden, self.memory_V.t())  # bitch_size * dims
        long_embs = long_query + memory_hidden  # [4096,128]

        # fusion interest
        gate_unit = torch.sigmoid(torch.matmul(short_arg_embs, self.fusion_W1) + torch.matmul(
            long_embs, self.fusion_W2) + torch.matmul(user_emb, self.fusion_W3))  # [4096,128]
        fusion_embs = gate_unit * short_arg_embs + \
            (1 - gate_unit) * long_embs  # [4096,128]

        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()

            # MF
            res = user_emb.mm(w2.t()) + b2

            # union-level
            res += fusion_embs.mm(w2.t())

            # item-item product
            # item_item_embs = torch.matmul(item_embs, self.item_item_W)  # [4096, 5, 35119]
            # rel_score = torch.matmul(item_item_embs, w2.t().unsqueeze(0))  # [4096, 5, 35119]
            rel_score = torch.matmul(item_embs, w2.t().unsqueeze(0))
            rel_score = torch.mean(rel_score, dim=1)
            res += rel_score
        else:
            # MF
            res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()

            # union-level
            res += torch.bmm(fusion_embs.unsqueeze(1),
                             w2.permute(0, 2, 1)).squeeze()

            # item-item product
            # item_item_embs = torch.matmul(item_embs, self.item_item_W)
            # rel_score = item_item_embs.bmm(w2.permute(0, 2, 1))  # [4096, 5, 6]
            rel_score = item_embs.bmm(w2.permute(0, 2, 1))
            rel_score = torch.mean(rel_score, dim=1)
            # print(rel_score.shape)
            res += rel_score

        return res


def pos_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

    Returns:
    Tensor with shape [length, hidden_size]
    """
    position = torch.FloatTensor(torch.range(0, length - 1))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) / (torch.FloatTensor(num_timescales) - 1))
    inv_timescales = min_timescale * \
        torch.exp(torch.FloatTensor(torch.range(0, num_timescales - 1))
                  * -log_timescale_increment)
    scaled_time = torch.unsqueeze(
        position, 1) * torch.unsqueeze(inv_timescales, 0)
    signal = torch.cat(
        (torch.sin(scaled_time), torch.cos(scaled_time)), axis=1)
    return signal


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class GNN(nn.Module):
    def __init__(self, dims_item, L, T, step, device):
        super(GNN, self).__init__()
        self.L = L
        self.T = T
        self.step = step
        self.hidden_size = dims_item
        self.input_size = dims_item * 2

        self.W1 = Parameter(torch.Tensor(
            self.input_size, self.hidden_size)).to(device)
        self.W1 = torch.nn.init.xavier_uniform_(self.W1)

    def GNNCell(self, A, hidden, for_pred=False):
        input_in1 = torch.matmul(A, hidden)
        input_in_item1 = torch.cat((input_in1, hidden), dim=2)

        # no b have item
        item_hidden1 = torch.matmul(input_in_item1, self.W1)
        item_embs1 = item_hidden1

        item_embs = torch.tanh(item_embs1) 
      
        return item_embs

    def forward(self, A, hidden, for_pred):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, for_pred)
        return hidden
