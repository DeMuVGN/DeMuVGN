import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
import ipdb

# --------------
### layers###
# --------------

# GCN layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # for 3_D batch, need a loop!!!

        if self.bias is not None:
            return output + self.bias
        else:
            return output


# Multihead attention layer
class MultiHead(Module):  # currently, allowed for only one sample each time. As no padding mask is required.
    def __init__(
            self,
            input_dim,
            num_heads,
            kdim=None,
            vdim=None,
            embed_dim=128,  # should equal num_heads*head dim
            v_embed_dim=None,
            dropout=0.1,
            bias=True,
    ):
        super(MultiHead, self).__init__()
        self.input_dim = input_dim
        self.kdim = kdim if kdim is not None else input_dim
        self.vdim = vdim if vdim is not None else input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.v_embed_dim = v_embed_dim if v_embed_dim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert self.v_embed_dim % num_heads == 0, "v_embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.input_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.v_embed_dim, bias=bias)

        self.out_proj = nn.Linear(self.v_embed_dim, self.v_embed_dim // self.num_heads, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)
        else:
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)

        nn.init.normal_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.q_proj.bias, 0.)

    def forward(
            self,
            query,
            key,
            value,
            need_weights: bool = False,
            need_head_weights: bool = False,
    ):
        """Input shape: Time x Batch x Channel
        Args:
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        batch_num, node_num, input_dim = query.size()

        assert key is not None and value is not None

        # project input
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * self.scaling

        # compute attention
        q = q.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2, -3).contiguous().view(
            batch_num * self.num_heads, node_num, self.head_dim)
        k = k.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2, -3).contiguous().view(
            batch_num * self.num_heads, node_num, self.head_dim)
        v = v.view(batch_num, node_num, self.num_heads, self.vdim).transpose(-2, -3).contiguous().view(
            batch_num * self.num_heads, node_num, self.vdim)
        attn_output_weights = torch.bmm(q, k.transpose(-1, -2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        # drop out
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        # collect output
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(batch_num, self.num_heads, node_num, self.vdim).transpose(-2,
                                                                                                 -3).contiguous().view(
            batch_num, node_num, self.v_embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_output_weights = attn_output_weights  # view: (batch_num, num_heads, node_num, node_num)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output


# Graphsage layer
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

        self.reset_parameters()

        # print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        # fuse info from neighbors. to be added:
        if adj.layout != torch.sparse_coo:
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (
                            adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            # print("spmm not implemented for batch training. Note!")

            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

        # perform conv
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


# GraphAT layers

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        if isinstance(adj, torch.sparse.FloatTensor):
            adj = adj.to_dense()

        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# --------------
### models ###
# --------------

# gcn_encode
class GCN_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


# sage model

class Sage_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Sage_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En2, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


# GAT model

class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, alpha=0.2, nheads=8):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))

        return x


class GAT_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, alpha=0.2, nheads=8):
        super(GAT_En2, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nembed)
        self.dropout = dropout

        self.attentions_2 = [GraphAttentionLayer(nembed, nembed, dropout=dropout, alpha=alpha, concat=True) for _ in
                             range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention2_{}'.format(i), attention)

        self.out_proj_2 = nn.Linear(nembed * nheads, nembed)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.normal_(self.out_proj_2.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj_2(x))
        return x


class GAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, alpha=0.2, nheads=8):
        super(GAT_Classifier, self).__init__()

        self.attentions = [GraphAttentionLayer(nembed, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nhid)

        self.dropout = dropout
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)
        nn.init.normal_(self.out_proj.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = self.mlp(x)

        return x


class Classifier(nn.Module):
    def __init__(self, nembed, hidden_layers, nclass, dropout):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # 输入层
        self.layers.append(nn.Linear(nembed, hidden_layers[0]))

        # 隐藏层
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        # 输出层
        self.layers.append(nn.Linear(hidden_layers[-1], nclass))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.1)

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层前不使用激活函数和dropout
                x = F.relu(x)
                x = self.dropout(x)
        return x


# 连接前用nhid=66
class Classifier_old(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp1 = nn.Linear(nhid, 32)
        self.mlp2 = nn.Linear(32, 16)
        self.mlp3 = nn.Linear(16, nclass)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight, std=0.1)
        nn.init.normal_(self.mlp2.weight, std=0.1)
        nn.init.normal_(self.mlp3.weight, std=0.1)

    def forward(self, x, adj):
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = F.relu(self.mlp2(x))
        x = self.dropout(x)
        x = self.mlp3(x)

        return x



class Classifier2(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier2, self).__init__()

        self.mlp1 = nn.Linear(nembed, 32)
        self.mlp2 = nn.Linear(32, 16)
        self.mlp3 = nn.Linear(16, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight, std=0.05)
        nn.init.normal_(self.mlp2.weight, std=0.05)
        nn.init.normal_(self.mlp3.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)

        return x


# 连接后用nhid*2=132
class Classifier2(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier2, self).__init__()

        self.mlp1 = nn.Linear(nhid * 2, 64)
        self.mlp2 = nn.Linear(64, 32)
        self.mlp3 = nn.Linear(32, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight, std=0.05)
        nn.init.normal_(self.mlp2.weight, std=0.05)
        nn.init.normal_(self.mlp3.weight, std=0.05)

    def forward(self, x, adj):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        return x


class Decoder(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))

        return adj_out


class BiGGNN(nn.Module):
    def __init__(self, config, device):
        super(BiGGNN, self).__init__()
        self.device = device
        hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        self.graph_direction = config.graph_direction  # 边的方向
        # self.graph_hops = config.hops
        self.graph_hops = config.hops
        self.word_dropout = config.dropout  # 丢弃词向量，一般在0~1之间
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        #####################################################
        self.static_graph_mp = GraphMessagePassing(config).to(device)
        self.static_gru_step = GRUStep(hidden_size, hidden_size).to(device)
        if self.graph_direction == 'all':
            self.static_gated_fusion = GatedFusion(hidden_size).to(device)
        self.graph_update = self.static_graph_update
        self.classifier = nn.Linear(hidden_size, 2).to(device)

        self.linear_fea = nn.Linear(in_features=66, out_features=hidden_size)

    def forward(self, node_feature, edge_vec, adj_in, adj_out, config, device):
        node_feature = self.linear_fea(node_feature)
        node_feature = node_feature.repeat(config.batch_size, 1, 1)
        adj_in = adj_in.repeat(config.batch_size, 1, 1)
        adj_out = adj_out.repeat(config.batch_size, 1, 1)
        node_feature = node_feature.to(device)
        adj_in = adj_in.to(device)
        adj_out = adj_out.to(device)
        node_state = self.graph_update(node_feature, edge_vec, adj_in, adj_out, device)
        return node_state

    def static_graph_update(self, node_feature, edge_vec, node2edge, edge2node, device):
        ''' Static graph update '''
        for _ in range(self.graph_hops):  # 重复所有跳f
            # 后方向的a
            bw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, node2edge,
                                                        edge2node)  # (num_nodes, dim)
            fw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, edge2node.transpose(1, 2),
                                                        node2edge.transpose(1, 2))
            agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state, device)
            node_feature = self.static_gru_step(node_feature, agg_state)
        return node_feature  # 节点特征矩阵

class GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config.hidden_size  # 128
        # 根据配置文件中指定的消息传递函数类型，初始化不同的参数。
        if config.message_function == 'edge_mm':  # 如果消息传递函数类型是 'edge_mm'
            # 初始化边权矩阵。这个矩阵的大小为边的类型数乘以隐藏层大小的平方。
            # 用于在消息传递中对不同类型的边应用不同的权重。
            self.edge_weight_tensor = torch.Tensor(config['num_edge_types'], hidden_size * hidden_size)
            self.edge_weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor))
            self.mp_func = self.msg_pass_edge_mm  # 设置消息传递函数为 msg_pass_edge_mm
        elif config.message_function == 'edge_network':  # 如果消息传递函数类型是 'edge_network'
            # 初始化边网络。这个网络的输入维度是边的嵌入维度，输出维度是隐藏层大小的平方。
            # 它可以将边的特征转化为影响节点状态更新的权重。
            self.edge_network = torch.Tensor(config['edge_embed_dim'], hidden_size, hidden_size)
            self.edge_network = nn.Parameter(nn.init.xavier_uniform_(self.edge_network))
            self.mp_func = self.msg_pass_edge_network  # 设置消息传递函数为 msg_pass_edge_network
        elif config.message_function == 'edge_pair':  # 如果消息传递函数类型是 'edge_pair'
            # 初始化一个线性层，用于处理边的特征。线性层的输入维度是边的嵌入维度，输出维度是隐藏层大小。
            self.linear_edge = nn.Linear(config['edge_embed_dim'], hidden_size, bias=False)
            self.mp_func = self.msg_pass  # 设置消息传递函数为 msg_pass
        elif config.message_function == 'no_edge':  # 如果消息传递函数类型是 'no_edge'
            # 在这种模式下，边的特征不会被考虑。所以，不需要初始化任何特定于边的参数。
            self.mp_func = self.msg_pass  # 设置消息传递函数为 msg_pass
        else:
            # 如果提供了一个未知的消息传递函数类型，则抛出一个运行时错误。
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    # 传递邻居信息，获取a
    # N' = bmm(a_in,bmm(a_out,fea))
    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)  # batch_size x num_edges x hidden_size
        if edge_vec is not None and self.config.message_function == 'edge_pair':
            node2edge_emb = node2edge_emb + self.linear_edge(edge_vec)
        agg_state = torch.bmm(edge2node, node2edge_emb)  # consider self-loop if preprocess not igore
        return agg_state

    # 通过边的权重对节点的状态进行加权平均得到新的节点状态
    def msg_pass_edge_mm(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)  # batch_size x num_edges x hidden_size
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = F.embedding(edge_vec[:, i], self.edge_weight_tensor).view(-1, node_state.size(-1),
                                                                                    node_state.size(
                                                                                        -1))  # batch_size x hidden_size x hidden_size
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1)  # batch_size x num_edges x hidden_size
        agg_state = torch.bmm(edge2node, new_node2edge_emb)
        return agg_state

    def msg_pass_edge_network(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)  # batch_size x num_edges x hidden_size
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):  # 对于每一个边
            edge_weight = torch.mm(edge_vec[:, i], self.edge_network.view(self.edge_network.size(0), -1)).view(
                (-1,) + self.edge_network.shape[-2:])
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1)  # batch_size x num_edges x hidden_size
        agg_state = torch.bmm(edge2node, new_node2edge_emb)  # 邻居消息的求和
        return agg_state


class GatedFusion(nn.Module):  # Fuse(a, b)
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        '''GatedFusion module'''
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)  # 4列矩阵线性变换为1列向量

    def forward(self, h_state, input, device):
        # 门控向量
        z = torch.sigmoid(self.fc_z(
            torch.cat([h_state, input, h_state * input, h_state - input], -1)))  # 这里cat将四列向量组成一个矩阵，-1表示按倒数第一维cat
        h_state = (1 - z) * h_state + z * input  # Fuse函数
        return h_state


class GRUStep(nn.Module):  # GRU模块
    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        # 维度把h_size + a_size变为h_size
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input):  # h_state = h, input = a
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))  # 更新门
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))  # 重置门
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))  # 隐藏状态
        h_state = (1 - z) * h_state + z * t  # 最终状态
        return h_state


# class GraphConvolution(torch.nn.Module):
#     def __init__(self, in_features, out_features, activation):
#         super(GraphConvolution, self).__init__()
#         self.linear = torch.nn.Linear(in_features, out_features)
#         self.activation = activation
#
#     def forward(self, x, adj):
#         out = torch.matmul(adj, self.linear(x))
#         out = self.activation(out)
#         return out


class EmbedFusion(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EmbedFusion, self).__init__()

        # 定义输入层
        self.input_layer = nn.Linear(input_size * 2, hidden_size)

        # 定义隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # 定义输出层
        self.output_layer = nn.Linear(hidden_size, input_size)

        # 定义激活函数
        self.activation = nn.ReLU()

    def forward(self, input1, input2):
        # 拼接输入向量
        x = torch.cat((input1, input2), dim=1)

        # 输入层
        x = self.input_layer(x)
        x = self.activation(x)

        # 隐藏层
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)

        # 输出层
        x = self.output_layer(x)

        return x


# 转换计算环境
def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x


class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    # 计算向量的余弦相似度
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # 对比损失函数
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)  # 温度因子,将相似度矩阵映射到一个较大的空间，从而增加相似度矩阵中不同样本之间的差异性
        refl_sim = f(self.sim(z1, z1))  # 计算自身相似度矩阵
        between_sim = f(self.sim(z1, z2))  # 计算正样本和负样本之间的相似度矩阵
        # 计算对比损失，半监督对比损失
        # between_sim.diag()就是u1与v1的相似度
        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))  # refl_sim.sum(1)按行求和

    # 为了减轻显存的压力，分batch计算semi_loss（与训练的batch_size不同）
    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)  # 温度因子
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # 计算自身相似度矩阵
            between_sim = f(self.sim(z1[mask], z2))  # 计算正样本和负样本之间的相似度矩阵
            # 计算对比损失
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() /
                                     (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(
                                                                                                                    i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = None):
        if batch_size is None:
            l1 = self.semi_loss(z1, z2)
            l2 = self.semi_loss(z2, z1)
        else:
            l1 = self.batched_semi_loss(z1, z2, batch_size)
            l2 = self.batched_semi_loss(z2, z1, batch_size)
        ret = (l1 + l2) * 0.5  # 取两个方向的损失的平均值
        ret = ret.mean() if mean else ret.sum()  # 是否取平
        return ret

class NCEAverage(nn.Module):
    # inputSize: 输入特征的维度
    # outputSize: 输出特征的维度，也是内存库（memory bank）的大小
    # K: 负采样的数量
    # T: 温度参数，用于调节相似度计算
    # momentum: 动量参数，用于更新内存库
    # Z: 归一化因子
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))


    def forward(self, l, ab, y, idx=None):  # y: 正样本的标签；idx：负样本的索引
        K = int(self.params[0].item())   # 负采样数量
        T = self.params[1].item()  # 温度参数
        Z_l = self.params[2].item()  # 用于归一化的常数
        Z_ab = self.params[3].item()  # 用于归一化的常数

        momentum = self.params[4].item()  # 内存库更新时的动量参数
        batchSize = l.size(0)  # 批次大小
        outputSize = self.memory_l.size(0)  # 内存库大小
        inputSize = self.memory_l.size(1)  # 输入特征维度

        # score computation
        if idx is None:
            # 生成一个采样结果idx，size：(batchSize, self.K + 1)
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        # 根据idx中的索引从NTM的memory_l和memory_ab中取出对应的数据
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))  # (batchSize, self.K + 1, 1)

        if self.use_softmax:  # 如果使用softmax
            # 归一化
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            # contiguous确保数据在内存中是连续
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            # 如果还没有被设置，则先根据输出结果的平均值乘以输出尺寸进行估计，并将其存储在self.params[2]和self.params[3]中
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            # 归一化
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj
