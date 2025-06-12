import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                                stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_tcn_m(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=[1, 3, 7]):        # ks=9 initial
        super(unit_tcn_m, self).__init__()

        pad1 = int((kernel_size[0] - 1) / 2)
        pad2 = int((kernel_size[1] - 1) / 2)
        pad3 = int((kernel_size[2] - 1) / 2)

        mid_channels = out_channels//3

        self.conv11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv31 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.conv12 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[0], 1), padding=(pad1, 0),
                                stride=(stride, 1))
        self.conv22 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[1], 1), padding=(pad2, 0),
                                stride=(stride, 1))
        self.conv32 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[2], 1), padding=(pad3, 0),
                                stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv11)
        conv_init(self.conv21)
        conv_init(self.conv31)
        conv_init(self.conv12)
        conv_init(self.conv22)
        conv_init(self.conv32)
        bn_init(self.bn, 1)

    def forward(self, x):
        x1 = self.conv12(self.conv11(x))
        x2 = self.conv22(self.conv21(x))
        x3 = self.conv32(self.conv31(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x  = self.bn(x)
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        # batchsize, channel, t, node_num
        N, C, T, V = x.size()
        #print(N, C, T, V)
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn_m(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # Input x: [N, C, T, V]
        x = x.permute(0, 2, 3, 1)  # [N, T, V, C]
        x = self.norm(x)  # Apply LayerNorm over C
        x = x.permute(0, 3, 1, 2)  # [N, C, T, V]
        return self.fn(x, **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input x: [N, C, T, V]
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1)  # [N, T, V, C]
        x = x.reshape(-1, C)  # [N*T*V, C]
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        x = x.reshape(N, T, V, C).permute(0, 3, 1, 2)  # [N, C, T, V]
        return x

###############################


class Temporal_Attention(nn.Module):
    def __init__(self, dim, heads=3, dropout_rate=0.1, relative_pos=True, drop_connectivity=False,max_T=100):
        super().__init__()
        self.heads = heads
        self.dim = dim  # Corresponds to in_channels and output channels
        self.drop_connect = drop_connectivity
        self.relative = relative_pos

        self.qkv_transform = nn.Conv2d(dim, dim * 3, kernel_size=1)  # q, k, v: dim
        self.attention_output = nn.Conv2d(dim, dim, kernel_size=1)
        conv_init(self.qkv_transform)
        conv_init(self.attention_output)
        self.dropout = nn.Dropout(dropout_rate)

        if self.relative:
            # Predefine key_rel_embed for a maximum T
            self.key_rel_embed = nn.Parameter(
                torch.randn(2 * max_T - 1, self.dim // self.heads, requires_grad=True)
            )

    def forward(self, x):
        N, C, T, V = x.size()
       
        x = x.permute(0, 3, 1, 2).reshape(N * V, C, 1, T)  # (N*V, C, 1, T)
        
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dim, self.dim, self.heads, N, V)
        
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)  # (N*V, heads, T, T)
        
        if self.relative:
            self.rel_k = self.key_rel_embed[:(2 * T - 1)]
            rel_logits = self.relative_logits(q)
            logits_sum = logits + rel_logits
        else:
            logits_sum = logits
        
        weights = torch.softmax(logits_sum, dim=-1)
        
        if self.drop_connect and self.training:
            mask_ = torch.bernoulli(0.5 * torch.ones((N * V, self.heads, T, T), device=x.device))
            weights = weights * mask_
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))  # (N*V, heads, T, dim/heads)
        attn_out = attn_out.view(N * V, self.heads, self.dim // self.heads, 1, T).permute(0, 1, 3, 4, 2)
        attn_out = attn_out.reshape(N * V, self.dim, 1, T)
        
        attn_out = self.attention_output(attn_out)  # (N*V, dim, 1, T)
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.view(N, V, self.dim, T).permute(0, 2, 3, 1)  # (N, dim, T, V)
        
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, heads, N, V):
        qkv = self.qkv_transform(x)  # (N*V, 3*dk, 1, T)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)  # Each: (N*V, dk, 1, T)
        q = self.split_heads_2d(q, heads)  # (N*V, heads, dk/heads, 1, T)
        k = self.split_heads_2d(k, heads)
        v = self.split_heads_2d(v, heads)
        dkh = dk // heads
        # No scaling: Removed q = q * (dkh ** -0.5)
        flat_q = q.view(N * V, heads, dkh, -1)
        flat_k = k.view(N * V, heads, dkh, -1)
        flat_v = v.view(N * V, heads, dv // heads, -1)
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, heads):
        B, C, F, T = x.size()
        return x.view(B, heads, C // heads, F, T)

    def relative_logits(self, q):
        B, Nh, dk, _, T = q.size()
        q = q.permute(0, 1, 3, 4, 2)
        q = q.reshape(B, Nh, T, dk)
        rel_logits = self.relative_logits_1d(q, self.rel_k)

        return rel_logits

    def relative_logits_1d(self, q, rel_k):
        rel_logits = torch.einsum('bhld,md->bhlm', q, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        B, Nh, L, L = rel_logits.size()
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1), device=x.device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = x.view(B, Nh, L * 2 * L)
        flat_pad = torch.zeros((B, Nh, L - 1), device=x.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = flat_x_padded.view(B, Nh, L + 1, 2 * L - 1)[:, :, :L, L - 1:]
        return final_x

# Assuming Residual, LayerNormalize, MLP_Block, unit_tcn_m, unit_tcn are defined
class Temporal_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Temporal_Attention(dim, heads=heads, dropout_rate=dropout)),
                Residual(LayerNormalize(dim, MLP_Block(dim, dim * 2, dropout=dropout)))
            ]))
    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)
            x = mlp(x)
        return x

class TCN_STRANSF_unit(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, stride=1, residual=True, dropout=0.1):
        super().__init__()
        self.transf1 = Temporal_Transformer(dim=in_channels, depth=1, heads=heads, dropout=dropout)
        self.tcn1 = unit_tcn_m(in_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        B, C, T, V = x.size()
        tx = self.transf1(x)
        tx = tx.contiguous().reshape(B, T, V, C).permute(0, 3, 1, 2)
        x = self.tcn1(tx) + self.residual(x)
        return self.relu(x)


##############################

class ZiT(nn.Module):
    def __init__(self, in_channels=3, num_person=5, num_point=17, num_head=6, graph=None, graph_args=dict()):
        super(ZiT, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        self.heads = num_head

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.A = torch.from_numpy(self.graph.A[0].astype(np.float32))
        self.l1 = TCN_GCN_unit(3, 48, self.graph.A, residual=False)
        self.l2 = TCN_STRANSF_unit(48, 48, heads=num_head)      
        self.l3 = TCN_STRANSF_unit(48, 48, heads=num_head)
        self.l4 = TCN_STRANSF_unit(48, 96, heads=num_head)
        self.l5 = TCN_STRANSF_unit(96, 96, heads=num_head)
        self.l6 = TCN_STRANSF_unit(96, 192, heads=num_head)
        self.l7 = TCN_STRANSF_unit(192, 192, heads=num_head)


    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        B, C_, T_, V_ = x.size()
        x = x.view(N, M, C_, T_, V_).mean(4)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class ZoT(nn.Module):
    def __init__(self, num_class=15, num_head=6):
        super(ZoT, self).__init__()

        self.heads = num_head

        self.conv1 = nn.Conv2d(192, num_head, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(192, num_head, kernel_size=(1, 1))
        conv_init(self.conv1)
        conv_init(self.conv2)

        self.l1 = TCN_STRANSF_unit(192, 276, heads=num_head)       # 192 276
        self.l2 = TCN_STRANSF_unit(276, 276, heads=num_head)

        self.fc = nn.Linear(276, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
    
    def forward(self, x):
        # N,C,T,M
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1.unsqueeze(3)
        x2 = x2.unsqueeze(4)
        mask = x1-x2
        N, C, T, M, M2 = mask.shape
        mask = mask.permute(0, 2, 1, 3, 4).contiguous().view(N*T, C, M, M2).detach()
        mask = mask.softmax(dim=-1)


        x = self.l1(x)
        x = self.l2(x)
        x = x.mean(3).mean(2)

        return self.fc(x)

class Model(nn.Module):
    def __init__(self, num_class=60, in_channels=3, num_person=2, num_point=17, num_head=6, graph=None, graph_args=dict()):
        super(Model, self).__init__()

        self.body_transf = ZiT(in_channels=in_channels, num_person=num_person, num_point=num_point, num_head=num_head, graph=graph, graph_args=graph_args)
        self.group_transf = ZoT(num_class=num_class, num_head=num_head)


    def forward(self, x):
        x = self.body_transf(x)
        x = self.group_transf(x)

        return x
