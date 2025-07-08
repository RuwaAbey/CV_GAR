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
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads = 3, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim*3, bias = True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            # Apply mask before softmax or after? Depends on how you want the mask to interact.
            # Original code applies after softmax, let's keep that for now.
            # If mask contains -inf for masked positions, apply before softmax for proper masking.
            dots = (dots + mask)*0.5 # The original code has this, it's a bit unusual.
                                    # Typically, you'd add -inf to mask values before softmax for true masking.
                                    # For a learnable mask, this might be a soft masking approach.
                                    # Let's assume it's intentional for now.


        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(Attention(dim, mlp_dim, heads = heads, dropout = dropout)),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim*2, dropout = dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, mlp_dim, heads = heads, dropout = dropout),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim*2, dropout = dropout)))
                ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x



class TCN_STRANSF_unit(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, stride=1, residual=True, dropout=0.1, mask=None, mask_grad=True):
        super(TCN_STRANSF_unit, self).__init__()
        # Spatial Transformer (operates on N=V for each T)
        self.spatial_transf = Transformer(dim=in_channels, depth=1, heads=heads, mlp_dim=in_channels, dropout=dropout)
        
        # Parallel Temporal Transformer (operates on N=T for each V)
        self.temporal_transf = Transformer(dim=in_channels, depth=1, heads=heads, mlp_dim=in_channels, dropout=dropout)

        # Multi-scale Temporal Convolutional Network
        self.tcn1 = unit_tcn_m(in_channels, out_channels, stride=stride)
        
        # Linear layer to combine outputs from spatial transformer, temporal transformer, and tcn
        # The sum of channels after concatenation should be in_channels + in_channels + out_channels
        # Let's adjust this for clearer channel management.
        # It's better to process outputs independently and sum them up or use a conv to merge.
        # For simplicity, let's process the input for each branch and then combine.
        # We'll apply tcn after summing transf outputs and original.
        
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
            
        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=mask_grad)
        else:
            self.mask = None # Ensure mask is None if not provided

        # Initialize normalization for parallel branches
        self.norm_spatial_transf = nn.BatchNorm2d(in_channels)
        self.norm_temporal_transf = nn.BatchNorm2d(in_channels)

        bn_init(self.norm_spatial_transf, 1)
        bn_init(self.norm_temporal_transf, 1)


    def forward(self, x, mask=None):
        B, C, T, V = x.size()
        
        # --- Spatial Transformer Branch ---
        # Reshape for spatial transformer: (B*T, V, C)
        # Each 'pixel' is a (T, V) coordinate, treating V as 'sequence length' for self-attention.
        spatial_transf_input = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        
        # Apply spatial transformer
        if mask is None and self.mask is not None: # Use pre-defined mask if no runtime mask
            spatial_transf_out = self.spatial_transf(spatial_transf_input, self.mask)
        else: # Use runtime mask or no mask
            spatial_transf_out = self.spatial_transf(spatial_transf_input, mask)
        
        # Reshape back: (B, C, T, V)
        spatial_transf_out = spatial_transf_out.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()
        spatial_transf_out = self.norm_spatial_transf(spatial_transf_out) # Normalize after spatial transf

        # --- Temporal Transformer Branch ---
        # Reshape for temporal transformer: (B*V, T, C)
        # Each 'pixel' is a (T, V) coordinate, treating T as 'sequence length' for self-attention.
        temporal_transf_input = x.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)
        
        # Apply temporal transformer (no mask for now, as it's typically for spatial graphs)
        temporal_transf_out = self.temporal_transf(temporal_transf_input)
        
        # Reshape back: (B, C, T, V)
        temporal_transf_out = temporal_transf_out.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()
        temporal_transf_out = self.norm_temporal_transf(temporal_transf_out) # Normalize after temporal transf
        
        # --- Combine Transformer Outputs (Element-wise sum is a common practice) ---
        # You could also concatenate and then use a 1x1 conv to merge channels if desired
        combined_transf_out = spatial_transf_out + temporal_transf_out

        # --- Multi-scale Temporal Convolutional Network Branch ---
        # Apply TCN to the combined transformer output (or original input, or concatenated).
        # Let's apply it to the sum of transformed outputs, as it provides a richer input.
        tcn_out = self.tcn1(combined_transf_out)
        
        # --- Residual Connection ---
        # The residual connection should operate on the input 'x' and be added to the final output.
        # Ensure channel dimensions match.
        final_out = tcn_out + self.residual(x)
        
        return self.relu(final_out)


class ZiT(nn.Module):
    def __init__(self, in_channels=3, num_person=5, num_point=18, num_head=6, graph=None, graph_args=dict()):
        super(ZiT, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        self.heads = num_head

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # Ensure self.graph.A is a numpy array for conversion to torch.from_numpy
        self.A = torch.from_numpy(self.graph.A[0].astype(np.float32))
        
        # Use the modified TCN_STRANSF_unit
        self.l1 = TCN_GCN_unit(3, 48, self.graph.A, residual=False)
        self.l2 = TCN_STRANSF_unit(48, 48, heads=num_head, mask=self.A, mask_grad=False)      
        self.l3 = TCN_STRANSF_unit(48, 48, heads=num_head, mask=self.A, mask_grad=False)
        self.l4 = TCN_STRANSF_unit(48, 96, heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l5 = TCN_STRANSF_unit(96, 96, heads=num_head, mask=self.A, mask_grad=True)
        self.l6 = TCN_STRANSF_unit(96, 192, heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l7 = TCN_STRANSF_unit(192, 192, heads=num_head, mask=self.A, mask_grad=True)


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

        self.l1 = TCN_STRANSF_unit(192, 276, heads=num_head)        # 192 276
        self.l2 = TCN_STRANSF_unit(276, 276, heads=num_head)

        # New: Attention mechanism for final pooling
        # We want to pool across T (dim 2) and M (dim 3)
        # The input to attention will be of shape (N, C, T*M)
        # We'll use a small MLP to generate attention weights.
        self.attention_pooling = nn.Sequential(
            nn.Linear(276, 128), # Input feature dimension is 276 (C_out from l2)
            nn.ReLU(),
            nn.Linear(128, 1) # Output 1 score per feature for attention
        )
        self.softmax_attention = nn.Softmax(dim=1) # Softmax over the flattened T*M dimension

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


        x = self.l1(x, mask)
        x = self.l2(x, mask)
        
        # New: Attention-based pooling
        # x is (N, C_out, T_out, M_out)
        N, C_out, T_out, M_out = x.size()

        # Reshape to (N, C_out, T_out * M_out) to apply attention across the flattened spatial-temporal dimension
        x_flat = x.view(N, C_out, T_out * M_out) # (N, 276, T_out*M_out)

        # Permute to (N, T_out * M_out, C_out) for the Linear layer
        x_for_attn = x_flat.permute(0, 2, 1) # (N, T_out*M_out, 276)

        # Get attention scores
        attn_scores = self.attention_pooling(x_for_attn) # (N, T_out*M_out, 1)

        # Apply softmax to get weights
        attn_weights = self.softmax_attention(attn_scores) # (N, T_out*M_out, 1)

        # Apply attention weights: element-wise multiply and sum
        # Reshape attn_weights back to (N, 1, T_out * M_out) for broadcasting
        x = (x_flat * attn_weights.permute(0, 2, 1)).sum(dim=2) # (N, C_out)
        
        return self.fc(x)

class Model(nn.Module):
    def __init__(self, num_class=15, in_channels=3, num_person=5, num_point=18, num_head=6, graph=None, graph_args=dict()):
        super(Model, self).__init__()

        self.body_transf = ZiT(in_channels=in_channels, num_person=num_person, num_point=num_point, num_head=num_head, graph=graph, graph_args=graph_args)
        self.group_transf = ZoT(num_class=num_class, num_head=num_head)


    def forward(self, x):
        x = self.body_transf(x)
        x = self.group_transf(x)

        return x