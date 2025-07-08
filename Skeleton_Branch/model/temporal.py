
class Temporal_Attention(nn.Module):
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
        B, C, T, V = x.size()
        
        # Joint dimension is put inside the batch, in order to process each joint along the time separately
        x = x.permute(0, 3, 1, 2).reshape(-1, C, 1, T)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dim, self.dim, self.heads)

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        rel_logits = self.relative_logits(q)
        logits_sum = torch.add(logits, rel_logits)

        weights = logits_sum.softmax(logits, dim=-1)

        if (self.drop_connect and self.training):
            mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * T, device=device))
            mask = mask.reshape(B, self.Nh, T).unsqueeze(2).expand(B, self.Nh, T, T)
            weights = weights * mask
            weights = weights / (weights.sum(3, keepdim=True) + 1e-8)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, self.dv // self.Nh))
        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        attn_out = self.combine_heads_2d(attn_out)


        attn_out = self.attn_out(attn_out)
        attn_out = attn_out.reshape(N, V, -1, T).permute(0, 2, 3, 1)

        
        #qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        #q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        #dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        #attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        #if mask is not None:
            #assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            #dots = (dots + mask)*0.5

        #out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        #out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)

        return out
    
    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.to_qkv(x)
        N, C, V1, T1 = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)
        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dkh, V1 * T1))
        flat_k = torch.reshape(k, (N, Nh, dkh, V1 * T1))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, V1 * T1))
        return flat_q, flat_k, flat_v, q, k, v
    
    def split_heads_2d(self, x, Nh):
        B, channels, F, V = x.size()
        ret_shape = (B, Nh, channels // Nh, F, V)
        split = torch.reshape(x, ret_shape)
        return split
    
    def relative_logits_1d(self, q, rel_k):
        rel_logits = torch.einsum('bhld,md->bhlm', q, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        B, Nh, L, L = rel_logits.size()
        return rel_logits
    
    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x
    
    def relative_logits(self, q):
        B, Nh, dk, _, T = q.size()
        q = q.permute(0, 1, 3, 4, 2)
        q = q.reshape(B, Nh, T, dk)
        rel_logits = self.relative_logits_1d(q, self.key_rel)
        return rel_logits

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(Temporal_Attention(dim, mlp_dim, heads = heads, dropout = dropout)),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim*2, dropout = dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Temporal_Attention(dim, mlp_dim, heads = heads, dropout = dropout),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim*2, dropout = dropout)))
                ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x) # go to attention
            x = mlp(x) #go to MLP_Block
        return x



class TCN_STRANSF_unit(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, stride=1, residual=True, dropout=0.1, mask=None, mask_grad=True):
        super(TCN_STRANSF_unit, self).__init__()
        self.transf1 = Transformer(dim=in_channels, depth=1, heads=heads, mlp_dim=in_channels, dropout=dropout)
        self.tcn1 = unit_tcn_m(in_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        
        if mask != None:
            self.mask = nn.Parameter(mask, requires_grad=mask_grad)

    def forward(self, x, mask=None):
        B, C, T, V= x.size()
        tx = x.permute(0, 3, 1, 2).contiguous()
        if mask==None:
            tx = self.transf1(tx)
        else:
            tx = self.transf1(tx)
        #tx = tx.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()

        x = self.tcn1(tx) + self.residual(x)
        return self.relu(x)

