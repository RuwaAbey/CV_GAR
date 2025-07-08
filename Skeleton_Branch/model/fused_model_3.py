import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.temporal_transformer_3 import ZiT as TemporalZiT
from model.zoom_new_2 import ZiT as SpatialZiT
from model.zoom_new_2 import ZoT

class FusedModel(nn.Module):
    def __init__(self, num_class=8, in_channels=3, num_person=12, num_point=17, num_head=6, graph=None, graph_args=dict()):
        super(FusedModel, self).__init__()
        self.spatial_zit = SpatialZiT(
            in_channels=in_channels,
            num_person=num_person,
            num_point=num_point,
            num_head=num_head,
            graph=graph,
            graph_args=graph_args
        )
        self.temporal_zit = TemporalZiT(
            in_channels=in_channels,
            num_person=num_person,
            num_point=num_point,
            num_head=num_head,
            graph=graph,
            graph_args=graph_args
        )
        self.group_transf = ZoT(
            num_class=num_class,
            num_head=num_head,
            num_person=num_person
        )
        # Gating network for per-(t,m) gates
        self.gate_conv = nn.Conv2d(192 * 2, 1, kernel_size=1)  # Input: 384 channels, Output: 1 channel for g(t,m)
        nn.init.xavier_uniform_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)
        # Auxiliary classifiers
        self.spatial_fc = nn.Linear(192, num_class)
        self.temporal_fc = nn.Linear(192, num_class)
        nn.init.normal_(self.spatial_fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.temporal_fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        # Input: (N, C, T, V, M)
        # Spatial branch
        spatial_out = self.spatial_zit(x)  # (N, C=192, T, M)
        # Temporal branch
        temporal_out = self.temporal_zit(x)  # (N, C=192, T, M)

        # Gated fusion
        # Concatenate features
        concat_out = torch.cat([spatial_out, temporal_out], dim=1)  # (N, C=384, T, M)
        # Compute gate g(t,m)
        gate = self.gate_conv(concat_out)  # (N, 1, T, M)
        gate = torch.sigmoid(gate)  # g(t,m) in [0, 1]
        # Compute convex combination: fused(t,m) = g(t,m) * s(t,m) + (1 - g(t,m)) * q(t,m)
        fused_out = gate * spatial_out + (1 - gate) * temporal_out  # (N, 192, T, M)

        # Group transformer
        out = self.group_transf(fused_out)  # (N, num_class)

        # Auxiliary outputs
        spatial_aux = self.spatial_fc(spatial_out.mean(3).mean(2))  # (N, num_class)
        temporal_aux = self.temporal_fc(temporal_out.mean(3).mean(2))  # (N, num_class)

        return out, spatial_aux, temporal_aux