import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.temporal_transformer_3 import ZiT as TemporalZiT
from model.zoom_new_2 import ZiT as SpatialZiT
from model.zoom_new_2 import ZoT  # Reusing ZoT from zoom_new_2 as it’s identical

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
        # Fusion layer to combine spatial and temporal features
        self.fusion_layer = nn.Linear(192 * 2, 192)  # Concatenate 192 (spatial) + 192 (temporal) → 192
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)

        # Auxiliary classifiers for individual branches (optional)
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

        # Concatenate along channel dimension
        fused_out = torch.cat([spatial_out, temporal_out], dim=1)  # (N, C=384, T, M)
        # Apply fusion layer to reduce channels
        N, C, T, M = fused_out.size()
        fused_out = fused_out.permute(0, 2, 3, 1).contiguous()  # (N, T, M, C=384)
        fused_out = fused_out.view(N * T * M, C)  # (N*T*M, 384)
        fused_out = self.fusion_layer(fused_out)  # (N*T*M, 192)
        fused_out = fused_out.view(N, T, M, 192).permute(0, 3, 1, 2).contiguous()  # (N, 192, T, M)

        # Group transformer
        out = self.group_transf(fused_out)  # (N, num_class)

        # Auxiliary outputs for individual branches (optional)
        spatial_aux = self.spatial_fc(spatial_out.mean(3).mean(2))  # (N, num_class)
        temporal_aux = self.temporal_fc(temporal_out.mean(3).mean(2))  # (N, num_class)

        return out, spatial_aux, temporal_aux