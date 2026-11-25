"""Four node/pair feedback blocks; geometry and PIEDA have separate readouts."""
import torch
from torch import nn
from torch.nn import functional as F
import screen_g_p1_bdb_20260914 as screen

class FeedbackModel(nn.Module):

    def __init__(self, condition):
        super().__init__()
        old = screen.ORIGINAL_MAKE_MODEL('v6', 'geo_pieda')
        for name in ('node_encoder', 'edge_encoder', 'grus', 'norms', 'distance_decoder', 'edge_decoder_trunk', 'component_heads'):
            setattr(self, name, getattr(old, name))
        self.edge_decoder_trunk[0] = nn.Linear(256, 256)
        self.pair_init = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256))
        self.pair_updates = nn.ModuleList([nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.SiLU()) for _ in range(4)])
        self.pair_norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(4)])
        self.pair_messages = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
        self.residual_decoder = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 5))
        self.pair_state = None
        if condition == 'R0':
            self.edge_decoder_trunk.requires_grad_(False)
            self.component_heads.requires_grad_(False)
        if condition != 'R2':
            self.residual_decoder.requires_grad_(False)

    def encode_nodes(self, x, edge_index, edge_attr):
        h, e = (self.node_encoder(x), self.edge_encoder(edge_attr))
        src, dst = edge_index
        p = self.pair_init(torch.cat([h[src], h[dst], e], -1))
        for update, norm, message, gru, hn in zip(self.pair_updates, self.pair_norms, self.pair_messages, self.grus, self.norms):
            p = norm(p + update(torch.cat([h[src], h[dst], p, e], -1)))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, message(p))
            h = hn(h + F.dropout(gru(agg, h), p=0.1, training=self.training))
        self.pair_state = p
        return h

    def decode_distance(self, h, ei, selected):
        u, v = ei[:, selected]
        return self.distance_decoder(torch.cat([h[u], h[v]], -1))

    def decode_edges(self, h, ei, geom, selected):
        z = self.edge_decoder_trunk(self.pair_state[selected])
        return torch.cat([head(z) for head in self.component_heads], -1)

def make_model(condition):
    return FeedbackModel(condition).cuda()
