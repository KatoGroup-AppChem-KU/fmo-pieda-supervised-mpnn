"""Cumulative gated node/pair variants; existing input encoders and heads reused."""
import torch
from torch import nn
from torch.nn import functional as F
from pair_feedback_model import FeedbackModel

class GatedFeedbackModel(FeedbackModel):

    def __init__(self, condition, stage):
        super().__init__(condition)
        assert stage in (1, 2, 3) and condition in ('R0', 'R1')
        self.stage = stage
        del self.grus
        if stage >= 2:
            self.component_projections = nn.ModuleList([nn.ModuleList([nn.Linear(256, 64) for _ in range(4)]) for _ in range(4)])
            self.pair_norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(64) for _ in range(4)]) for _ in range(4)])
        self.node_gates = nn.ModuleList([nn.ModuleList([nn.Linear(512, 256) for _ in range(4 if stage == 3 else 1)]) for _ in range(4)])
        for layer in self.node_gates:
            for gate in layer:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -2.0)
        masks = torch.zeros(4, 256)
        for k in range(4):
            masks[k, k * 64:(k + 1) * 64] = 1
        self.register_buffer('component_masks', masks)

    def encode_nodes(self, x, edge_index, edge_attr):
        h, e = (self.node_encoder(x), self.edge_encoder(edge_attr))
        src, dst = edge_index
        p = self.pair_init(torch.cat([h[src], h[dst], e], -1))
        for layer, (update, norm, message, hn) in enumerate(zip(self.pair_updates, self.pair_norms, self.pair_messages, self.norms)):
            shared = update(torch.cat([h[src], h[dst], p, e], -1))
            if self.stage == 1:
                p = norm(p + shared)
            else:
                p = torch.cat([norm[k](p[:, k * 64:(k + 1) * 64] + F.silu(project(shared))) for k, project in enumerate(self.component_projections[layer])], -1)
            if self.stage < 3:
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, message(p))
                gate = torch.sigmoid(self.node_gates[layer][0](torch.cat([h, agg], -1)))
                delta = gate * agg
            else:
                delta = torch.zeros_like(h)
                for k, gate_layer in enumerate(self.node_gates[layer]):
                    msg = F.linear(p[:, k * 64:(k + 1) * 64], message.weight[:, k * 64:(k + 1) * 64], message.bias / 4)
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)
                    gate = torch.sigmoid(gate_layer(torch.cat([h, agg], -1)))
                    delta = delta + gate * agg
            h = hn(h + F.dropout(delta, p=0.1, training=self.training))
        self.pair_state = p
        return h

    def decode_edges(self, h, ei, geom, selected):
        if self.stage == 1:
            return super().decode_edges(h, ei, geom, selected)
        p = self.pair_state[selected]
        predictions = [self.component_heads[0](self.edge_decoder_trunk(p))]
        for k in range(4):
            predictions.append(self.component_heads[k + 1](self.edge_decoder_trunk(p * self.component_masks[k])))
        return torch.cat(predictions, -1)

def make_model(condition, stage):
    return GatedFeedbackModel(condition, stage).cuda()
