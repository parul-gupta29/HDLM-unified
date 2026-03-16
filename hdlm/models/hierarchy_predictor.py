import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchyPredictor(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        num_levels=2,
        max_length=512
    ):
        super().__init__()

        self.num_levels = num_levels
        self.unknown_level = num_levels

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        self.hierarchy_embedding = nn.Embedding(
            num_levels + 1,
            hidden_size
        )

        self.position_embedding = nn.Embedding(
            max_length,
            hidden_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_levels
        )

    def forward(
        self,
        input_ids,
        hierarchy_input,
        attention_mask=None
    ):

        B, L = input_ids.shape

        pos = torch.arange(L, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(B, L)

        token_emb = self.token_embedding(input_ids)
        hier_emb = self.hierarchy_embedding(hierarchy_input)
        pos_emb = self.position_embedding(pos)

        x = token_emb + hier_emb + pos_emb

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        hidden = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        return self.classifier(hidden)

    def compute_loss(self, logits, hierarchy_labels, mask):

        mask = mask.bool()

        masked_logits = logits[mask]
        masked_labels = hierarchy_labels[mask]

        if masked_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return F.cross_entropy(masked_logits, masked_labels)
