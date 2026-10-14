"""
Model definitions.

`fusion` selects how the two modalities combine, so every ablation is the same
script with a different flag:

  gate       scalar sigmoid gate, h = g*s + (1-g)*p          (v10, recommended)
  concat     plain concatenation                             (fusion baseline)
  crossattn  PPI queries the residue sequence                (v22, repaired)
  seq_only   sequence branch alone                           (ABLATION)
  ppi_only   network branch alone                            (ABLATION)

Two corrections:

1. AttentionPooling now takes a padding mask. Without it, softmax spread weight
   over the padded tail -- every protein shorter than max_len had its
   representation diluted, worst for the shortest ones.

2. BatchNorm1d over [B, H, L] pooled its statistics across the padded tail as
   well as real residues. With max_len=1024, a 150-residue protein is 85%
   zeros, so its batch statistics were dominated by padding -- and the
   distortion varied with sequence length, which is exactly the kind of
   length-dependent bias you do not want in a function predictor. Replaced with
   a masked variant that normalises over valid positions only.

3. CrossModalAttention in update22 was inert. It passed the *pooled* sequence
   vector as key/value with sequence length 1. Softmax over a single key is
   identically 1.0, so the attention weights were constant and the query --
   the PPI embedding -- had no influence on the output whatsoever. The module
   reduced to out_proj(W_v @ seq_feat), a linear map of the sequence, and the
   only surviving PPI path was the additive ppi_proj residual. Here the
   key/value is the full residue map [B, L, H], so the network context genuinely
   selects which residues matter, which is what the design intended.
"""
import torch
import torch.nn as nn


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d whose statistics ignore padded positions."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, mask=None):
        # x: [B, C, L], mask: [B, L]
        if self.training:
            if mask is None:
                mean = x.mean(dim=(0, 2))
                var = x.var(dim=(0, 2), unbiased=False)
            else:
                m = mask.unsqueeze(1).to(x.dtype)          # [B, 1, L]
                n = m.sum() * 1.0
                mean = (x * m).sum(dim=(0, 2)) / n.clamp(min=1)
                diff = (x - mean.view(1, -1, 1)) * m
                var = (diff ** 2).sum(dim=(0, 2)) / n.clamp(min=1)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var.detach())
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class DilatedMultiScaleBlock(nn.Module):
    """
    Four parallel receptive fields + squeeze-and-excitation + residual.

    `dilations` is now configurable. The original fixed set (1, 1, 4, 8) gave
    the whole encoder a maximum receptive field of just 17 residues -- a single
    block with kernel 3 at dilation 8 spans (3-1)*8+1 = 17. Pfam functional
    domains run 100-300 residues, so the model could only ever act as a local
    motif detector before attention pooling. Stacking blocks with growing
    dilation fixes that: see `build_encoder` below.
    """

    def __init__(self, in_channels, out_channels, reduction=16,
                 dilations=(1, 1, 4, 8), kernels=(3, 5, 3, 3)):
        super().__init__()
        unit = out_channels // 4
        k0, k1, k2, k3 = kernels
        d0, d1, d2, d3 = dilations
        self.conv3 = nn.Conv1d(in_channels, unit, k0,
                               padding=(k0 - 1) * d0 // 2, dilation=d0)
        self.conv5 = nn.Conv1d(in_channels, unit, k1,
                               padding=(k1 - 1) * d1 // 2, dilation=d1)
        self.dilated_4 = nn.Conv1d(in_channels, unit, k2,
                                   padding=(k2 - 1) * d2 // 2, dilation=d2)
        self.dilated_8 = nn.Conv1d(in_channels, unit, k3,
                                   padding=(k3 - 1) * d3 // 2, dilation=d3)
        self.receptive_field = max((k - 1) * d + 1
                                   for k, d in zip(kernels, dilations))

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, max(out_channels // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(out_channels // reduction, 4), out_channels, 1),
            nn.Sigmoid(),
        )
        self.bottleneck = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = MaskedBatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x, mask=None):
        residual = self.bottleneck(x)
        out = torch.cat([self.conv3(x), self.conv5(x),
                         self.dilated_4(x), self.dilated_8(x)], dim=1)
        if mask is not None:
            # Keep padded columns at zero so they cannot leak into the SE
            # global-average statistics.
            m = mask.unsqueeze(1).to(out.dtype)
            out = out * m
            se_w = self.se(out * out.shape[-1] / m.sum(dim=2, keepdim=True)
                           .clamp(min=1))
            residual = residual * m
        else:
            se_w = self.se(out)
        out = out * se_w
        out = self.bn(out + residual, mask)
        out = self.act(out)
        return out * mask.unsqueeze(1).to(out.dtype) if mask is not None else out


class MultiScaleEncoder(nn.Module):
    """
    Stack of DilatedMultiScaleBlocks with geometrically growing dilation.

    Receptive field is additive across blocks: 1 + sum(rf_b - 1). With the
    default 3 blocks the encoder sees ~337 residues instead of 17, which is
    the range where whole functional domains actually live.
    """

    def __init__(self, in_channels, hidden_dim, num_blocks=3, reduction=16,
                 schedule="aggressive"):
        """
        schedule:
          "moderate"   dilations grow 2x per block -> theoretical RF ~113
          "aggressive" dilations grow 4x per block -> theoretical RF ~337

        Note this is a THEORETICAL receptive field. It bounds what the CNN can
        aggregate; it is not evidence the model learns features at that span.
        The ESM-2 input is already contextualised across the full sequence, so
        this only governs the additional local aggregation on top.
        """
        super().__init__()
        growth = {"moderate": 2, "aggressive": 4}[schedule]
        self.schedule = schedule
        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            base = growth ** b
            dil = (1, 1, 4 * base, 8 * base)
            self.blocks.append(DilatedMultiScaleBlock(
                in_channels if b == 0 else hidden_dim,
                hidden_dim, reduction, dilations=dil))
        self.receptive_field = 1 + sum(b.receptive_field - 1
                                       for b in self.blocks)

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: [B, H, L]
        scores = self.attn(x.transpose(1, 2))            # [B, L, 1]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(scores, dim=1)
        return torch.sum(x * w.transpose(1, 2), dim=2)   # [B, H]


class CrossModalAttention(nn.Module):
    """PPI embedding attends over the residue-level sequence features."""

    def __init__(self, seq_dim, ppi_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ppi_to_seq = nn.Linear(ppi_dim, seq_dim)
        self.attn = nn.MultiheadAttention(seq_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(seq_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residues, pooled, ppi_feat, mask=None):
        # residues: [B, L, H] -- a real sequence, so softmax is meaningful
        q = self.ppi_to_seq(ppi_feat).unsqueeze(1)       # [B, 1, H]
        pad = ~mask if mask is not None else None
        attended, _ = self.attn(q, residues, residues,
                                key_padding_mask=pad)
        return self.norm(pooled + self.dropout(attended.squeeze(1)))



class GOLabelAttentionHead(nn.Module):
    """
    GO-conditioned evidence retrieval.

    GO embeddings act as queries over a small set of protein evidence tokens.

    Protein residues
        -> learned token pooling
        -> K protein tokens

    GO embedding
        -> GO query
        -> attention over protein tokens
        -> GO-specific evidence
        -> GO-specific logit

    The branch is added residually to the original classifier.
    """

    def __init__(
        self,
        hidden_dim,
        go_embeddings,
        n_tokens=16,
        d_v=64,
        dropout=0.3,
        label_chunk_size=512,
    ):
        super().__init__()

        n_labels, go_dim = go_embeddings.shape

        self.n_tokens = n_tokens
        self.d_v = d_v
        self.label_chunk_size = label_chunk_size

        # Learned queries used to compress residue representations
        # into a small set of protein evidence tokens.
        self.token_q = nn.Parameter(
            torch.randn(n_tokens, hidden_dim) * 0.02
        )

        # GO embeddings are fixed ontology-derived representations.
#        self.register_buffer(
#            "go_embeddings",
#            go_embeddings.float().clone()
#        )
        
        self.go_embeddings = nn.Parameter(
            go_embeddings.float().clone()
        )

        # Project GO representation into protein attention space.
        self.go_proj = nn.Linear(
            go_dim,
            hidden_dim,
        )

        # Protein token key/value projections.
        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            d_v,
        )

        # Convert GO-specific evidence into one logit.
        self.score = nn.Linear(
            d_v,
            1,
        )

        self.dropout = nn.Dropout(dropout)

        self.scale = hidden_dim ** -0.5

        # Residual contribution starts at zero.
        # Therefore the initial model output is the baseline output.
#        self.gamma = nn.Parameter(
#            torch.zeros(1)
#        )
        
        self.go_gate = nn.Parameter(
            torch.full((go_embeddings.shape[0],), -4.0)
        )

    def forward(self, residues, mask):
        """
        residues: [B, L, H]
        mask:     [B, L] -- True for valid residues
        """

        B = residues.size(0)

        # --------------------------------------------------------------
        # 1. Compress residue sequence into K protein evidence tokens.
        # --------------------------------------------------------------
        q = self.token_q.unsqueeze(0).expand(
            B,
            -1,
            -1,
        )  # [B, K, H]

        token_scores = torch.bmm(
            q,
            residues.transpose(1, 2),
        ) * self.scale  # [B, K, L]

        token_scores = token_scores.masked_fill(
            ~mask.unsqueeze(1),
            float("-inf"),
        )

        token_attn = torch.softmax(
            token_scores,
            dim=-1,
        )

        protein_tokens = torch.bmm(
            token_attn,
            residues,
        )  # [B, K, H]

        # --------------------------------------------------------------
        # 2. Protein tokens -> keys and values.
        # --------------------------------------------------------------
        k = self.k_proj(
            protein_tokens
        )  # [B, K, H]

        v = self.v_proj(
            protein_tokens
        )  # [B, K, d_v]

        # --------------------------------------------------------------
        # 3. GO embeddings -> queries.
        # --------------------------------------------------------------
        go_q = self.go_proj(
            self.go_embeddings
        )  # [T, H]

        # --------------------------------------------------------------
        # 4. Each GO term attends to protein evidence tokens.
        #
        # Process labels in chunks so we never materialize
        # [B, 19323, d_v] unnecessarily.
        # --------------------------------------------------------------
        logits_chunks = []

        n_labels = go_q.size(0)

        for start in range(
            0,
            n_labels,
            self.label_chunk_size,
        ):

            end = min(
                start + self.label_chunk_size,
                n_labels,
            )

            q_chunk = go_q[start:end]  # [C, H]

            # [B, C, K]
            scores = torch.einsum(
                "ch,bkh->bck",
                q_chunk,
                k,
            ) * self.scale

            attention = torch.softmax(
                scores,
                dim=-1,
            )

            # [B, C, d_v]
            context = torch.einsum(
                "bck,bkd->bcd",
                attention,
                v,
            )

            # [B, C]
            chunk_logits = self.score(
                self.dropout(context)
            ).squeeze(-1)

            logits_chunks.append(
                chunk_logits
            )

        go_logits = torch.cat(
            logits_chunks,
            dim=1,
        )

        # Residual GO contribution.
        #return self.gamma * go_logits

        gate = torch.sigmoid(self.go_gate)
        
        return gate.unsqueeze(0) * go_logits
        


class LightProteinFusion(nn.Module):
#    def __init__(self, num_classes, esm_dim=640, ppi_dim=128, hidden_dim=256,
#                 dropout=0.3, fusion="gate", num_blocks=3,
#                 dilation_schedule="aggressive",
#                 go_embeddings=None,
#                 ): #go_embeddings=None, extra added later
                 
    def __init__(self, num_classes, esm_dim=640, ppi_dim=128, hidden_dim=256,
                 dropout=0.3, fusion="gate", num_blocks=3,
                 dilation_schedule="aggressive", go_embeddings=None,
                 clf_hidden=512):
             
        super().__init__()
        self.fusion = fusion
        self.uses_seq = fusion != "ppi_only"
        self.uses_ppi = fusion != "seq_only"

        if self.uses_seq:
            self.encoder = MultiScaleEncoder(esm_dim, hidden_dim, num_blocks,
                                             schedule=dilation_schedule)
            self.attn_pool = AttentionPooling(hidden_dim)
            self.norm_seq = nn.LayerNorm(hidden_dim)
            self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        if self.uses_ppi:
            self.norm_ppi = nn.LayerNorm(ppi_dim)
            self.ppi_proj = nn.Linear(ppi_dim, hidden_dim)

        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, 1), nn.Sigmoid())
            clf_in = hidden_dim
        elif fusion == "concat":
            clf_in = 2 * hidden_dim
        elif fusion == "crossattn":
            self.cross = CrossModalAttention(hidden_dim, ppi_dim)
            clf_in = hidden_dim
        else:
            clf_in = hidden_dim

#        self.classifier = nn.Sequential(
#            nn.Linear(clf_in, 512), nn.GELU(), nn.Dropout(dropout),
#            nn.Linear(512, num_classes),
#        )
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, clf_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(clf_hidden, num_classes),
        )
        self.go_label_attn = ( ##added
            GOLabelAttentionHead(
                hidden_dim=hidden_dim,
                go_embeddings=go_embeddings,
                n_tokens=16,
                d_v=64,
                dropout=dropout,
                label_chunk_size=512,
            )
            if go_embeddings is not None
            else None
        )

    def forward(self, x_seq, mask, x_ppi):
        s = p = None
        residues = None

        if self.uses_seq:
            feat = self.encoder(x_seq, mask)                 # [B, H, L]
            residues = feat.transpose(1, 2)                  # [B, L, H]
            pooled = self.attn_pool(feat, mask)
            s = self.seq_proj(self.norm_seq(pooled))
        if self.uses_ppi:
            p = self.ppi_proj(self.norm_ppi(x_ppi))

        if self.fusion == "gate":
            g = self.gate(torch.cat([s, p], dim=1))
            h = g * s + (1 - g) * p
        elif self.fusion == "concat":
            h, g = torch.cat([s, p], dim=1), None
        elif self.fusion == "crossattn":
            h = self.cross(residues, s, x_ppi, mask)
            g = None
        elif self.fusion == "seq_only":
            h, g = s, None
        else:
            h, g = p, None

        if g is None:
            g = torch.zeros(x_seq.size(0), 1, device=x_seq.device)
        #return self.classifier(h), g
        
        logits = self.classifier(h) ##added part

        if (
            self.go_label_attn is not None
            and residues is not None
        ):
            logits = logits + self.go_label_attn(
                residues,
                mask,
            )
        
        return logits, g


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
class GOEmbeddingClassifier(nn.Module):
    """
    Original MLP classifier with an additional ontology-informed
    label-specific component.

    h
      -> Linear(input_dim, 512)
      -> GELU
      -> Dropout
      -> [ W_free + 0.1 * W_GO ]
      -> logits

    W_GO is generated from trainable GO embeddings.
    """

    def __init__(self, input_dim, go_embeddings, dropout=0.3):
        super().__init__()

        if go_embeddings.ndim != 2:
            raise ValueError(
                f"go_embeddings must be 2D, got shape {tuple(go_embeddings.shape)}"
            )

        n_labels, go_dim = go_embeddings.shape

        # Same hidden MLP as the original classifier.
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pretrained GO embeddings, fine-tuned jointly.
        self.go_embeddings = nn.Parameter(
            go_embeddings.float().clone()
        )

        # Original unrestricted label-specific classifier.
        self.weight_free = nn.Parameter(
            torch.empty(n_labels, 512)
        )

        self.bias = nn.Parameter(
            torch.zeros(n_labels)
        )

        # Learn how GO embedding contributes to classifier weights.
        self.go_to_classifier = nn.Linear(
            go_dim, 512, bias=False
        )

        # Fixed contribution for this first experiment.
        self.register_buffer(
            "go_scale",
            torch.tensor(0.1)
        )

        nn.init.xavier_uniform_(self.weight_free)

    def forward(self, h):
        # h: [batch, input_dim]

        h = self.hidden(h)  # [batch, 512]

        # GO embedding -> label-specific classifier weights.
        # [n_labels, 128] -> [n_labels, 512]
        weight_go = self.go_to_classifier(
            self.go_embeddings
        )

        # Original free classifier + GO-informed component.
        weight = (
            self.weight_free
            + self.go_scale * weight_go
        )

        # [batch, 512] @ [512, n_labels]
        logits = h @ weight.T + self.bias

        return logits