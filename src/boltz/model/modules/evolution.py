import torch
from torch import nn

import boltz.model.layers.initialize as init
from boltz.model.layers.pairformer import PairformerNoSeqModule
from boltz.model.modules.encodersv2 import PairwiseConditioning
from boltz.model.modules.utils import LinearNoBias


class EvolutionModule(nn.Module):
    """Evolutionary landscape prediction head."""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        head_hidden_dim: int = 384,
        num_dist_bins=64,
        max_dist=22,
        use_interface_mask: bool = False,
        groups: dict = {},
    ):
        super().__init__()
        self.use_interface_mask = use_interface_mask

        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )

        self.pairformer_stack = PairformerNoSeqModule(token_z, **pairformer_args)
        self.evolution_heads = EvolutionHeads(
            token_z=token_z,
            hidden_dim=head_hidden_dim,
        )

    def forward(
        self,
        s_inputs,
        z,
        x_pred,
        feats,
        multiplicity=1,
        use_kernels=False,
        use_interface_mask=None,
    ):
        if use_interface_mask is None:
            use_interface_mask = self.use_interface_mask

        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)

        if use_interface_mask:
            rec_mask = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0)
            rec_mask = rec_mask * pad_token_mask
            lig_mask = (
                feats["affinity_token_mask"]
                .repeat_interleave(multiplicity, 0)
                .to(torch.bool)
            )
            lig_mask = lig_mask * pad_token_mask
            pair_mask = (
                lig_mask[:, :, None] * rec_mask[:, None, :]
                + rec_mask[:, :, None] * lig_mask[:, None, :]
                + lig_mask[:, :, None] * lig_mask[:, None, :]
            )
        else:
            pair_mask = pad_token_mask[:, :, None] * pad_token_mask[:, None, :]

        z = self.pairformer_stack(
            z,
            pair_mask=pair_mask,
            use_kernels=use_kernels,
        )

        # evolution heads
        out_dict = self.evolution_heads(
            z=z,
            feats=feats,
            multiplicity=multiplicity,
            use_interface_mask=use_interface_mask,
        )

        return out_dict


class EvolutionHeads(nn.Module):
    def __init__(self, token_z, hidden_dim):
        super().__init__()

        self.pool_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, hidden_dim),
            nn.ReLU(),
        )

        self.to_evo_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, feats, multiplicity=1, use_interface_mask=False):
        pad_token_mask = (
            feats["token_pad_mask"]
            .repeat_interleave(multiplicity, 0)
            .unsqueeze(-1)
        )

        if use_interface_mask:
            rec_mask = (
                (feats["mol_type"] == 0)
                .repeat_interleave(multiplicity, 0)
                .unsqueeze(-1)
            )
            rec_mask = rec_mask * pad_token_mask
            lig_mask = (
                feats["affinity_token_mask"]
                .repeat_interleave(multiplicity, 0)
                .to(torch.bool)
                .unsqueeze(-1)
            ) * pad_token_mask
            pool_mask = (
                lig_mask[:, :, None] * rec_mask[:, None, :]
                + rec_mask[:, :, None] * lig_mask[:, None, :]
                + (lig_mask[:, :, None] * lig_mask[:, None, :])
            ) * (
                1
                - torch.eye(lig_mask.shape[1], device=lig_mask.device)
                .unsqueeze(-1)
                .unsqueeze(0)
            )
        else:
            N = pad_token_mask.shape[1]
            pool_mask = (
                pad_token_mask[:, :, None] * pad_token_mask[:, None, :]
            ) * (
                1
                - torch.eye(N, device=pad_token_mask.device)
                .unsqueeze(-1)
                .unsqueeze(0)
            )

        g = torch.sum(z * pool_mask, dim=(1, 2)) / (
            torch.sum(pool_mask, dim=(1, 2)) + 1e-7
        )

        g = self.pool_mlp(g)
        evo_energy = self.to_evo_energy(g).reshape(-1, 1)

        return {"evo_energy": evo_energy}
