import torch
from torch import nn

import boltz.model.layers.initialize as init
from boltz.model.layers.pairformer import PairformerNoSeqModule
from boltz.model.modules.encodersv2 import PairwiseConditioning
from boltz.model.modules.utils import LinearNoBias


class EvolutionModule(nn.Module):
    """Evolutionary landscape prediction head.

    Architecturally mirrors AffinityModule: takes detached trunk outputs
    (pair representation z, single representation s_inputs, predicted
    coordinates x_pred), builds a distance-conditioned pair feature tensor,
    refines it through a small PairformerNoSeq stack, pools to a global
    vector, and projects to a scalar evolutionary energy.

    The energy is trained with relative losses (Bradley-Terry, margin
    ranking) since only relative ordering is known, not absolute values.

    Two masking modes controlled by use_interface_mask:
      False (default): attend/pool over ALL valid token pairs
      True:            attend/pool only over cross-interface pairs
                       (ligand-receptor + receptor-ligand + ligand-ligand),
                       identical to the affinity head

    Parameters
    ----------
    token_s : int
        Single/token representation dimension (from trunk).
    token_z : int
        Pair representation dimension (from trunk).
    pairformer_args : dict
        Arguments for PairformerNoSeqModule (num_blocks, dropout,
        pairwise_head_width, pairwise_num_heads, activation_checkpointing).
    head_hidden_dim : int
        Hidden dimension for the prediction MLP heads.
    num_dist_bins : int
        Number of distance bins for the coordinate-derived distogram.
    max_dist : float
        Maximum distance (Angstrom) for distance binning.
    use_interface_mask : bool
        Default masking mode. Can be overridden per forward call.
    """

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
        """Forward pass: condition → refine → pool → predict energy.

        Parameters
        ----------
        s_inputs : Tensor [B, N, token_s]
            Single/token representation (detached from trunk).
        z : Tensor [B, N, N, token_z]
            Pair representation (detached from trunk).
        x_pred : Tensor [B*mult, N_atoms, 3] or [B, mult, N_atoms, 3]
            Predicted atom coordinates.
        feats : dict
            Feature dictionary (token_pad_mask, mol_type, affinity_token_mask,
            token_to_rep_atom, etc.).
        multiplicity : int
            Number of diffusion samples per batch element.
        use_kernels : bool
            Whether to use fused CUDA kernels.
        use_interface_mask : bool or None
            Override the default masking mode. None uses self.use_interface_mask.

        Returns
        -------
        dict with 'evo_energy' : Tensor [B*mult, 1]
        """
        if use_interface_mask is None:
            use_interface_mask = self.use_interface_mask

        # --- Step 1: Normalize and project pair representation ---
        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        # --- Step 2: Inject single rep via outer-sum conditioning ---
        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        # --- Step 3: Compute distogram from predicted coordinates ---
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

        # --- Step 4: Pairwise conditioning (merge z with distogram) ---
        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        # --- Step 5: Build pair mask ---
        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)

        if use_interface_mask:
            # Cross-interface masking (same as affinity head):
            # only ligand-receptor, receptor-ligand, and ligand-ligand pairs
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
            # All valid token pairs (default)
            pair_mask = pad_token_mask[:, :, None] * pad_token_mask[:, None, :]

        # --- Step 6: Refine through PairformerNoSeq stack ---
        z = self.pairformer_stack(
            z,
            pair_mask=pair_mask,
            use_kernels=use_kernels,
        )

        # --- Step 7: Predict evolutionary energy ---
        out_dict = self.evolution_heads(
            z=z,
            feats=feats,
            multiplicity=multiplicity,
            use_interface_mask=use_interface_mask,
        )

        return out_dict


class EvolutionHeads(nn.Module):
    """Prediction head: pool refined pair representation → MLP → scalar energy.

    Pools the refined pair representation z into a global vector via
    masked mean over relevant pair positions (excluding diagonal),
    then projects through a two-stage MLP to produce a scalar
    evolutionary energy E.

    Low E = evolutionarily compatible (e.g. same-species matched pair).
    High E = evolutionarily mismatched (e.g. cross-species swapped pair).

    Parameters
    ----------
    token_z : int
        Pair representation dimension.
    hidden_dim : int
        Hidden dimension for the MLP.
    """

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
            # Cross-interface pooling (same regions as affinity head)
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
            # All-pairs pooling (exclude diagonal self-pairs)
            N = pad_token_mask.shape[1]
            pool_mask = (
                pad_token_mask[:, :, None] * pad_token_mask[:, None, :]
            ) * (
                1
                - torch.eye(N, device=pad_token_mask.device)
                .unsqueeze(-1)
                .unsqueeze(0)
            )

        # Global mean pooling over pair positions → [B*mult, token_z]
        g = torch.sum(z * pool_mask, dim=(1, 2)) / (
            torch.sum(pool_mask, dim=(1, 2)) + 1e-7
        )

        g = self.pool_mlp(g)
        evo_energy = self.to_evo_energy(g).reshape(-1, 1)

        return {"evo_energy": evo_energy}
