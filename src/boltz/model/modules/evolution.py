from pathlib import Path

import torch
from torch import nn

import boltz.model.layers.initialize as init
from boltz.model.layers.pairformer import PairformerNoSeqModule
from boltz.model.modules.encodersv2 import PairwiseConditioning
from boltz.model.modules.utils import LinearNoBias


# The z-track submodules of a trunk PairformerLayer. These are exactly the
# parameters a PairformerNoSeqLayer owns, and they carry the same names, which
# is what lets TrunkTailModule load trunk weights into a NoSeq stack.
Z_TRACK_SUBMODULES = (
    "tri_mul_out",
    "tri_mul_in",
    "tri_att_start",
    "tri_att_end",
    "transition_z",
)


class TrunkTailModule(nn.Module):
    """The last few trunk pairformer layers, re-run (and fine-tuned) at train time.

    The default evolution pipeline caches the trunk's FINAL pair rep ``z`` and
    trains only the head on top of it, so the trunk features are fixed: if they
    do not linearly separate native from swap, no head can fix that. This module
    instead consumes the trunk state captured at the INPUT to trunk pairformer
    layer ``split_layer`` (see ``cache_evolution_features.py --split_layer``) and
    re-runs layers ``split_layer .. num_blocks-1``, letting the last
    ``num_unfrozen`` of them receive gradients. The trunk features themselves can
    then adapt to the task.

    Why a NoSeq stack is exact here, not an approximation
    -----------------------------------------------------
    In ``PairformerLayer.forward`` the pair rep is built only from itself --
    ``tri_mul_out``, ``tri_mul_in``, ``tri_att_start``, ``tri_att_end``,
    ``transition_z``. The sequence track reads ``z`` but never writes back to it,
    so across a stack ``z_{i+1} = f(z_i)`` with no dependence on ``s`` at all.
    ``PairformerNoSeqLayer.forward`` is that same z-track, op for op, with the
    same submodule names. So a PairformerNoSeqModule holding the trunk's z-track
    weights reproduces the trunk's ``z`` exactly, while skipping the s-track --
    which would otherwise burn compute and, since its params could not receive
    gradients from an energy read off ``z``, would trip DDP's unused-parameter
    check. This is also why the cache stores no ``s``.

    Frozen prefix
    -------------
    Layers before the unfrozen suffix are run under ``no_grad``: their output is
    a deterministic function of the cache, so recomputing them costs a forward
    but no activation memory or gradient. They exist so that ONE cache (split at
    layer L) can serve several choices of ``num_unfrozen`` without re-running the
    caching pass, which is ~200 GB and many GPU-hours.

    Parameters
    ----------
    token_z : int
        Pair representation dimension (128 for Boltz-2).
    split_layer : int
        Index of the first trunk pairformer layer in this tail. Must match the
        ``split_layer`` recorded in the cache files.
    num_blocks : int
        Total number of trunk pairformer blocks (64 for Boltz-2).
    num_unfrozen : int
        How many of the trailing layers get gradients. The rest run frozen.
    dropout : float
        Dropout inside the tail layers. Defaults to 0.0, matching the value the
        trunk was pretrained with (PairformerArgsV2.dropout); raising it changes
        the pretrained function.
    """

    def __init__(
        self,
        token_z: int = 128,
        split_layer: int = 60,
        num_blocks: int = 64,
        num_unfrozen: int = 2,
        dropout: float = 0.0,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        n_tail = num_blocks - split_layer
        if n_tail <= 0:
            raise ValueError(
                f"split_layer={split_layer} must be < num_blocks={num_blocks}"
            )
        if not 0 < num_unfrozen <= n_tail:
            raise ValueError(
                f"num_unfrozen={num_unfrozen} must be in (0, {n_tail}] for "
                f"split_layer={split_layer}, num_blocks={num_blocks}"
            )

        self.split_layer = split_layer
        self.num_blocks = num_blocks
        self.num_unfrozen = num_unfrozen
        self.n_tail = n_tail
        self.n_frozen = n_tail - num_unfrozen

        # PairformerNoSeqModule gives us the z-track stack plus the DDP-safe
        # (use_reentrant=False) activation checkpointing already wired up.
        self.stack = PairformerNoSeqModule(
            token_z,
            num_blocks=n_tail,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            activation_checkpointing=activation_checkpointing,
        )

        for layer in self.stack.layers[: self.n_frozen]:
            for p in layer.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        """Params that actually receive gradients (for the optimizer param group)."""
        return [p for p in self.parameters() if p.requires_grad]

    def load_trunk_weights(self, checkpoint_path: str) -> int:
        """Copy trunk z-track weights for layers [split_layer, num_blocks) into the stack.

        Returns the number of tensors loaded. Raises if the checkpoint does not
        contain the expected keys, so a silently randomly-initialized tail can
        never reach training.
        """
        ckpt = torch.load(
            str(Path(checkpoint_path)), map_location="cpu", weights_only=False
        )
        state = ckpt.get("state_dict", ckpt)

        n_loaded = 0
        for i in range(self.n_tail):
            trunk_idx = self.split_layer + i
            prefix = f"pairformer_module.layers.{trunk_idx}."
            sub_state = {
                k[len(prefix) :]: v
                for k, v in state.items()
                if k.startswith(prefix)
                and k[len(prefix) :].split(".")[0] in Z_TRACK_SUBMODULES
            }
            if not sub_state:
                raise KeyError(
                    f"No z-track weights found under '{prefix}' in {checkpoint_path}. "
                    f"Expected keys like '{prefix}tri_mul_out.*'. Is this a Boltz-2 "
                    f"checkpoint, and is num_blocks={self.num_blocks} correct?"
                )
            # strict=True: PairformerNoSeqLayer owns exactly the z-track params,
            # so anything missing or extra means the architectures disagree.
            self.stack.layers[i].load_state_dict(sub_state, strict=True)
            n_loaded += len(sub_state)

        del ckpt, state
        return n_loaded

    def forward(self, z, pair_mask, use_kernels: bool = False):
        # Run the frozen prefix without building a graph, then the trainable
        # suffix through the stack's own (checkpointed) path. Slicing layers
        # directly mirrors PairformerNoSeqModule.forward.
        if self.n_frozen:
            chunk = None
            if not self.training:
                from boltz.data import const

                chunk = 128 if z.shape[1] > const.chunk_size_threshold else 512
            with torch.no_grad():
                for layer in self.stack.layers[: self.n_frozen]:
                    z = layer(z, pair_mask, chunk, use_kernels)
            z = z.detach()

        for i in range(self.n_frozen, self.n_tail):
            layer = self.stack.layers[i]
            if self.stack.activation_checkpointing and self.training:
                z = torch.utils.checkpoint.checkpoint(
                    layer,
                    z,
                    pair_mask,
                    None,
                    use_kernels,
                    use_reentrant=False,
                )
            else:
                chunk = None
                if not self.training:
                    from boltz.data import const

                    chunk = 128 if z.shape[1] > const.chunk_size_threshold else 512
                z = layer(z, pair_mask, chunk, use_kernels)
        return z


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
        head_dropout: float = 0.0,
        zero_init_energy: bool = True,
        use_trunk_tail: bool = False,
        trunk_tail_args: dict = None,
        groups: dict = {},
    ):
        super().__init__()
        self.use_interface_mask = use_interface_mask

        # Optional trunk-tail fine-tuning: when on, the cache holds the pair rep
        # at the INPUT to trunk layer `split_layer`, and this stack re-runs the
        # trailing trunk layers (last `num_unfrozen` trainable) to reproduce and
        # adapt the final trunk `z` before the head sees it. See TrunkTailModule.
        self.use_trunk_tail = use_trunk_tail
        if use_trunk_tail:
            self.trunk_tail = TrunkTailModule(
                token_z=token_z, **(trunk_tail_args or {})
            )
        else:
            self.trunk_tail = None

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
            dropout=head_dropout,
            zero_init_energy=zero_init_energy,
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

        # --- Step 0: Trunk tail (optional) ---
        # Reproduce/adapt the final trunk pair rep from the cached pre-split z.
        # The trunk always attends over ALL valid token pairs (full pad mask),
        # independent of the head's interface-masking choice below.
        if self.use_trunk_tail:
            trunk_pad = feats["token_pad_mask"]
            trunk_pair_mask = trunk_pad[:, :, None] * trunk_pad[:, None, :]
            z = self.trunk_tail(z, pair_mask=trunk_pair_mask, use_kernels=use_kernels)

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
    dropout : float
        Dropout probability applied inside the MLP heads. Regularizes the
        head against memorizing per-complex energies (the main failure mode
        observed: train loss -> 0 while val/loss diverges). 0 disables it.
    """

    def __init__(self, token_z, hidden_dim, dropout: float = 0.0,
                 zero_init_energy: bool = True):
        super().__init__()

        self.pool_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_z, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.to_evo_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # zero_init_energy=True: energies start at exactly 0 → BT loss starts at
        # log(2), margin loss well-behaved at step 0. But it also zeroes the
        # gradient into the pairformer at init, so the head can stall at the
        # log(2) plateau. zero_init_energy=False gives a small nonzero init so
        # energies (and gradients into the trunk) are live from step 0.
        if zero_init_energy:
            init.final_init_(self.to_evo_energy[-1].weight)
        else:
            nn.init.normal_(self.to_evo_energy[-1].weight, std=0.02)
        init.bias_init_zero_(self.to_evo_energy[-1].bias)

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
