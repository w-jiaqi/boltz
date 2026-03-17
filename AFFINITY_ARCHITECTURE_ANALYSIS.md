# Affinity & Binding Likelihood Head: Architecture Analysis and Plan for a New Head

## Table of Contents
1. [High-Level Data Flow](#high-level-data-flow)
2. [Inputs to the Affinity Module](#inputs-to-the-affinity-module)
3. [AffinityModule Architecture (Algorithm 31)](#affinitymodule-architecture)
4. [AffinityHeadsTransformer: The Prediction Heads](#affinityheadstransformer)
5. [How It Integrates into Boltz2](#integration-into-boltz2)
6. [Training Regime](#training-regime)
7. [Comparison with Confidence Module](#comparison-with-confidence-module)
8. [Plan for Training a New Head](#plan-for-training-a-new-head)

---

## High-Level Data Flow

```
Trunk (frozen)                     Diffusion (frozen)
     │                                   │
     ├── s_inputs (token_s=384)          │
     ├── s (token_s=384)                 │
     └── z (token_z=128)                 └── x_pred (best sample by iPTM)
              │                                   │
              │    ┌──────────────────────────────┘
              │    │
              v    v
         ┌─────────────────────────────────────────┐
         │           AffinityModule                 │
         │                                          │
         │  1. z_norm → z_linear (z)                │
         │  2. s → outer product → add to z         │
         │  3. x_pred → distogram → PairwiseCond    │
         │  4. cross_pair_mask (lig×rec + rec×lig)  │
         │  5. PairformerNoSeq stack (z only)        │
         │  6. AffinityHeadsTransformer              │
         │     ├── global pool over cross-pairs      │
         │     ├── MLP                               │
         │     ├── affinity_pred_value (regression)  │
         │     └── affinity_logits_binary (classif.) │
         └─────────────────────────────────────────┘
```

---

## Inputs to the Affinity Module

The `AffinityModule.forward()` receives these inputs:

### 1. `s_inputs` — Single (per-token) representation
- **Shape**: `[B, N, token_s]` where `token_s=384`
- **Source**: `InputEmbedder(feats, affinity=True)` — re-run with `affinity=True`
- **Key difference in affinity mode**: MSA profile is replaced with a single-sequence profile (`max_seqs=1`), stored as `profile_affinity` and `deletion_mean_affinity`. Everything else (atom encoder, res_type encoding) stays the same.
- **Detached** from trunk gradients (`.detach()`)

### 2. `z` — Pair representation (cross-pair masked)
- **Shape**: `[B, N, N, token_z]` where `token_z=128`
- **Source**: Output of the trunk's pairformer stack
- **Pre-masking**: Before being passed to the affinity module, `z` is element-wise multiplied by `cross_pair_mask` in `Boltz2.forward()`:
  ```python
  z_affinity = z * cross_pair_mask[None, :, :, None]
  ```
  where `cross_pair_mask = lig×rec + rec×lig + lig×lig` — only interface and intra-ligand pairs are non-zero
- **Detached** from trunk gradients

### 3. `x_pred` — Predicted 3D coordinates
- **Shape**: `[1, 1, N_atoms, 3]`
- **Source**: Best diffusion sample, selected by highest iPTM score:
  ```python
  argsort = torch.argsort(dict_out["iptm"], descending=True)
  best_idx = argsort[0].item()
  coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][None, None]
  ```
- **Detached** from structure module gradients

### 4. `feats` — Feature dictionary
Key features used:
- `token_to_rep_atom`: Maps tokens to their representative atom (for computing token-level distances from atom coordinates)
- `affinity_token_mask`: Binary mask marking which tokens belong to the binder/ligand chain
- `token_pad_mask`: Padding mask
- `mol_type`: Chain type (0=protein, 1=DNA, 2=RNA, 3=nonpolymer)

---

## AffinityModule Architecture

**File**: `src/boltz/model/modules/affinity.py`

### Step-by-step forward pass:

#### Step 1: Pair representation initialization
```python
z = self.z_linear(self.z_norm(z))    # LayerNorm + Linear(token_z → token_z)
```

#### Step 2: Augment z with single representation outer product
```python
z = z + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]    # [B, N, 1, token_z]
      + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]    # [B, 1, N, token_z]
```
This injects per-token information into the pair representation via an additive outer-product-like operation (but without explicit multiplication — it's two separate linear projections added, not a true outer product).

#### Step 3: Compute distogram from predicted coordinates
```python
x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)  # atom→token coords
d = torch.cdist(x_pred_repr, x_pred_repr)                   # pairwise distances
distogram = (d.unsqueeze(-1) > self.boundaries).sum(-1).long()  # bin index
distogram = self.dist_bin_pairwise_embed(distogram)          # Embedding(64, token_z)
```
- Distance boundaries: linearly spaced from 2 to `max_dist` (22 Å), 63 boundaries → 64 bins
- Each bin index is embedded into a `token_z`-dimensional vector

#### Step 4: PairwiseConditioning — combine trunk z with distogram
```python
z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)
```
The `PairwiseConditioning` module:
1. Concatenates `z` and `distogram` along the feature dimension → `[B, N, N, 2*token_z]`
2. LayerNorm + Linear projection → `[B, N, N, token_z]`
3. Two transition blocks (Linear→ReLU→Linear) with residual connections

#### Step 5: Cross-pair masking
```python
cross_pair_mask = lig_mask[:,:,None] * rec_mask[:,None,:]    # lig→rec
                + rec_mask[:,:,None] * lig_mask[:,None,:]    # rec→lig
                + lig_mask[:,:,None] * lig_mask[:,None,:]    # lig→lig
```
Where:
- `lig_mask` = `affinity_token_mask` (binder chain tokens)
- `rec_mask` = `mol_type == 0` (protein tokens)

#### Step 6: PairformerNoSeq stack
```python
z = self.pairformer_stack(z, pair_mask=cross_pair_mask, use_kernels=use_kernels)
```
This is `PairformerNoSeqModule` — a pairformer **without** a sequence track (no single representation updates). Each layer consists of:
- Triangle multiplication outgoing (with dropout)
- Triangle multiplication incoming (with dropout)
- Triangle attention starting node (with dropout)
- Triangle attention ending node (with dropout)
- Pair transition (Linear→ReLU→Linear + residual)

#### Step 7: Prediction heads
```python
out_dict = self.affinity_heads(z=z, feats=feats, multiplicity=multiplicity)
```

---

## AffinityHeadsTransformer

**File**: `src/boltz/model/modules/affinity.py`, class `AffinityHeadsTransformer`

### Global pooling over cross-pair positions
```python
# Rebuild cross_pair_mask (with self-diagonal excluded)
cross_pair_mask = (lig×rec + rec×lig + lig×lig) * (1 - I)

# Weighted mean pooling
g = sum(z * cross_pair_mask, dim=(1,2)) / (sum(cross_pair_mask, dim=(1,2)) + 1e-7)
# g shape: [B, token_z]
```

### Shared MLP trunk
```python
g = self.affinity_out_mlp(g)
# affinity_out_mlp = Linear(token_z, token_z) → ReLU → Linear(token_z, input_token_s) → ReLU
# g shape: [B, input_token_s]
```

### Head 1: Affinity value prediction (regression)
```python
affinity_pred_value = self.to_affinity_pred_value(g)
# to_affinity_pred_value = Linear(s, s) → ReLU → Linear(s, s) → ReLU → Linear(s, 1)
# Output: scalar log10(IC50) value
```

### Head 2: Binding likelihood (binary classification)
```python
affinity_pred_score = self.to_affinity_pred_score(g)
# to_affinity_pred_score = Linear(s, s) → ReLU → Linear(s, s) → ReLU → Linear(s, 1)
affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score)
# to_affinity_logits_binary = Linear(1, 1)
# At inference: sigmoid(affinity_logits_binary) → probability
```

The binding likelihood is a two-stage architecture: first produce a "score" with a deep MLP, then a simple linear layer maps it to binary logits. This allows the score to capture intermediate binding information that the final linear layer calibrates.

---

## Integration into Boltz2

**File**: `src/boltz/model/models/boltz2.py`, `Boltz2.forward()`

### Execution order:
1. **Trunk**: InputEmbedder → MSA module → Pairformer stack → produces `s`, `z`, `s_inputs`
2. **Structure**: Diffusion sampling → produces `sample_atom_coords`
3. **Confidence**: ConfidenceModule → produces iPTM, pLDDT, etc.
4. **Affinity** (only if `affinity_prediction=True`):
   - Mask `z` to keep only cross-interface pairs
   - Select best diffusion sample by iPTM
   - Re-run `InputEmbedder` with `affinity=True` (different MSA features)
   - Run `AffinityModule` with detached inputs
   - Apply sigmoid to logits for probability
   - Optionally apply MW correction: `model_coef * pred_value + mw_coef * MW^0.3 + bias`

### Ensemble mode:
When `affinity_ensemble=True`, two independent `AffinityModule` instances run on the same inputs, and their outputs are averaged:
```python
ensemble_pred_value = (module1_pred_value + module2_pred_value) / 2
ensemble_prob_binary = (module1_prob_binary + module2_prob_binary) / 2
```

---

## Training Regime

### What is frozen vs. trained:
```python
if not structure_prediction_training:
    for name, param in self.named_parameters():
        if name.split(".")[0] not in ["confidence_module", "affinity_module"]
           and "out_token_feat_update" not in name:
            param.requires_grad = False
```
- **Frozen**: Trunk (InputEmbedder, MSA, Pairformer), Diffusion/Structure module, Distogram module
- **Trained**: `confidence_module`, `affinity_module`, `out_token_feat_update`

### Data pipeline for affinity:
1. **AffinityCropper**: Spatial cropping centered on the ligand
   - Computes minimum distance from each token to ligand atoms
   - Sorts by distance, greedily adds neighborhoods up to `max_tokens`
   - Limits protein tokens to `max_tokens_protein=200`
2. **Featurizer**: Produces `affinity_token_mask` from token data
3. **MSA features**: For affinity, produces single-sequence profile (`max_seqs=1`)
4. **Affinity MW**: Ligand molecular weight stored for optional correction

### Loss functions:
The affinity head uses a **separate checkpoint** (`boltz2_aff.ckpt`). The loss functions are not present in the open-source code — the training is done with proprietary affinity labels (IC50 values, binding/non-binding labels). However, based on the architecture:
- **Value head**: Likely MSE or Huber loss on log10(IC50)
- **Binary head**: Likely BCE loss on binding/non-binding labels

### Optimizer:
- AdamW with betas=(0.9, 0.95), eps=1e-8
- AlphaFold-style LR scheduler with warmup and decay
- Weight decay with exclusions for norms, embeddings, and position encodings

---

## Comparison with Confidence Module

| Aspect | AffinityModule | ConfidenceModule |
|--------|---------------|------------------|
| Pairformer type | `PairformerNoSeqModule` (pair only) | `PairformerModule` (pair + sequence) |
| Input masking | Cross-pair mask (interface only) | Full pair mask (all pairs) |
| Distance conditioning | Distogram via `PairwiseConditioning` | Distogram added directly to `z` |
| Single repr in pairformer | Not used (no seq track) | Updated via attention pair bias |
| Output granularity | Global scalar (pooled) | Per-token (pLDDT) + per-pair (PDE, PAE) |
| Prediction targets | Regression + binary classification | Multi-class classification (binned) |
| Pooling strategy | Mean over cross-pair positions | Per-element or per-pair outputs |

---

## Plan for Training a New Head

### Architecture template (following affinity pattern):

```
NewModule(nn.Module):
├── z_norm (LayerNorm)
├── z_linear (Linear, token_z → token_z)
├── s_to_z_prod_in1 (LinearNoBias, token_s → token_z)
├── s_to_z_prod_in2 (LinearNoBias, token_s → token_z)
├── dist_bin_pairwise_embed (Embedding, num_dist_bins × token_z)
├── pairwise_conditioner (PairwiseConditioning)
├── pairformer_stack (PairformerNoSeqModule)
└── prediction_heads (NewHeadsTransformer)
    ├── global_pool (mean over relevant pair positions)
    ├── shared_mlp (Linear → ReLU → Linear → ReLU)
    └── output_head(s) (Linear → ReLU → Linear → ReLU → Linear → output_dim)
```

### Steps to implement:

1. **Define the new module** in `src/boltz/model/modules/your_head.py`
   - Copy `AffinityModule` and `AffinityHeadsTransformer` as starting points
   - Decide on your masking strategy (cross-pair? all pairs? different subsets?)
   - Decide on output: regression, classification, or both
   - Decide on pooling: global mean? Attention-weighted? Per-token?

2. **Integrate into Boltz2** in `src/boltz/model/models/boltz2.py`
   - Add `new_head_args` parameter to `__init__`
   - Add `new_head_prediction: bool = False` flag
   - Instantiate `NewModule` in `__init__`
   - Add forward pass after confidence (follows same detach pattern)
   - Add parameter freezing exclusion for `new_module`

3. **Add data pipeline support**
   - Add any new token-level masks in `src/boltz/data/types.py`
   - Add feature computation in `src/boltz/data/feature/featurizerv2.py`
   - Optionally add a custom cropper in `src/boltz/data/crop/`

4. **Add loss functions** in `src/boltz/model/loss/your_loss.py`
   - For regression: MSE/Huber on your target values
   - For classification: BCE or cross-entropy
   - Add to `training_step` in `boltz2.py`

5. **Add training config** in `scripts/train/configs/your_head.yaml`
   - Set `structure_prediction_training: false`
   - Provide `new_head_args` with pairformer config
   - Set loss weights

6. **Add inference support**
   - Update `predict_step` to include new outputs
   - Add writer support in `src/boltz/data/write/`
   - Update CLI in `src/boltz/main.py`

### Key hyperparameters to decide:

| Parameter | Affinity default | Notes |
|-----------|-----------------|-------|
| `num_dist_bins` | 64 | Number of distance bins for distogram |
| `max_dist` | 22 Å | Maximum distance for binning |
| Pairformer blocks | Configurable | More blocks = more capacity but slower |
| `pairwise_head_width` | 32 | Width of triangle attention heads |
| `pairwise_num_heads` | 4 | Number of triangle attention heads |
| `token_z` (internal) | 128 | Pair representation dimension (matches trunk) |
| `input_token_s` | from transformer_args | Single representation dimension for heads |
| Dropout | 0.25 | Pairformer dropout |

### Critical design decisions:

1. **What pairs to attend over**: The affinity module uses cross-pair masking (ligand-receptor interface). Your head should define its own masking based on the prediction task.

2. **Whether to use sequence track**: Affinity uses `PairformerNoSeqModule` (pair only). If your task needs per-token predictions, consider using the full `PairformerModule` (like confidence does).

3. **Pooling strategy**: Affinity does simple mean pooling over cross-pair positions. For different tasks, consider:
   - Attention-weighted pooling
   - Per-token output (if predicting per-residue properties)
   - Per-pair output (if predicting pairwise properties)

4. **Input detachment**: Following the affinity pattern, inputs from the trunk and structure module should be `.detach()`ed to avoid backpropagating through the frozen trunk.

5. **Separate checkpoint**: The affinity module is trained and loaded as a separate checkpoint. Your new head could follow the same pattern or be trained jointly with confidence.
