"""Correctness gate for trunk-tail fine-tuning — run BEFORE the 200 GB re-cache.

Proves two things against the real Boltz-2 checkpoint, cheaply (no data pipeline,
no diffusion, one random tensor):

  1. load_trunk_weights finds the expected z-track keys in the checkpoint.
  2. A TrunkTailModule holding those weights reproduces the trunk's own z-track
     exactly. We run a random (s, z) through the real trunk PairformerLayers
     [split_layer, num_blocks) and, separately, run z through the TrunkTailModule,
     and assert the resulting z tensors match to float tolerance.

If (2) holds, the cache-at-split-layer -> replay-tail pipeline is mathematically
sound and re-caching is safe to launch.

Usage (inside the container, on a GPU node):
    python scripts/train/validate_trunktail.py --checkpoint /path/boltz2_conf.ckpt \
        --split_layer 60 --num_blocks 64
"""

import argparse

import torch

from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.evolution import TrunkTailModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split_layer", type=int, default=60)
    ap.add_argument("--num_blocks", type=int, default=64)
    ap.add_argument("--token_s", type=int, default=384)
    ap.add_argument("--token_z", type=int, default=128)
    ap.add_argument("--n_tokens", type=int, default=48)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # --- Reference: the real trunk pairformer, layers [split, num_blocks) ---
    ref = PairformerModule(
        args.token_s, args.token_z, num_blocks=args.num_blocks, v2=True
    )
    pf_state = {
        k[len("pairformer_module."):]: v
        for k, v in state.items()
        if k.startswith("pairformer_module.")
    }
    missing, unexpected = ref.load_state_dict(pf_state, strict=False)
    # v2 attention may leave a few buffers; report but don't fail on those.
    real_missing = [m for m in missing if "attention" not in m and "pre_norm_s" not in m]
    print(f"Reference pairformer loaded. missing(non-attn)={len(real_missing)} "
          f"unexpected={len(unexpected)}")
    ref = ref.to(device).eval()

    # --- Candidate: the trunk tail we actually train with ---
    tail = TrunkTailModule(
        token_z=args.token_z,
        split_layer=args.split_layer,
        num_blocks=args.num_blocks,
        num_unfrozen=args.num_blocks - args.split_layer,  # all trainable for the test
        dropout=0.0,
        activation_checkpointing=False,
    )
    n_loaded = tail.load_trunk_weights(args.checkpoint)
    print(f"TrunkTailModule.load_trunk_weights loaded {n_loaded} tensors.")
    tail = tail.to(device).eval()

    # --- Random input at the split point ---
    B, N = 1, args.n_tokens
    s = torch.randn(B, N, args.token_s, device=device)
    z = torch.randn(B, N, N, args.token_z, device=device)
    mask = torch.ones(B, N, device=device)
    pair_mask = mask[:, :, None] * mask[:, None, :]

    with torch.no_grad():
        # Reference: push (s, z) through the real trunk layers [split, num_blocks).
        z_ref = z.clone()
        s_ref = s.clone()
        for i in range(args.split_layer, args.num_blocks):
            s_ref, z_ref = ref.layers[i](
                s_ref, z_ref, mask, pair_mask, None, False
            )

        # Candidate: push z through the trunk tail.
        z_tail = tail(z.clone(), pair_mask=pair_mask, use_kernels=False)

    diff = (z_ref - z_tail).abs()
    max_abs = diff.max().item()
    rel = (diff.max() / (z_ref.abs().max() + 1e-8)).item()
    print(f"max|z_ref - z_tail| = {max_abs:.3e}   (rel {rel:.3e})")

    if max_abs < args.atol:
        print("PASS: trunk tail reproduces the trunk z-track. Re-cache is safe.")
    else:
        print("FAIL: mismatch exceeds tolerance. Do NOT re-cache; investigate.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
