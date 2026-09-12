"""fp32 -> bf16 safetensors conversion + weight inventory printer.

Walks a directory of *.safetensors, casts floating tensors to bf16,
writes them alongside (or into --out-dir), and prints an inventory:
per-file tensor count, dtype histogram, size on disk.

Run: PYTHONPATH=voice-pipeline python scripts/convert_weights.py SRCDIR
       [--out-dir DST] [--inventory-only]
"""
import argparse
import collections
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inventory(paths):
    total_b = 0
    for p in paths:
        try:
            from safetensors import safe_open
        except Exception as exc:
            print(f"OMITTED: safetensors missing ({exc})")
            return 2
        dtypes = collections.Counter()
        n = 0
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                dtypes[str(t.dtype)] += 1
                n += 1
        size = os.path.getsize(p)
        total_b += size
        print(f"{os.path.basename(p)}: {n} tensors {dict(dtypes)} {size / 1e6:.1f}MB")
    print(f"total {len(paths)} files {total_b / 1e6:.1f}MB")
    return 0


def convert(src, dst):
    import torch
    from safetensors.torch import load_file, save_file

    sd = load_file(src)
    out = {}
    n_cast = 0
    for k, v in sd.items():
        if torch.is_floating_point(v) and v.dtype == torch.float32:
            out[k] = v.to(torch.bfloat16)
            n_cast += 1
        else:
            out[k] = v
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    save_file(out, dst)
    print(f"{os.path.basename(src)}: cast {n_cast}/{len(sd)} fp32->bf16 -> {dst}")
    return n_cast


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("srcdir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--inventory-only", action="store_true")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.srcdir, "*.safetensors")))
    if not paths:
        print(f"no safetensors in {args.srcdir}")
        return 1
    if args.inventory_only:
        return inventory(paths)
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file  # noqa: F401
    except Exception as exc:
        print(f"OMITTED: torch/safetensors unavailable ({exc})")
        return 2
    out_dir = args.out_dir or (args.srcdir.rstrip("/") + "-bf16")
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        convert(p, os.path.join(out_dir, os.path.basename(p)))
    return inventory(sorted(glob.glob(os.path.join(out_dir, "*.safetensors"))))


if __name__ == "__main__":
    raise SystemExit(main())
