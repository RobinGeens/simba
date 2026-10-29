"""Which exp the installed selective-scan CUDA kernel uses: "exact" (upstream mamba) or "approx" (fp16 hardware model)."""

import torch
from mamba_ssm.ops import selective_scan_interface  # loads selective_scan_cuda after torch (it needs libc10)


def scan_exp_mode(device="cuda"):
    """Run the forward kernel on inputs whose second output is exp(A) and compare with torch.

    seqlen 2, dstate 1, u = [1, 0], delta = 1, B = C = 1: h1 = 1, h2 = exp(A) * h1, so y[..., 1] = exp(A).
    Max relative error on A in [-8, 0]: exact ~4e-7, approx ~7e-2.
    """
    n = 4096  # channels sit on grid.y (max 65535)
    A = torch.linspace(-8.0, 0.0, n, device=device).view(n, 1)
    u = torch.tensor([1.0, 0.0], device=device).expand(1, n, 2).contiguous()
    ones = torch.ones(n, 1, device=device)
    with torch.no_grad():
        y = selective_scan_interface.selective_scan_fn(u, torch.ones(1, n, 2, device=device), A, ones, ones)
    ref = A.view(n).double().exp()
    err = ((y[0, :, 1].double() - ref).abs() / ref).max().item()
    mode = "exact" if err < 1e-5 else "approx" if err > 1e-3 else "unknown"
    return mode, err


def check_scan_exp(expected, device="cuda", log=print):
    """Raise if the kernel's exp is not the one the model expects (model.MAMBA_EXP)."""
    if expected not in ("exact", "approx"):
        raise ValueError(f'The model must declare MAMBA_EXP = "exact" or "approx", got {expected!r}')
    mode, err = scan_exp_mode(device)
    so_file = selective_scan_interface.selective_scan_cuda.__file__
    log(f"Selective-scan kernel exp: {mode} (max rel err {err:.1e}; {so_file})")
    if mode != expected:
        raise RuntimeError(
            f"The selective-scan kernel uses the {mode} exp, but this model expects {expected}. "
            "Rebuild mamba (scripts/install_mamba.sh) or restore the matching build from env_backups/."
        )
    return mode
