"""Same as PGD in cleverhands, but with additional statistics tracking."""
import numpy as np
import torch

from cleverhans.torch.utils import optimize_linear
from cleverhans.torch.utils import clip_eta

def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
    return_grad: bool = False,
):
    """
    Same as original Fast Gradient Method, but with optional gradient return.
    """
    # If input is 3D (single image), automatically add batch dim
    added_batch_dim = False
    if x.dim() == 3:                  # (C, H, W)
        x = x.unsqueeze(0)            # -> (1, C, H, W)
        added_batch_dim = True
        if y is not None and y.dim() == 0:
            y = y.unsqueeze(0)        # Scalar label -> (1,)

    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return (x, torch.zeros_like(x)) if return_grad else x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)
    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf with requires_grad=True
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        with torch.no_grad():
            _, y = torch.max(model_fn(x), 1)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)
    if targeted:
        loss = -loss

    # Backprop to get input gradient
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()
    g = x.grad.detach()  # Save gradient for PGD statistics

    optimal_perturbation = optimize_linear(g, eps, norm)
    adv_x = x + optimal_perturbation

    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError("One-sided clipping not supported")
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)

    # If originally 3D, squeeze back
    if return_grad:
        if added_batch_dim:
            return adv_x.squeeze(0), g.squeeze(0)
        return adv_x, g
    else:
        if added_batch_dim:
            return adv_x.squeeze(0)
        return adv_x
    
def _unit_dir(flat_g, eps=1e-12):
    # Normalize each sample by L2 norm to get direction (unit vector)
    gn = flat_g.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return flat_g / gn

def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
    return_stats: bool = False,
    track_per_sample: bool = False,
    track_K: int = 16,
):
    """
    Compatible with original PGD parameters/behavior.
    When return_stats=True, returns (adv_x, stats); otherwise, returns adv_x only.
    stats includes:
      - cos_to_prev: [T]
      - sign_consistency: [T]
      - grad_norm: [T]
      - dir_var: float
      - (optional) sample_dirs: Tensor[T, K, D]
    """
    # If input is 3D (single image), automatically add batch dim
    added_batch_dim = False
    if x.dim() == 3:               # (C, H, W)
        x = x.unsqueeze(0)         # -> (1, C, H, W)
        added_batch_dim = True
        if y is not None and y.dim() == 0:  # Scalar label
            y = y.unsqueeze(0)     # -> (1,)

    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop step for PGD when norm=1."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if eps == 0:
        return (x, None) if return_stats else x
    if eps_iter < 0:
        raise ValueError(f"eps_iter must be >= 0, got {eps_iter}")
    if eps_iter == 0:
        return (x, None) if return_stats else x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError(f"clip_min ({clip_min}) must be <= clip_max ({clip_max})")

    asserts = []
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)
    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialization
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Constrain initial perturbation
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    # Avoid label leaking
    if y is None:
        with torch.no_grad():
            _, y = torch.max(model_fn(x), 1)

    # ---- Statistics containers ----
    if return_stats:
        cos_to_prev_list = []
        sign_consistency_list = []
        grad_norm_list = []
        mean_dir_over_batch = []
        dir_var_step_list = [] 
        if track_per_sample:
            K = min(track_K, x.size(0))
            sample_dirs = torch.zeros(nb_iter, K, x[0].numel(), device=x.device)
        prev_g_flat = None

    # ---- Main loop ----
    i = 0
    while i < nb_iter:
        # Call FGM for one step (step size = eps_iter) and get the input gradient for this step
        adv_x_step, g = fast_gradient_method(
            model_fn,
            adv_x,
            eps=eps_iter,          # One step size
            norm=norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
            sanity_checks=False,
            return_grad=True,       # Get gradient for this step
        )

        if return_stats:
            B = g.size(0)
            g_flat = g.view(B, -1)
            g_dir = _unit_dir(g_flat)                    # [B,D]
            g_norm = g_flat.norm(p=2, dim=1)             # [B]

            grad_norm_list.append(g_norm.mean().item())

            if prev_g_flat is None:
                cos_to_prev = torch.full((B,), float('nan'), device=g.device)
                sign_consistency = torch.full((B,), float('nan'), device=g.device)
            else:
                prev_dir = _unit_dir(prev_g_flat)
                cos_to_prev = (g_dir * prev_dir).sum(dim=1).clamp(-1, 1)
                sign_consistency = (torch.sign(g_flat) == torch.sign(prev_g_flat)).float().mean(dim=1)

            cos_to_prev_list.append(cos_to_prev.mean().item())
            sign_consistency_list.append(sign_consistency.mean().item())
            mean_dir_over_batch.append(g_dir.mean(dim=0))   # [D]
            
            mean_dir = g_dir.mean(dim=0, keepdim=True)                  # [1, D]
            mean_dir = mean_dir / (mean_dir.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12))
            mean_dir = torch.nan_to_num(mean_dir, nan=0.0)
            align_to_mean = (g_dir * mean_dir).sum(dim=1)               # [B]
            dir_var_step_t = 1.0 - align_to_mean.mean().item()
            dir_var_step_list.append(dir_var_step_t)

            if track_per_sample:
                sample_dirs[i, :K] = g_dir[:K]

            prev_g_flat = g_flat

        # Projection to eps-ball consistent with original version
        # (Note: FGM internally clips to data domain; here we maintain the norm constraint on the cumulative perturbation)
        eta = adv_x_step - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

        i += 1

    # Final checks
    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        asserts.append(eps + clip_min <= clip_max)
    if sanity_checks:
        assert np.all(asserts)

    if not return_stats:
        # If 3D originally, restore to 3D
        if added_batch_dim:
            adv_x = adv_x.squeeze(0)
        return adv_x

    if added_batch_dim:
        adv_x = adv_x.squeeze(0)

    # ---- Aggregate dir_var ----
    if len(mean_dir_over_batch) > 1:
        md = torch.stack(mean_dir_over_batch, dim=0)               # [T,D]
        md = md / (md.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12))
        md = torch.nan_to_num(md, nan=0.0)
        sim = md @ md.t()                                          # [T,T]
        tri = torch.triu(torch.ones_like(sim), diagonal=1).bool()
        if tri.any():
            mean_pair_cos = sim[tri].mean().item()
            dir_var_time = 1.0 - float(mean_pair_cos)
        else:
            dir_var_time = float('nan')
    else:
        dir_var_time = float('nan')

    stats = {
        "cos_to_prev": cos_to_prev_list,
        "sign_consistency": sign_consistency_list,
        "grad_norm": grad_norm_list,
        "dir_var": dir_var_time,
        "dir_var_step": dir_var_step_list # New "per-step within-batch dispersion"
    }
    if track_per_sample:
        stats["sample_dirs"] = sample_dirs  # [T,K,D]

    return adv_x, stats