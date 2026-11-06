"""Flat-gradient helpers and the leak projection (one-pager section 6)."""
import torch


def vjp(outputs, grad_outputs, params) -> torch.Tensor:
    """Flat gradient of sum(<outputs, grad_outputs>) w.r.t. params (graph kept)."""
    gs = torch.autograd.grad(outputs, params, grad_outputs, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(gs, params, strict=True)])


def assign(params, flat: torch.Tensor) -> None:
    """Write a flat gradient into .grad of params."""
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n


def adam_metric(opt: torch.optim.Optimizer, params) -> torch.Tensor | None:
    """Diagonal preconditioner 1 / (sqrt(v_hat) + eps) of Adam(W), flat; None before step 1.

    Adam's update is (approximately) -lr * metric * g, so the leak of a context step is
    g_tgt . (metric * g_ctx) rather than g_tgt . g_ctx.
    """
    st = [opt.state.get(p, {}) for p in params]
    if not st or "exp_avg_sq" not in st[0]:
        return None
    b2, eps = opt.param_groups[0]["betas"][1], opt.param_groups[0]["eps"]
    out = []
    for s in st:
        k = float(s["step"])
        out.append((1 / ((s["exp_avg_sq"] / (1 - b2**k)).sqrt() + eps)).reshape(-1))
    return torch.cat(out)


def project(g_ctx, g_tgt, alpha=1.0, metric=None):
    """Remove the leak component of g_ctx along g_tgt (only if it lowers the loss via the target).

    ell = g_tgt . M g_ctx (M = metric or identity). If ell > 0:
        g' = g_ctx - alpha * ell / (g_tgt . M g_tgt) * g_tgt,   so  g_tgt . M g' = (1-alpha) ell.
    Returns (g', ell).
    """
    m = 1.0 if metric is None else metric
    ell = torch.dot(g_tgt, m * g_ctx)
    if ell > 0:
        g_ctx = g_ctx - alpha * ell / torch.dot(g_tgt, m * g_tgt).clamp_min(1e-30) * g_tgt
    return g_ctx, ell.item()
