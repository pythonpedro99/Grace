"""Flat-gradient helpers and the leak projection on the actual AdamW step (one-pager 6, 10).

Why a custom optimizer: the one-pager derives the projection for plain SGD. With AdamW the
encoder step is  -lr * (M m_hat + wd theta)  with M = 1/(sqrt(v_hat)+eps): a preconditioned
*momentum* (about 90% past gradients at beta1 = 0.9) plus decoupled weight decay. Projecting only
the current gradient leaves both the momentum and the decay free to keep leaking. SplitAdamW
keeps one first moment for the context path and one for everything else (their sum is exactly
AdamW's), and projects the context part of the update that is actually applied.
"""
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def vjp(outputs, grad_outputs, params) -> torch.Tensor:
    """Flat gradient of sum(<outputs, grad_outputs>) w.r.t. params (graph kept)."""
    gs = torch.autograd.grad(outputs, params, grad_outputs, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(gs, params, strict=True)])


def project(g, g_S, alpha=1.0, metric=None):
    """Remove the leak component of g along g_S (only if it lowers the loss via the target).

    ell = g_S . M g (M = metric or identity). If ell > 0:
        g' = g - alpha * ell / (g_S . M g_S) * g_S,   so  g_S . M g' = (1 - alpha) ell.
    Returns (g', ell).
    """
    m = 1.0 if metric is None else metric
    ell = torch.dot(g_S, m * g)
    if ell > 0:
        g = g - alpha * ell / torch.dot(g_S, m * g_S).clamp_min(1e-30) * g_S
    return g, ell.item()


class SplitAdamW:
    """AdamW on a flat parameter vector with split first moments (context path / the rest).

        v      <- b2 v + (1-b2) (g_ctx + g_rest)^2         shared second moment, raw gradients
        M       = 1 / (sqrt(v_hat) + eps)
        u_ctx   = M m_ctx_hat + wd theta                    context-side update (decay included)
        u_rest  = M m_rest_hat
        theta  <- theta - lr (u_ctx + u_rest)

    Leak of the context step (first order, per unit lr): ell = g_S . u_ctx.
    proj_on="update" (default): if ell > 0, remove c g_S from the context momentum with
        c = alpha ell / (g_S . M g_S), so that g_S . u_ctx = (1 - alpha) ell exactly.
    proj_on="grad": the one-pager sketch, the current g_ctx is projected (Euclidean) before it
        enters the momentum. Momentum and decay are not controlled: an ablation.
    sgd=True: no momentum, M = 1; then both modes coincide with the one-pager's SGD derivation.
    Without projection this is exactly AdamW (momentum is linear in the gradient).
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.0, sgd=False):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd, self.sgd = lr, betas, eps, wd, sgd
        th = parameters_to_vector(self.params)
        self.m_ctx, self.m_rest, self.v = (torch.zeros_like(th) for _ in range(3))
        self.t = 0

    @torch.no_grad()
    def step(self, g_ctx, g_rest, g_S=None, alpha=0.0, proj_on="update") -> dict:
        """Apply one step. Returns diagnostics: leak before / after projection, the cosine of
        g_S and the context update, the norm ratio after / before, and the applied context
        update u_ctx (the parameter step of the context side is -lr * u_ctx)."""
        self.t += 1
        th = parameters_to_vector(self.params)
        b1 = 0.0 if self.sgd else self.b1
        if self.sgd:
            M = torch.ones_like(th)
        else:
            g = g_ctx + g_rest
            self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            M = 1 / ((self.v / (1 - self.b2**self.t)).sqrt() + self.eps)
        bc = 1 - b1**self.t

        def u_of(m):
            return M * m / bc + self.wd * th

        m_raw = b1 * self.m_ctx + (1 - b1) * g_ctx
        u_pre = u_of(m_raw)
        out = {}
        if g_S is not None:
            pre = torch.dot(g_S, u_pre)
            out["leak_pre"] = pre.item()
            out["leak_cos"] = (pre / (g_S.norm() * u_pre.norm()).clamp_min(1e-30)).item()
        if g_S is not None and alpha > 0 and proj_on == "grad":
            g_ctx, _ = project(g_ctx, g_S, alpha)
            m_raw = b1 * self.m_ctx + (1 - b1) * g_ctx
        elif g_S is not None and alpha > 0 and out["leak_pre"] > 0:
            c = alpha * out["leak_pre"] / torch.dot(g_S, M * g_S).clamp_min(1e-30)
            m_raw = m_raw - c * bc * g_S
        self.m_ctx = m_raw
        self.m_rest = b1 * self.m_rest + (1 - b1) * g_rest
        u_ctx = u_of(self.m_ctx)
        if g_S is not None:
            out["leak_post"] = torch.dot(g_S, u_ctx).item()
            out["proj_ratio"] = (u_ctx.norm() / u_pre.norm().clamp_min(1e-30)).item()
        out["u_ctx"] = u_ctx
        vector_to_parameters(th - self.lr * (u_ctx + M * self.m_rest / bc), self.params)
        return out
