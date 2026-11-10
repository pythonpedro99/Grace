"""Grid world: a Craftax-shaped tile world with the same four kinds of content as the clip world.

    slow   the HUD: n_slow inventory-like slots, integers 0..levels-1 drawn as tile brightness. Each
           slot steps by +-1 with probability hud_p per frame (uniform stationary law), drawn i.i.d.
           per clip, so the HUD alone can satisfy SIGReg. Also the agent's position (x, y): one-cell
           moves leave it ~96% predictable without the action, so it is slow content too.
    fast   the agent's facing direction (one-hot). A move action turns the agent even when the move
           is blocked (as in Craftax): it changes in ~60% of steps and only the action tells how.
           Drawn as the yellow half of the white agent tile.
    hard   the "hand": the material of the cell the agent faces after its action (one-hot over
           empty, wood, stone, water), shown in the HUD slot after the inventory. Predicting it
           needs the action, the new position and the materials around the agent: a pattern
           (two chained lookups) the predictor has to learn.
    noise  a flicker mob (green) that jumps to a random free cell every frame, and n_twinkle fixed
           floor cells with a random brightness every frame. Unpredictable.

The map holds wood (brown), stone (grey) and water (blue) on a share `density` of the cells; they
block movement, and the border counts as stone. random_map: a new map per clip (as Craftax), else
one map fixed by the seed. Actions are one-hot over (noop, up, down, left, right); the predictor sees
all T-1 of them. The frame is W x W tiles: H map rows, then hud_rows HUD rows.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .env import GROUPS, Clip

ACT_DIM = 5
MOVES = torch.tensor([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])  # (dx, dy): noop, up, down, left, right
DIRS = MOVES[1:]  # facing 0..3 = up, down, left, right
N_MAT = 4  # empty (floor), wood, stone, water
EMPTY, WOOD, STONE, WATER = range(N_MAT)
MAT_COL = torch.tensor([[0.12, 0.12, 0.12], [0.55, 0.33, 0.12], [0.55, 0.55, 0.55],
                        [0.15, 0.35, 0.85]])
AGENT, FACE = torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 0.85, 0.0])
MOB, TWINKLE = torch.tensor([0.2, 0.9, 0.2]), torch.tensor([0.9, 0.2, 0.8])


@dataclass(frozen=True)
class GridCfg:
    W: int = 10  # map width in cells = frame width in tiles
    H: int = 8  # map rows; W - H HUD rows below them make the frame square
    tile: int = 4  # pixels per cell; res = tile * W must be divisible by 8
    T: int = 3  # frames per clip; the last one is the target
    n_slow: int = 16  # HUD slots (the hand takes the next slot)
    levels: int = 10  # values per HUD slot
    hud_p: float = 0.03  # per-frame probability that a slot steps by +-1
    density: float = 0.35  # share of map cells holding a material
    random_map: bool = True  # a new map per clip (False: one map, fixed by the seed)
    n_twinkle: int = 4  # noise tiles (always floor)
    seed: int = 0  # fixes the HUD colours, the twinkle cells and (random_map=False) the map

    @property
    def res(self) -> int:
        return self.tile * self.W


class GridEnv:
    """Samples i.i.d. clips on `device`. Uses the global torch RNG (seed it per run)."""

    act_dim = ACT_DIM
    gt_extra = ("local",)  # extra ground-truth inputs for run_gt: materials around the agent

    def __init__(self, cfg: GridCfg = GridCfg(), device="cpu"):
        c = cfg
        assert c.H < c.W and c.res % 8 == 0 and c.T >= 2
        assert c.n_slow + 1 <= (c.W - c.H) * c.W, "HUD slots and the hand must fit the HUD rows"
        self.cfg, self.dev = c, device
        gen = torch.Generator().manual_seed(c.seed)
        cells = torch.randperm(c.H * c.W, generator=gen)[:c.n_twinkle]
        self.tw_y, self.tw_x = (cells // c.W).to(device), (cells % c.W).to(device)
        hue = torch.rand(c.n_slow, 3, generator=gen) * 0.7 + 0.3
        self.hud_col = (hue / hue.max(1, keepdim=True).values).to(device)
        self.fixed_map = None if c.random_map else self._maps(1, gen).to(device)
        self.mat_col, self.moves = MAT_COL.to(device), MOVES.to(device)
        # within-tile masks of the facing half: up, down, left, right
        r = torch.arange(c.res, device=device) % c.tile < c.tile // 2
        yy, xx = r[:, None].expand(c.res, c.res), r[None, :].expand(c.res, c.res)
        self.face_mask = torch.stack([yy, ~yy, xx, ~xx])  # (4, res, res)
        self.dims = {"slow": c.n_slow + 2, "fast": 4, "hard": N_MAT,
                     "noise": 2 + c.n_twinkle}

    def _maps(self, n, gen=None) -> torch.Tensor:
        """(n, H, W) materials; the twinkle cells are always floor."""
        c, dev = self.cfg, "cpu" if gen is not None else self.dev
        u = torch.rand(n, c.H, c.W, generator=gen, device=dev)
        kind = torch.randint(1, N_MAT, (n, c.H, c.W), generator=gen, device=dev)
        m = torch.where(u < c.density, kind, torch.zeros_like(kind))
        m[:, self.tw_y.to(dev), self.tw_x.to(dev)] = EMPTY
        return m

    def _rand_free(self, free, k=None):
        """Uniform free cell per clip (k: per clip and frame). free: (n, H, W) bool -> (..., 2) xy."""
        n, H, W = free.shape
        shape = (n, H * W) if k is None else (n, k, H * W)
        score = torch.rand(shape, device=self.dev)
        fr = free.flatten(1) if k is None else free.flatten(1)[:, None]
        i = (score * fr).argmax(-1)
        return torch.stack([i % W, i // W], -1)

    @torch.no_grad()
    def sample(self, n: int, render: bool = True) -> Clip:
        c, dev, T = self.cfg, self.dev, self.cfg.T
        mat = self.fixed_map.expand(n, -1, -1) if self.fixed_map is not None else self._maps(n)
        pad = F.pad(mat, (1, 1, 1, 1), value=STONE)  # border = stone
        ni = torch.arange(n, device=dev)

        def at(p):  # material at xy p: (n, 2); everything off the map is stone
            return pad[ni, p[:, 1].clamp(-1, c.H) + 1, p[:, 0].clamp(-1, c.W) + 1]

        free = mat == EMPTY
        p, f = self._rand_free(free), torch.randint(4, (n,), device=dev)
        a = torch.randint(ACT_DIM, (n, T - 1), device=dev)
        hud = [torch.randint(c.levels, (n, c.n_slow), device=dev)]
        P, Fc = [p], [f]
        for t in range(T - 1):
            mv = a[:, t] > 0
            f = torch.where(mv, a[:, t] - 1, f)
            q = p + self.moves[a[:, t]]
            p = torch.where((at(q) == EMPTY)[:, None], q, p)
            P.append(p)
            Fc.append(f)
            step = (torch.rand(n, c.n_slow, device=dev) < c.hud_p) * (
                2 * torch.randint(2, (n, c.n_slow), device=dev) - 1)
            hud.append((hud[-1] + step).clamp(0, c.levels - 1))
        P, Fc, hud = torch.stack(P, 1), torch.stack(Fc, 1), torch.stack(hud, 1)  # (n, T, ...)
        dirs = DIRS.to(dev)
        hand = torch.stack([at(P[:, t] + dirs[Fc[:, t]]) for t in range(T)], 1)  # (n, T)
        # materials at distance 1 and 2 along the four directions: all the hand can depend on
        local = torch.stack([at(P[:, t] + k * dirs[d]) for t in range(T) for k in (1, 2)
                             for d in range(4)], 1).view(n, T, 8)
        mob = self._rand_free(free, T)  # (n, T, 2)
        tw = torch.rand(n, T, c.n_twinkle, device=dev)

        def xy(v):  # integer xy -> [-1, 1]
            return torch.stack([v[..., 0] / (c.W - 1), v[..., 1] / (c.H - 1)], -1) * 2 - 1

        f_ = {"slow": torch.cat([hud / (c.levels - 1) * 2 - 1, xy(P)], -1),
              "fast": F.one_hot(Fc, 4).float(),
              "hard": F.one_hot(hand, N_MAT).float(),
              "noise": torch.cat([xy(mob), tw * 2 - 1], -1),
              "local": F.one_hot(local, N_MAT).float().flatten(2),
              # the whole map, constant in a clip: not a group, for latent_determined_by()
              "map": F.one_hot(mat, N_MAT).float().flatten(1)[:, None].expand(-1, T, -1)}
        a1 = F.one_hot(a, ACT_DIM).float()
        if not render:
            return Clip(None, a1, f_)
        return Clip(self._render(mat, P, Fc, hand, hud, mob, tw), a1, f_)

    def _render(self, mat, P, Fc, hand, hud, mob, tw):
        c, dev = self.cfg, self.dev
        n, T = P.shape[:2]
        rows = c.W  # frame is W x W tiles
        g = torch.zeros(n, T, rows, c.W, 3, device=dev)
        g[:, :, :c.H] = self.mat_col[mat][:, None]
        g[:, :, self.tw_y, self.tw_x] = (self.mat_col[EMPTY] + tw[..., None] * TWINKLE.to(dev)
                                         ).clamp(max=1)
        k = torch.arange(c.n_slow + 1, device=dev)
        hy, hx = c.H + k // c.W, k % c.W
        g[:, :, hy[:-1], hx[:-1]] = hud[..., None] / (c.levels - 1) * self.hud_col
        g[:, :, hy[-1], hx[-1]] = self.mat_col[hand]
        flat = g.view(n, T, rows * c.W, 3)
        ni, ti = torch.arange(n, device=dev)[:, None], torch.arange(T, device=dev)[None]
        flat[ni, ti, mob[..., 1] * c.W + mob[..., 0]] = MOB.to(dev)
        agent = torch.zeros(n, T, rows * c.W, device=dev, dtype=torch.bool)
        agent[ni, ti, P[..., 1] * c.W + P[..., 0]] = True
        up = c.tile
        x = g.repeat_interleave(up, 2).repeat_interleave(up, 3)  # (n, T, res, res, 3)
        am = agent.view(n, T, rows, c.W).repeat_interleave(up, 2).repeat_interleave(up, 3)
        half = self.face_mask[Fc]  # (n, T, res, res)
        x = torch.where((am & half)[..., None], FACE.to(dev),
                        torch.where(am[..., None], AGENT.to(dev), x))
        return (x * 2 - 1).permute(0, 1, 4, 2, 3).contiguous()  # (n, T, 3, res, res) in [-1, 1]

    def parts(self) -> dict:
        """Named parts of the groups: part -> (group, columns within the group)."""
        n = self.cfg.n_slow
        return {"inventory": ("slow", slice(0, n)), "position": ("slow", slice(n, n + 2)),
                "facing": ("fast", slice(0, 4)), "hand": ("hard", slice(0, N_MAT)),
                "mob": ("noise", slice(0, 2)), "twinkle": ("noise", slice(2, 2 + self.cfg.n_twinkle))}

    def factors(self, clip: Clip, t: int = -1) -> torch.Tensor:
        """All groups at frame t, concatenated in GROUPS order: (n, sum of dims)."""
        return torch.cat([clip.f[g][:, t] for g in GROUPS], -1)

    def group_cols(self) -> dict:
        out, i = {}, 0
        for g in GROUPS:
            out[g] = slice(i, i + self.dims[g])
            i += self.dims[g]
        return out
