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
all T-1 of them. The frame is W x W tiles: H map rows, then W - H HUD rows.

Knobs for the mass experiments (notebooks/04_mass_law.ipynb), all off by default:
    hand_tiles     the hand is shown in this many HUD tiles (its pixel mass grows with it)
    compass_tiles  the facing is also shown in this many HUD tiles (a fixed-location copy of the marker)
    hand_contrast  the hand tiles' colour contrast against the average hand colour (1: the material colours; 0.5: a quarter of the
                   mass), for masses below one tile
    agent_shape    "marker" (default): a white tile with a yellow half toward the facing; "arrow": the agent is a white arrow
                   (4 rotations) on the terrain, so the facing is the agent's whole silhouette (as in Craftax, where the sprite changes with
                   the facing)
    mob_mode       "jump" (default): the mob jumps to a random free cell every frame (noise, group noise); "chase": n_mobs mobs persist and
                   move one free step per frame, toward the agent with probability mob_chase (else a random step or none), blocked by
                   materials, the agent and the border; the agent cannot walk into a mob. Predictable given the agent and the action, so
                   their position is in the group fast (part mob)
    hand_p         per clip, the hand tiles are visible with this probability (frequency against size); hidden tiles show
                   the average hand colour, so being hidden adds no variance of its own (as an absent block in Craftax
                   shows ordinary terrain, not a blank)
`f["tile_class"]` holds the ground-truth class of every tile (TILE_CLASSES) for the class-balanced loss,
and `component_masses()` measures each component's share of the pixel variance.
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
HAND_SHARE = torch.tensor([0.38, 0.20, 0.32, 0.10])  # how often the hand shows empty, wood, stone, water (density 0.35)
COMPASS_COL = torch.tensor([[0.95, 0.30, 0.30], [0.30, 0.95, 0.30], [0.30, 0.30, 0.95], [0.95, 0.95, 0.30]])
# ground-truth tile classes (f["tile_class"]): what a tile shows
TILE_CLASSES = (["map:" + m for m in ("floor", "wood", "stone", "water")] + ["twinkle", "mob"]
                + ["agent:" + d for d in ("up", "down", "left", "right")]
                + [f"inventory:{v}" for v in range(10)] + ["hand:" + m for m in ("floor", "wood", "stone", "water")]
                + ["compass:" + d for d in ("up", "down", "left", "right")] + ["hud:empty"])
TC = {name: i for i, name in enumerate(TILE_CLASSES)}


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
    hand_tiles: int = 1  # HUD tiles that show the hand
    compass_tiles: int = 0  # HUD tiles that show the facing (0: only the marker on the agent)
    hand_p: float = 1.0  # probability per clip that the hand tiles are visible
    hand_contrast: float = 1.0  # contrast of the hand tiles against the average hand colour (mass scales with its square)
    agent_shape: str = "marker"  # "marker" | "arrow"
    mob_mode: str = "jump"  # "jump" | "chase"
    n_mobs: int = 1  # chase mode only
    mob_chase: float = 0.7  # chase mode: probability of a step toward the agent

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
        assert c.n_slow + c.hand_tiles + c.compass_tiles <= (c.W - c.H) * c.W, \
            "the inventory, hand and compass tiles must fit the HUD rows"
        assert c.levels <= 10, "tile classes cover inventory levels 0..9"
        assert c.agent_shape in ("marker", "arrow") and c.mob_mode in ("jump", "chase")
        assert c.mob_mode == "chase" or c.n_mobs == 1, "n_mobs is for the chase mode"
        self.cfg, self.dev = c, device
        self.keep_state = False  # variants(): keep the sampled state of the last sample() call in self._state
        self.hide = set()  # components left out of the frame (component_masses): hand, compass, facing, agent, mob, twinkle, inventory
        gen = torch.Generator().manual_seed(c.seed)
        cells = torch.randperm(c.H * c.W, generator=gen)[:c.n_twinkle]
        self.tw_y, self.tw_x = (cells // c.W).to(device), (cells % c.W).to(device)
        hue = torch.rand(c.n_slow, 3, generator=gen) * 0.7 + 0.3
        self.hud_col = (hue / hue.max(1, keepdim=True).values).to(device)
        self.fixed_map = None if c.random_map else self._maps(1, gen).to(device)
        self.mat_col, self.moves = MAT_COL.to(device), MOVES.to(device)
        self.compass_col = COMPASS_COL.to(device)
        self.hand_mean = (HAND_SHARE[:, None] * MAT_COL).sum(0).to(device)  # what a hidden hand tile shows
        # within-tile masks of the facing half: up, down, left, right
        r = torch.arange(c.res, device=device) % c.tile < c.tile // 2
        yy, xx = r[:, None].expand(c.res, c.res), r[None, :].expand(c.res, c.res)
        self.face_mask = torch.stack([yy, ~yy, xx, ~xx])  # (4, res, res)
        # within-tile silhouettes of the arrow agent: up, down, left, right
        t = c.tile
        ramp = (torch.arange(t)[:, None] + 1) / 2  # half width of the arrow in row i
        up = (torch.arange(t)[None, :] + 0.5 - t / 2).abs() <= ramp
        arrows = torch.stack([up, up.flip(0), up.T, up.T.flip(1)])  # apex up, down, left, right
        self.arrow_mask = arrows.repeat(1, c.W, c.W).to(device)  # (4, res, res)
        chase = c.mob_mode == "chase"
        self.dims = {"slow": c.n_slow + 2, "fast": 4 + (2 * c.n_mobs if chase else 0), "hard": N_MAT,
                     "noise": c.n_twinkle + (0 if chase else 2)}

    def _maps(self, n, gen=None) -> torch.Tensor:
        """(n, H, W) materials; the twinkle cells are always floor."""
        c, dev = self.cfg, "cpu" if gen is not None else self.dev
        u = torch.rand(n, c.H, c.W, generator=gen, device=dev)
        kind = torch.randint(1, N_MAT, (n, c.H, c.W), generator=gen, device=dev)
        m = torch.where(u < c.density, kind, torch.zeros_like(kind))
        m[:, self.tw_y.to(dev), self.tw_x.to(dev)] = EMPTY
        return m

    def _chase(self, m, agent, pad, ni):
        """One step of the mobs m (n, k, 2) given the agent's new cell: toward it along the longer axis with probability
        mob_chase, else a random move (or none); a blocked step (material, border, the agent) leaves the mob where it is."""
        c, dev = self.cfg, self.dev
        n, k = m.shape[:2]
        d = agent[:, None] - m
        along_x = d[..., 0].abs() >= d[..., 1].abs()
        zero = torch.zeros_like(d[..., 0])
        toward = torch.where(along_x[..., None], torch.stack([d[..., 0].sign(), zero], -1),
                             torch.stack([zero, d[..., 1].sign()], -1))
        rnd = self.moves[torch.randint(ACT_DIM, (n, k), device=dev)]
        q = m + torch.where((torch.rand(n, k, device=dev) < c.mob_chase)[..., None], toward, rnd)
        free = pad[ni[:, None], q[..., 1].clamp(-1, c.H) + 1, q[..., 0].clamp(-1, c.W) + 1] == EMPTY
        free = free & ~(q == agent[:, None]).all(-1)
        return torch.where(free[..., None], q, m)

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
        chase = c.mob_mode == "chase"
        if chase:  # mobs start on free cells other than the agent's
            free0 = free.clone()
            free0[ni, p[:, 1], p[:, 0]] = False
            mob_t = [self._rand_free(free0, c.n_mobs)]  # (n, k, 2)
        a = torch.randint(ACT_DIM, (n, T - 1), device=dev)
        hud = [torch.randint(c.levels, (n, c.n_slow), device=dev)]
        P, Fc = [p], [f]
        for t in range(T - 1):
            mv = a[:, t] > 0
            f = torch.where(mv, a[:, t] - 1, f)
            q = p + self.moves[a[:, t]]
            ok = at(q) == EMPTY
            if chase:
                ok = ok & ~(mob_t[-1] == q[:, None]).all(-1).any(-1)  # a mob blocks the agent
            p = torch.where(ok[:, None], q, p)
            P.append(p)
            if chase:
                mob_t.append(self._chase(mob_t[-1], p, pad, ni))
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
        mob = torch.stack(mob_t, 1) if chase else self._rand_free(free, T)[:, :, None]  # (n, T, k, 2)
        tw = torch.rand(n, T, c.n_twinkle, device=dev)
        # the hand tiles' visibility per clip (drawn only when hand_p < 1: default clips are unchanged)
        vis = (torch.rand(n, device=dev) < c.hand_p if c.hand_p < 1
               else torch.ones(n, dtype=torch.bool, device=dev))

        if self.keep_state:
            self._state = dict(mat=mat, P=P, Fc=Fc, hand=hand, hud=hud, mob=mob, tw=tw, vis=vis)

        def xy(v):  # integer xy -> [-1, 1]
            return torch.stack([v[..., 0] / (c.W - 1), v[..., 1] / (c.H - 1)], -1) * 2 - 1

        mob_xy = xy(mob).flatten(-2)  # (n, T, 2k)
        face = F.one_hot(Fc, 4).float()
        f_ = {"slow": torch.cat([hud / (c.levels - 1) * 2 - 1, xy(P)], -1),
              "fast": torch.cat([face, mob_xy], -1) if chase else face,
              "hard": F.one_hot(hand, N_MAT).float(),
              "noise": (tw * 2 - 1) if chase else torch.cat([mob_xy, tw * 2 - 1], -1),
              "local": F.one_hot(local, N_MAT).float().flatten(2),
              # the whole map, constant in a clip: not a group, for latent_determined_by()
              "map": F.one_hot(mat, N_MAT).float().flatten(1)[:, None].expand(-1, T, -1),
              "hand_visible": vis.float()[:, None, None].expand(-1, T, 1)}
        a1 = F.one_hot(a, ACT_DIM).float()
        if not render:
            return Clip(None, a1, f_)
        x, cls = self._render(mat, P, Fc, hand, hud, mob, tw, vis)
        f_["tile_class"] = cls
        return Clip(x, a1, f_)

    def _render(self, mat, P, Fc, hand, hud, mob, tw, vis):
        """The frames (n, T, 3, res, res) in [-1, 1] and the class of every tile (n, T, W * W)."""
        c, dev, hide = self.cfg, self.dev, self.hide
        n, T = P.shape[:2]
        rows = c.W  # frame is W x W tiles
        g = torch.zeros(n, T, rows, c.W, 3, device=dev)
        cls = torch.full((n, T, rows, c.W), TC["hud:empty"], dtype=torch.long, device=dev)
        g[:, :, :c.H] = self.mat_col[mat][:, None]
        cls[:, :, :c.H] = mat[:, None]  # map:floor .. map:water are 0..3
        if "twinkle" not in hide:
            g[:, :, self.tw_y, self.tw_x] = (self.mat_col[EMPTY] + tw[..., None] * TWINKLE.to(dev)).clamp(max=1)
            cls[:, :, self.tw_y, self.tw_x] = TC["twinkle"]
        n_hud = c.n_slow + c.hand_tiles + c.compass_tiles
        k = torch.arange(n_hud, device=dev)
        hy, hx = c.H + k // c.W, k % c.W
        s0, s1 = c.n_slow, c.n_slow + c.hand_tiles
        if "inventory" not in hide:
            g[:, :, hy[:s0], hx[:s0]] = hud[..., None] / (c.levels - 1) * self.hud_col
            cls[:, :, hy[:s0], hx[:s0]] = TC["inventory:0"] + hud
        if c.hand_tiles and "hand" not in hide:
            shown = vis[:, None, None]  # (n, 1, 1)
            col = self.mat_col[hand][:, :, None]  # (n, T, 1, 3)
            col = self.hand_mean + c.hand_contrast * (col - self.hand_mean)
            g[:, :, hy[s0:s1], hx[s0:s1]] = torch.where(shown[..., None], col, self.hand_mean.expand_as(col))
            cls[:, :, hy[s0:s1], hx[s0:s1]] = torch.where(shown, TC["hand:floor"] + hand[..., None],
                                                          torch.full_like(hand[..., None], TC["hud:empty"]))
        if c.compass_tiles and "compass" not in hide:
            g[:, :, hy[s1:], hx[s1:]] = self.compass_col[Fc][:, :, None]
            cls[:, :, hy[s1:], hx[s1:]] = TC["compass:up"] + Fc[..., None]
        flat = g.view(n, T, rows * c.W, 3)
        cflat = cls.view(n, T, rows * c.W)
        ni, ti = torch.arange(n, device=dev)[:, None], torch.arange(T, device=dev)[None]
        if "mob" not in hide:
            idx = mob[..., 1] * c.W + mob[..., 0]  # (n, T, k)
            flat[ni[..., None], ti[..., None], idx] = MOB.to(dev)
            cflat[ni[..., None], ti[..., None], idx] = TC["mob"]
        agent = torch.zeros(n, T, rows * c.W, device=dev, dtype=torch.bool)
        if "agent" not in hide:
            agent[ni, ti, P[..., 1] * c.W + P[..., 0]] = True
            cflat[ni, ti, P[..., 1] * c.W + P[..., 0]] = TC["agent:up"] + Fc
        up = c.tile
        x = g.repeat_interleave(up, 2).repeat_interleave(up, 3)  # (n, T, res, res, 3)
        am = agent.view(n, T, rows, c.W).repeat_interleave(up, 2).repeat_interleave(up, 3)
        if c.agent_shape == "arrow":  # the silhouette is the facing; hiding the facing fixes it to "up"
            sil = self.arrow_mask[torch.zeros_like(Fc) if "facing" in hide else Fc]
            x = torch.where((am & sil)[..., None], AGENT.to(dev), x)
        else:
            half = self.face_mask[Fc]  # (n, T, res, res)
            face = AGENT if "facing" in hide else FACE
            x = torch.where((am & half)[..., None], face.to(dev),
                            torch.where(am[..., None], AGENT.to(dev), x))
        return (x * 2 - 1).permute(0, 1, 4, 2, 3).contiguous(), cflat

    @torch.no_grad()
    def variants(self, n: int) -> dict:
        """Counterfactual last frames: the same sampled state with one thing changed. name -> (x (n, 3, res, res), valid (n,) bool),
        for "base", "facing" (another direction, everything else as is), "position" (the agent one free cell away), "hand" (another
        material in the hand tiles) and "twinkle" (fresh twinkle brightness). Used to ask an encoder how much its latent moves when
        only that thing changes, without a probe."""
        c, dev = self.cfg, self.dev
        self.keep_state = True
        try:
            self.sample(n)
        finally:
            self.keep_state = False
        st = {k: (v[:, -1:] if k in ("P", "Fc", "hand", "hud", "mob", "tw") else v) for k, v in self._state.items()}
        ones = torch.ones(n, dtype=torch.bool, device=dev)

        def draw(**over):
            a = {**st, **over}
            return self._render(a["mat"], a["P"], a["Fc"], a["hand"], a["hud"], a["mob"], a["tw"], a["vis"])[0][:, 0]

        out = {"base": (draw(), ones)}
        out["facing"] = (draw(Fc=(st["Fc"] + torch.randint(1, 4, (n, 1), device=dev)) % 4), ones)
        out["hand"] = (draw(hand=(st["hand"] + torch.randint(1, N_MAT, (n, 1), device=dev)) % N_MAT), ones)
        out["twinkle"] = (draw(tw=torch.rand_like(st["tw"])), ones)
        q = st["P"] + DIRS.to(dev)[torch.randint(4, (n,), device=dev)][:, None]  # one cell in a random direction
        inside = (q[..., 0] >= 0) & (q[..., 0] < c.W) & (q[..., 1] >= 0) & (q[..., 1] < c.H)
        qc = torch.stack([q[..., 0].clamp(0, c.W - 1), q[..., 1].clamp(0, c.H - 1)], -1)
        free = self._state["mat"][torch.arange(n, device=dev), qc[:, 0, 1], qc[:, 0, 0]] == EMPTY
        on_mob = (st["mob"] == qc[:, :, None]).all(-1).any(-1)[:, 0]
        out["position"] = (draw(P=qc), inside[:, 0] & free & ~on_mob)
        return out

    def parts(self) -> dict:
        """Named parts of the groups: part -> (group, columns within the group)."""
        c = self.cfg
        n = c.n_slow
        mob, tw = (("fast", slice(4, 4 + 2 * c.n_mobs)), ("noise", slice(0, c.n_twinkle))) if c.mob_mode == "chase" \
            else (("noise", slice(0, 2)), ("noise", slice(2, 2 + c.n_twinkle)))
        return {"inventory": ("slow", slice(0, n)), "position": ("slow", slice(n, n + 2)),
                "facing": ("fast", slice(0, 4)), "hand": ("hard", slice(0, N_MAT)), "mob": mob, "twinkle": tw}

    def factors(self, clip: Clip, t: int = -1) -> torch.Tensor:
        """All groups at frame t, concatenated in GROUPS order: (n, sum of dims)."""
        return torch.cat([clip.f[g][:, t] for g in GROUPS], -1)

    def group_cols(self) -> dict:
        out, i = {}, 0
        for g in GROUPS:
            out[g] = slice(i, i + self.dims[g])
            i += self.dims[g]
        return out


@torch.no_grad()
def component_masses(cfg: GridCfg, n: int = 2048, seed: int = 0) -> dict:
    """Share of the pixel variance (per-pixel variance over clips, mean over pixels) that each component explains: the
    variance over clips of the component's contribution (the frames minus the same frames with the component left out),
    over the variance of the frames. A component that looks the same in every clip contributes nothing, however bright."""
    def draw(hide=()):
        env = GridEnv(cfg, "cpu")
        env.hide = set(hide)
        torch.manual_seed(seed)
        return env.sample(n).x

    x = draw()
    var = x.var(0).mean().item()
    out = {}
    for name, hide in (("hand", ("hand",)), ("compass", ("compass",)), ("facing marker", ("facing",)),
                       ("agent (with marker)", ("agent",)), ("mob", ("mob",)), ("twinkle", ("twinkle",)),
                       ("inventory (all slots)", ("inventory",))):
        if name == "compass" and not cfg.compass_tiles:
            continue
        out[name] = (x - draw(hide)).var(0).mean().item() / var
    out["inventory (one slot)"] = out["inventory (all slots)"] / cfg.n_slow
    return out
