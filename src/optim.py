import torch


def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _scale_lr(lr: float, rows: int, cols: int, adjust_rms_norm: bool) -> float:
    if adjust_rms_norm:
        return 0.2 * lr * max(rows, cols) ** 0.5
    return lr * max(1.0, rows / cols) ** 0.5


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_rms_norm: bool = True,
        adamw_weight_decay: float = 0.01,
        adamw_betas: tuple = (0.9, 0.999),
    ):
        param_list = list(params)
        # Register all params in the base class so param_groups[0]["lr"] is
        # accessible for LR logging in module.py.
        super().__init__(param_list, defaults=dict(lr=lr))

        self._muon_params = [p for p in param_list if p.ndim >= 2]
        self._adamw_params = [p for p in param_list if p.ndim < 2]
        self._momentum = momentum
        self._nesterov = nesterov
        self._ns_steps = ns_steps
        self._adjust_rms_norm = adjust_rms_norm

        self._adamw = (
            torch.optim.AdamW(
                self._adamw_params,
                lr=lr,
                weight_decay=adamw_weight_decay,
                betas=adamw_betas,
            )
            if self._adamw_params
            else None
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = self.param_groups[0]["lr"]

        for p in self._muon_params:
            if p.grad is None:
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - self._momentum)

            # Nesterov: interpolate grad toward the updated momentum buffer
            g = p.grad.lerp_(buf, self._momentum) if self._nesterov else buf.clone()

            # Reshape to 2-D: handles both linear (2-D) and conv (4-D) weights
            orig_shape = g.shape
            g_2d = g.reshape(g.size(0), -1)

            update = _newton_schulz_orthogonalize(g_2d, steps=self._ns_steps)

            rows, cols = update.shape
            scaled_lr = _scale_lr(lr, rows, cols, self._adjust_rms_norm)
            p.add_(update.reshape(orig_shape), alpha=-scaled_lr)

        if self._adamw is not None:
            self._adamw.step()

        return loss

    # ------------------------------------------------------------------ #
    # Checkpoint support                                                   #
    # ------------------------------------------------------------------ #

    def state_dict(self):
        base = super().state_dict()
        base["adamw"] = self._adamw.state_dict() if self._adamw is not None else None
        return base

    def load_state_dict(self, state_dict: dict):
        adamw_sd = state_dict.pop("adamw", None)
        super().load_state_dict(state_dict)
        if self._adamw is not None and adamw_sd is not None:
            self._adamw.load_state_dict(adamw_sd)
