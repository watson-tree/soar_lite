"""The default restart rule: sample-based expected improvement (first order).

`EIRestartRule` decides continue-vs-restart by comparing the closed-form
expected improvement of one more step against one restart, both fitted from
the evaluated-sample statistics the engine forwards. It implements the engine's
`RestartRule` contract and owns only that decision; no surrogate model is fit.
"""

import math

from soar_lite.engine import RestartRule


# --- restart rule: EI_continue vs EI_restart ---
def normal_cdf(z: float) -> float:
    """Return the standard normal CDF Φ(z) = (1 + erf(z/√2))/2."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_pdf(z: float) -> float:
    """Return the standard normal PDF φ(z) = exp(-z²/2)/√(2π)."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def gaussian_ei(sigma: float, incumbent: float, mu: float) -> float:
    """Return the expected improvement over the incumbent `B` for X ~ N(mu, sigma²).

    EI = sigma·(z·Φ(z) + φ(z)) with z = (B - mu)/sigma; sigma = 0 degenerates
    exactly to max(B - mu, 0). Shared closed form: EI_continue (mu = f_k + mu_p)
    and EI_restart (mu = mu_r) both route through here.
    """
    if sigma == 0.0:
        return max(incumbent - mu, 0.0)
    z = (incumbent - mu) / sigma
    return sigma * (z * normal_cdf(z) + normal_pdf(z))


def ei_continue(sigma_p: float, incumbent: float, f_k: float, mu_p: float) -> float:
    """Return the expected improvement of one more step from f_k.

    L_next ~ N(f_k + mu_p, sigma_p²), so EI_continue = gaussian_ei(sigma_p, B,
    f_k + mu_p).
    """
    return gaussian_ei(sigma_p, incumbent, f_k + mu_p)


def ei_restart(incumbent: float, group_mins: list[float]) -> float:
    """Return the expected improvement of one restart.

    Fits N(mu_r, sigma_r²) over the probe group-min history {r_i} (mean +
    population std, non-finite skipped): EI_restart = gaussian_ei(sigma_r, B, mu_r).
    M = 0 → 0.0 (no evidence); M = 1 → sigma_r = 0 → degenerate max(B - mu_r, 0).

    Skipping isfinite, not just NaN, is load-bearing: one +inf sample makes mu_r
    inf and variance inf - inf = NaN, so sigma_r = NaN and this returns NaN -
    and `NaN > EI_continue` is False, which permanently silences every restart.
    """
    r = [v for v in group_mins if math.isfinite(v)]
    if not r:
        return 0.0
    mu_r = sum(r) / len(r)
    variance = sum((v - mu_r) ** 2 for v in r) / len(r)
    return gaussian_ei(math.sqrt(variance), incumbent, mu_r)



class EIRestartRule(RestartRule):
    """Sample-based restarts: the paths estimate their own restart payoff.

    The rule fits no surrogate - the K i.i.d. sampled starts are the sample. The
    lightweight restart rule of a SOAR-style adaptive-restart engine
    [Mathesen et al. 2021]: it never fits a model, so it stays cheap in high
    dimension, and it samples rather than learns the restart payoff. Restart
    fires only when a fresh draw's expected improvement exceeds continuing the
    path's (``ei_restart`` vs ``ei_continue``) - both closed forms off the same
    evidence. The i.i.d. assumption is what makes this exact: every path starts
    as an i.i.d. draw from the injected sampler, so each recorded restart payoff
    is a real sample of "what a fresh start buys", and the shared pool below
    estimates that single global (mu_r, sigma_r) with minimal variance.

    The verdict compares two sample estimates of comparable form - the path's
    own one-step spread against the pool's - and what makes the comparison
    readable is that the two are worth about the same number of samples: both
    relative sds are ~1/sqrt(2(n - 1)), so a shorter continue side reads the last
    few steps' own jitter instead of the path's, an understated sigma shrinks
    EI_continue, and the restart tail wins verdicts the evidence does not
    support. That bias never shows up as an error, only as a worse incumbent, so
    the rule **derives** the two counts instead of exposing a knob: the
    continue side is read over `stage_window()` = the pool's size M, which is the
    most samples a verdict can have in as many rounds as the restart side
    already has, and a caller cannot ask for a window its own pool cannot back.

    Equal weights are the only form that uses a whole stage: a windowed estimate
    that fades its older samples is worth fewer effective samples than it holds
    (its Kish size tends to (2 - alpha)/alpha, never above the number of folds),
    so on a stage that ends in a verdict discounting can only throw evidence
    away. Recency would only pay if the drift within one stage were large, and a
    stage is as long as the pool is large - short. Nothing is lost over a longer
    horizon either: a restart drops the path's stage (the next Δ opens a new one
    against the pool as it stands), and the verdict is asked again every
    `stage_window()` rounds, so the loop re-reads the path as regularly as it
    collects evidence.

    `group_mins` is **one pool for every path**, not a copy per path: a restart
    payoff is a draw from the sampler+incumbent distribution, the same quantity
    for all paths, so pooling every event estimates that single global
    (mu_r, sigma_r) with the smallest variance - a per-path subset would
    estimate the same global value from a shorter sample. NaN/inf Δ are kept in
    a path's stage but never read (guard), and still count towards the window.
    The incumbent B is a should_restart argument, not a field (the engine owns
    B).
    """

    def __init__(self) -> None:
        """Start with an empty payoff pool and no per-path evidence."""
        self.group_mins: list[float] = []  # one shared restart-payoff pool {r_i}
        self.pool_size = 0  # usable pool samples: the M the window is matched to
        self.deltas: dict[int, list[float]] = {}  # per path: Δ since its last verdict

    def stage_window(self) -> int:
        """Return how many Δ a path is read over: the pool's size.

        Read fresh at each verdict rather than fixed per stage, because the pool
        only grows - samples are appended, never dropped or aged out - so `M` at
        the verdict is an upper bound on the stage that produced it, and matched
        counts is the property the verdict needs. Nothing pads it: the read is a
        plain mean and population sd over the stage, so extra folds are extra
        evidence and no fold buys more than its weight.

        The only floor is 1, and it is not a prior: a window of zero samples has
        no centre to read (`ei_continue` returns 0.0 on an empty stage, and
        `rounds_until_verdict` would go negative), which is reachable only
        before any finite restart payoff exists. At M = 1 the read is its own
        degenerate branch - a stage of one fold has no spread, and a pool of one
        gives `sigma_r = 0`, so no restart can fire until a second payoff lands.

        This method is the seam a rule variant moves (a capped window, a forced
        cadence for an experiment); the rule itself takes no knob, so its own
        cadence can only follow the evidence.
        """
        return max(1, self.pool_size)

    def observe_continue(self, i: int, delta: float) -> None:
        """Append path i's one-step value change Δ to its current stage.

        A non-finite Δ is stored, not dropped: a stalled path folds one every
        round and its verdict has to stay due on the same clock as its moving
        cohort (the engine reports the stall through the value, not by skipping
        the call), so the guard belongs at the read, where one +inf sample would
        otherwise make the mean and spread NaN - and `NaN > EI_continue` is
        False, which silently disables that path's restarts.
        """
        self.deltas.setdefault(i, []).append(delta)

    def observe_restart(self, r: float) -> None:
        """Append one restart event's payoff r to the shared payoff pool.

        Non-finite payoffs stay in the list (ei_restart filters them, and a
        single +inf must not corrupt the fit) but do not count towards M: the
        window is matched to the samples sigma_r is actually fitted on.
        """
        self.group_mins.append(r)
        if math.isfinite(r):
            self.pool_size += 1

    def stage_deltas(self, i: int) -> list[float]:
        """Return path i's finite Δ values from the stage it is in now."""
        return [d for d in self.deltas.get(i, ()) if math.isfinite(d)]

    def ei_continue(self, i: int, f_k: float, incumbent: float) -> float:
        """Return the expected improvement of path i continuing one more step.

        The stage's plain mean and population std - the same ÷n the restart side
        fits its pool with, so the read is worth exactly the samples it holds.
        An empty stage has neither centre nor spread and returns 0.0: the
        degenerate "continuing buys nothing" read (the same one a zero-spread
        stage gives), which leaves the verdict resting on the restart side's own
        evidence. One fold is the other degenerate corner (sigma = 0), where the
        read is exactly max(B - (f_k + mu), 0).
        """
        d = self.stage_deltas(i)
        if not d:
            return 0.0
        mu = sum(d) / len(d)
        sd = math.sqrt(sum((v - mu) ** 2 for v in d) / len(d))
        return ei_continue(sd, incumbent, f_k, mu)

    def ei_restart(self, incumbent: float) -> float:
        """Return the expected improvement of any path restarting (global)."""
        return ei_restart(incumbent, self.group_mins)

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:
        """Return True iff path i's verdict is due and restart beats continuing.

        Self-gated: due once the path has folded a full window of Δ since its
        last verdict, and the stage closes here whichever way the verdict goes -
        the gate is what opens a verdict, so a stage left open would be re-asked
        unchanged every round. A full stage also closes while there is no
        incumbent (B = inf): the verdict has nothing to compare against yet, and
        the engine owns B.
        """
        if len(self.deltas.get(i, ())) < self.stage_window():
            return False
        restart = math.isfinite(incumbent) and (
            self.ei_restart(incumbent) > self.ei_continue(i, f_k, incumbent)
        )
        self.deltas[i] = []  # the stage's one read is taken; the next Δ opens one
        return restart

    def rounds_until_verdict(self) -> int:
        """Report the rounds until the earliest path's stage fills up.

        `window - folds` per path: exact while the pool stands still and an
        underestimate once it grows, since `window` only lengthens. That
        direction is the safe one - the engine stops when this says a vote is out
        of reach, and an overestimated cadence there would be a stop for a vote
        that was still due. A path that has just restarted counts from 0, i.e. as
        a full window: it folds its first Δ at the end of the current round. 0
        with no path evidence at all - no cadence to bound, so the engine keeps
        its ordinary rounds, which is only reachable before the first fold.
        """
        window = self.stage_window()
        return min((window - len(d) for d in self.deltas.values()), default=0)

    def restart(self, i: int) -> None:
        """Drop path i's stage after a restart (the shared payoff pool is kept)."""
        self.deltas[i] = []
