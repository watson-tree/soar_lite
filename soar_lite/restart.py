"""The default restart rule: sample-based expected improvement (first order).

`EIRestartRule` decides continue-vs-restart by comparing the closed-form
expected improvement of one more step against one restart, both fitted from
the evaluated-sample statistics the engine forwards. It implements the engine's
`RestartRule` contract and owns only that decision; no surrogate model is fit.
"""

import math
import statistics

from soar_lite.engine import RestartRule

# Φ⁻¹(3/4): normal-consistency constant that turns a MAD into a scale estimate.
PHI14 = 0.6744897501960817
# (4/3)^(1/5) ≈ 1.06: the normal-reference constant of Scott's rule.
SCOTT = (4.0 / 3.0) ** 0.2


# --- restart rule: EI_continue vs EI_restart ---
class EIRestartRule(RestartRule):
    """Sample-based restarts: the paths estimate their own restart payoff.

    The rule fits no surrogate - the n_paths i.i.d. sampled starts are the
    sample. The lightweight restart rule of a SOAR-style adaptive-restart engine
    [Mathesen et al. 2021]: it never fits a model, so it stays cheap in high
    dimension, and it samples rather than learns the restart payoff. Restart
    fires only when a fresh draw's expected improvement exceeds continuing the
    path's (continue-payoff vs restart-payoff) - both closed forms off the same
    evidence. The i.i.d. assumption is what makes this exact: every path starts
    as an i.i.d. draw from the injected sampler, so each recorded group-min is
    a real sample of "what a fresh start buys", and the shared pool below
    estimates that single global payoff distribution with minimal variance.

    The verdict compares two estimates of the same KDE form - the path's own
    one-step sample against the pool's - and what makes the comparison readable
    is that the two are worth about the same number of samples: a shorter
    continue side reads the last few steps' own noise instead of the stage's, an
    understated spread shrinks EI_continue, and the restart tail wins verdicts
    the evidence does not support. That bias never shows up as an error, only as
    a worse incumbent, so the rule **derives** the two counts instead of
    exposing a knob: the continue side is read over `_stage_window()` = the
    pool's size M, which is the most samples a verdict can have in as many
    iterations as the restart side already has, and a caller cannot ask for a
    window its own pool cannot back.

    Equal weights are the only form that uses a whole stage: a windowed estimate
    that fades its older samples is worth fewer effective samples than it holds
    (its Kish size tends to (2 - alpha)/alpha, never above the number of folds),
    so on a stage that ends in a verdict discounting can only throw evidence
    away. Recency would only pay if the drift within one stage were large, and a
    stage is as long as the pool is large - short. Nothing is lost over a longer
    stage either: a restart drops the path's stage (the next Δ opens a new one
    against the pool as it stands), and the verdict is asked again every
    `_stage_window()` iterations, so the loop re-reads the path as regularly as it
    collects evidence.

    `group_mins` is **one pool for every path**, not a copy per path: a group-min
    is a draw from the sampler, the same quantity for all paths, so pooling every
    event estimates that single global payoff distribution with the smallest
    variance - a per-path subset would
    estimate the same global value from a shorter sample. NaN/inf Δ are kept in
    a path's stage but never read (guard), and still count towards the window.
    The incumbent B is a should_restart argument, not a field (the engine owns
    B), and it is the raw minimum: unfiltered, like the group-mins in the pool and
    like the `f_k` it is compared with.
    """

    def __init__(self) -> None:
        """Start with an empty group-min pool and no per-path evidence."""
        self.group_mins: list[float] = []  # one shared group-min pool
        self.pool_size = 0  # usable pool samples: the M the window is matched to
        self.deltas: dict[int, list[float]] = {}  # per path: Δ since its last verdict

    def observe_continue(self, i: int, delta: float) -> None:
        """Append path i's one-step value change Δ to its current stage.

        A non-finite Δ is stored, not dropped: a stalled path folds one every
        iteration and its verdict has to stay due on the same clock as its moving
        cohort (the engine reports the stall through the value, not by skipping
        the call), so the guard belongs at the read, where one +inf sample must
        not enter the KDE - and `NaN > EI_continue` is False, which silently
        disables that path's restarts.
        """
        self.deltas.setdefault(i, []).append(delta)

    def observe_restart(self, group_min: float) -> None:
        """Append one restart event's group minimum to the shared group-min pool.

        Non-finite group-mins stay in the list (`_restart_payoff` filters them,
        and a single +inf must not corrupt the KDE) but do not count towards M:
        the window is matched to the samples the KDE is actually built on.
        """
        self.group_mins.append(group_min)
        if math.isfinite(group_min):
            self.pool_size += 1

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:
        """Return True iff path i's verdict is due and restart beats continuing.

        Self-gated: due once the path has folded a full window of Δ since its
        last verdict, and the stage closes here whichever way the verdict goes -
        the gate is what opens a verdict, so a stage left open would be re-asked
        unchanged every iteration. A full stage also closes while there is no
        incumbent (B = inf, i.e. no finite point of any kind served yet): the
        verdict has nothing to compare against, and the engine owns B.

        Cold start is the one verdict that is not the EI comparison: with an
        empty group-min pool there is no restart evidence at all, so this returns
        True and the engine's first iteration probes every path. A pool that is
        non-empty but entirely non-finite is *not* cold start - the probe ran and
        every start was dead, so an empty `group_mins`, not `pool_size == 0`, is
        the test.
        """
        if not self.group_mins:  # no probe has run: draw the initialization one
            return True
        if len(self.deltas.get(i, ())) < self._stage_window():
            return False
        restart = math.isfinite(incumbent) and (
            self._restart_payoff(incumbent) > self._continue_payoff(i, f_k, incumbent)
        )
        self.deltas[i] = []  # the stage's one read is taken; the next Δ opens one
        return restart

    def restart(self, i: int) -> None:
        """Drop path i's stage after a restart (the shared group-min pool is kept)."""
        self.deltas[i] = []

    def _stage_window(self) -> int:
        """Return how many Δ a path is read over: the pool's size.

        Read fresh at each verdict rather than fixed per stage, because the pool
        only grows - samples are appended, never dropped or aged out - so `M` at
        the verdict is an upper bound on the stage that produced it, and matched
        counts is the property the verdict needs. Nothing pads it: the KDE
        weights every sample equally, so extra folds are extra evidence and no
        fold buys more than its weight.

        The only floor is 1, and it is not a prior: a window of zero samples has
        no centre to read (`_continue_payoff` returns 0.0 on an empty stage),
        which is reachable only before any finite group-min exists. At M = 1
        the read is its own degenerate branch - one fold has no spread (MAD = 0,
        so h = 0), and a pool of one gives `EI_restart = max(B - group_min, 0) <= 0`,
        so no restart can fire until a second group-min lands.

        This method is the seam a rule variant moves (a capped window, a forced
        cadence for an experiment); the rule itself takes no knob, so its own
        cadence can only follow the evidence.
        """
        return max(1, self.pool_size)

    def _stage_deltas(self, i: int) -> list[float]:
        """Return path i's finite Δ values from the stage it is in now."""
        return [delta for delta in self.deltas.get(i, ()) if math.isfinite(delta)]

    def _continue_payoff(self, i: int, f_k: float, incumbent: float) -> float:
        """Return the expected improvement of path i continuing one more step.

        Same scott_mad KDE as the restart side, over the stage's Δ: the read is
        worth exactly the samples it holds. An empty stage returns 0.0 - the
        degenerate "continuing buys nothing" read - which leaves the verdict
        resting on the restart side's own evidence. One fold (or a constant
        stage) has MAD = 0, so h = 0 and the read is exactly max(B - (f_k + Δ), 0).

        The next value is f_k + Δ, so the KDE's centres are f_k + Δ_j while its
        bandwidth comes from the Δ themselves - no single Gaussian is assumed, so
        a skewed Δ stream is read as it is.
        """
        deltas = self._stage_deltas(i)
        if not deltas:
            return 0.0
        h = _scott_mad_bandwidth(deltas)
        total = sum(_gaussian_ei(h, incumbent, f_k + delta) for delta in deltas)
        return total / len(deltas)

    def _restart_payoff(self, incumbent: float) -> float:
        """Return the expected improvement of any path restarting (global).

        The pool holds one group-min per path start, so it is read as a mixture
        of draws - a scott_mad KDE (one Gaussian kernel per sample) - rather than
        one Gaussian fitted to all of it. M = 0 → 0.0 (no evidence); M = 1 (or a
        constant pool) → h = 0 → degenerate max(B - group_min, 0).

        Skipping isfinite, not just NaN, is load-bearing: one +inf sample would
        blow the bandwidth to inf (or NaN), so the read stops being
        evidence-backed and the verdict either fires unconditionally (h = inf)
        or never fires again (h = NaN).
        """
        group_mins = [v for v in self.group_mins if math.isfinite(v)]
        if not group_mins:
            return 0.0
        h = _scott_mad_bandwidth(group_mins)
        total = sum(_gaussian_ei(h, incumbent, group_min) for group_min in group_mins)
        return total / len(group_mins)


def _gaussian_ei(sigma: float, incumbent: float, mu: float) -> float:
    """Return the expected improvement over the incumbent `B` for X ~ N(mu, sigma²).

    EI = sigma·(z·Φ(z) + φ(z)) with z = (B - mu)/sigma; sigma = 0 degenerates
    exactly to max(B - mu, 0). The scott_mad KDE on both sides decomposes into
    one call per sample: EI_continue at mu = f_k + Δ_j and EI_restart at mu = r_i.
    """
    if sigma == 0.0:
        return max(incumbent - mu, 0.0)
    z = (incumbent - mu) / sigma
    return sigma * (z * _normal_cdf(z) + _normal_pdf(z))


def _scott_mad_bandwidth(samples: list[float]) -> float:
    """Return the scott_mad KDE bandwidth 1.06 · mad_scale · n^(-1/5).

    Scott's normal-reference bandwidth on a robust MAD scale: the breadth is a
    property of the sample's body rather than of a long tail, which is what a
    plain sd gets wrong on a wide start mix. Degenerates to h = 0 (empirical EI)
    when the sample has no spread.
    """
    return SCOTT * _mad_sigma(samples) * len(samples) ** -0.2


def _mad_sigma(samples: list[float]) -> float:
    """Return the MAD-based scale estimate MAD / Φ⁻¹(3/4).

    Median absolute deviation, normal-consistent: for a Gaussian it estimates the
    sd with the same breakdown resistance the median has. One sample - or a sample
    with no spread - gives 0, the degenerate bandwidth the KDE reads as h = 0.
    """
    med = statistics.median(samples)
    return statistics.median([abs(v - med) for v in samples]) / PHI14


def _normal_cdf(z: float) -> float:
    """Return the standard normal CDF Φ(z) = (1 + erf(z/√2))/2."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_pdf(z: float) -> float:
    """Return the standard normal PDF φ(z) = exp(-z²/2)/√(2π)."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
