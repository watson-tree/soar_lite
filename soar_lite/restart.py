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


# --- restart rule: the no-restart arm ---
class NoRestartRule(RestartRule):
    """Never restart: one seeding probe, then every path descends on its own.

    The degenerate opposite of `EIRestartRule`, for callers who want the pure
    multi-start descent (or the serial single chain) as a baseline instead of
    the EI rule's emergent behaviour. It implements the `RestartRule` contract
    itself rather than subclassing `EIRestartRule`, because none of that rule's
    pool/stage bookkeeping is ever read here.

    The cold start is the one verdict it cannot refuse: the engine asks every
    path before any probe has run, and a False there would never seed the flock.
    So the first ask answers True, the probe fills the pool, and every ask after
    it answers False. `_seeded` flips in `observe_restart` - which the engine
    calls after the probe - never in `should_restart`, which it calls for all
    paths before probing; flipping it in the query would seed only the first
    path and leave the rest on their `inf` placeholders forever.
    """

    def __init__(self) -> None:
        """Start unseeded: the engine's first ask is the seeding probe."""
        self._seeded = False

    def observe_continue(self, i: int, delta: float | None) -> None:
        """No-op: this rule never weighs continuing against restarting."""

    def observe_restart(self, group_min: float) -> None:  # noqa: ARG002
        """Record that the seeding probe ran; every ask after it answers False."""
        self._seeded = True

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Return True only for the cold start, which is how the flock seeds."""
        return not self._seeded

    def restart(self, i: int) -> None:
        """No-op: there is no per-path evidence to reset."""


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
    a worse final value, so the rule **derives** the two counts instead of
    exposing a knob: the continue side is read over the newest M folds of its
    stage, and `_stage_window()` = M + 1 is how many rounds that stage spans - M
    folds plus the round it opened in, which is a probe for a restarted path and
    the verdict itself for a continuing one. A caller cannot ask for a window its
    own pool cannot back. A stalled path reads no stage at all - its continue
    side is 0 by definition - so the two counts are only compared on paths that
    actually folded M.

    Equal weights are the only form that uses a whole stage: a windowed estimate
    that fades its older samples is worth fewer effective samples than it holds
    (its Kish size tends to (2 - alpha)/alpha, never above the number of folds),
    so on a stage that ends in a verdict discounting can only throw evidence
    away. Recency would only pay if the drift within one stage were large, and a
    stage is as long as the pool is large - short. Nothing is lost over a longer
    stage either: a restart drops the path's stage (the next Δ opens a new one
    against the pool as it stands), and the verdict is asked again every
    `_stage_window()` rounds, so the loop re-reads the path as regularly as it
    collects evidence.

    `group_mins` is **one pool for every path**, not a copy per path: a group-min
    is a draw from the sampler, the same quantity for all paths, so pooling every
    event estimates that single global payoff distribution with the smallest
    variance - a per-path subset would
    estimate the same global value from a shorter sample. A stalled iteration
    arrives as `None`, never as a value: it advances the path's stage clock and
    the read sees no continuation sample, so a stall is never averaged into a
    stage that spans it. In the pool every non-finite group-min is a zero term of
    the same mean, whose own improvement is exactly 0: a `+inf` is a dead start,
    and a `nan` or a `-inf` is an event outside the finite set that says as
    little about what a restart buys. All three dilute the finite KDE by the
    finite fraction |finite| / M instead of vanishing from the read, so the
    restart side is always a mean over the same M samples the window is.
    The raw best `best_fun_raw` is a should_restart argument, not a field (the
    engine owns it), and it is the raw minimum: unfiltered, like the group-mins
    in the pool and like the `f_k` it is compared with.
    """

    def __init__(self) -> None:
        """Start with an empty group-min pool and no per-path evidence."""
        self.group_mins: list[float] = []  # one shared group-min pool
        self.deltas: dict[int, list[float]] = {}  # per path: Δ since its last verdict
        self.rounds: dict[int, int] = {}  # per path: iterations since its last verdict

    def observe_continue(self, i: int, delta: float | None) -> None:
        """Fold path i's one-step change, or record a stall on its stage clock.

        Both shapes advance the path's stage clock by one iteration - one report
        per iteration is the engine's contract - so a stage comes due on the same
        round whatever mix of moves and stalls it took to get there. Only the
        evidence differs.

        `None` is the engine's stall report, not a value: the path did not move
        this iteration, so continuing bought nothing and its continue side is 0
        by definition. The stage is dropped - there is no continuation left to
        average - and the clock still counts the round. A finite Δ is the sample:
        it joins the stage and counts the same one round.

        A non-finite float is rejected: the engine spells a stall `None`, so an
        `inf`/`nan` here is a caller that lost the distinction, and storing it
        would put a value where absence belongs.
        """
        if delta is None:
            self.deltas[i] = []
        else:
            if not math.isfinite(delta):
                msg = f"a continue delta is finite or None (a stall), got {delta!r}"
                raise ValueError(msg)
            self.deltas.setdefault(i, []).append(delta)
        self.rounds[i] = self.rounds.get(i, 0) + 1

    def observe_restart(self, group_min: float) -> None:
        """Append one restart event's group minimum to the shared group-min pool.

        Every value is kept as served. A `+inf` is a dead start, a `nan` is a
        start that was never a number, and a `-inf` is a start below the finite
        range - all three are observations the read triages (`_restart_payoff`)
        as zero terms, not errors to drop at the door; only the read knows what
        each is worth.
        """
        self.group_mins.append(group_min)

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:
        """Return True iff path i's verdict is due and restart beats continuing.

        One clock, one comparison. The stage clock counts rounds, not folds: a
        cohort path advances it by one per `observe_continue`, whether it moved
        (a Δ) or stalled (`None`), and a restarted path is opened at 1 by this
        rule's `restart` hook - the round it was probed in. Every path therefore
        ticks exactly once per round and a stage is due on the same round as its
        cohort whatever it lived through - a path that stalls mid-stage keeps the
        wait it has already served instead of starting a second one, and a path
        that restarts does not open its next stage a round behind.

        A due stage is read against the engine's best: restart fires when
        `EI_restart >= EI_continue`, and a stalled stage is the same comparison at
        the degenerate `EI_continue` - `_continue_payoff` returns 0.0 on the empty
        stage a stall leaves, so a stalled path is exactly "does restarting buy
        anything at all". One formula covers both shapes; the difference is what
        the continue side has to read.

        The tie is the load-bearing half of that `>=`. Both sides read 0 when the
        evidence is degenerate - a pool with no spread or of dead starts, a
        continue side that lost ground or stalled - and continuing at 0 is worth
        nothing, while a probe is always one real i.i.d. group-min. The tie is
        therefore the price of buying the evidence neither side has. It cannot
        thrash because the window is the pool's sample count plus one: every tie
        probes, every probe appends a sample, so the next ask on that path comes
        a window later - the interval grows 1, 2, 3, ... and the probe rate falls
        to 0.

        **A verdict closes its stage** - `deltas` and the clock are cleared here,
        whichever way the comparison went. A window is one stage, so its end does
        not depend on the answer: leaving a stage open after a False verdict would
        re-ask it unchanged every iteration and turn the clock into "iterations
        since the last restart", which is not what a due test can be built on. The
        next Δ or stall opens the next stage, so a path that never gets restarted
        is re-judged once per window - the same cadence a continuing path has.

        With no best (`best_fun_raw` = `+inf`: no finite point of any kind served
        yet) there is nothing to weigh, and the same rule applies as at cold
        start - a verdict with nothing to compare is a probe. It restarts (and
        still closes the stage): the probe is the only action that can buy the
        first finite sample, so answering False would leave the one path that
        could re-draw blocked from asking, and a single unlucky initialization
        draw - one all-dead probe group - would lock the run until the horizon.
        The dead samples still count in the pool, so the window keeps growing and
        the re-probing decays like any other probe, 1, 2, 3, ... rounds apart.

        Cold start is the other verdict that is not the EI comparison: with an
        empty group-min pool there is no restart evidence at all, so this returns
        True and the engine's first iteration probes every path. A pool that is
        non-empty but entirely non-finite is *not* cold start - the probe ran and
        every start was dead - so an empty `group_mins` is the test; a dead pool
        prices the restart at 0 whatever made it dead (`+inf`, `nan` or `-inf`),
        which ties a stalled continue side and re-probes once per window, each
        dead sample spacing the next ask one round further out. With no finite
        best at all the branch above fires first, at that same cadence.
        """
        if not self.group_mins:  # no probe has run: draw the initialization one
            return True
        if self.rounds.get(i, 0) < self._stage_window():
            return False
        # Said outright rather than left to inf arithmetic: `_restart_payoff` and
        # `_continue_payoff` both read inf here, and inf >= inf happens to be
        # True today - a coincidence, not a decision.
        restart = not math.isfinite(best_fun_raw) or (
            self._restart_payoff(best_fun_raw)
            >= self._continue_payoff(i, f_k, best_fun_raw)
        )
        self.deltas[i] = []  # the stage is read; the next Δ or stall opens the next
        self.rounds[i] = 0
        return restart

    def restart(self, i: int) -> None:
        """Open path i's fresh stage clock at the round it was probed in.

        The engine calls this for every path it re-probes (and at initialization),
        in the same round the verdict fired in. The verdict already closed the
        stage (`should_restart`), so there is no evidence to drop here - but the
        round is not free either: the path spent it on its probe instead of a
        fold, while its cohort spent the same round folding and was ticked for
        it. Counting it here is what keeps every path on the same round as its
        cohort instead of one round late per restart.

        The clock is set, not incremented: the verdict that just ran left it at 0,
        and this round is the new stage's first, so M further folds make the
        stage M + 1 rounds - exactly `_stage_window()`.
        """
        self.rounds[i] = 1

    def _continue_payoff(self, i: int, f_k: float, best_fun_raw: float) -> float:
        """Return the expected improvement of path i continuing one more step.

        Same scott_mad KDE as the restart side, over the stage's Δ: the read is
        worth exactly the samples it holds. An empty stage returns 0.0 - the
        degenerate "continuing buys nothing" read (what a stalled path always
        has, since a stall drops its stage) - which leaves the verdict resting on
        the restart side's own evidence. One fold (or a constant stage) has MAD =
        0, so h = 0 and the read is exactly max(f⋆ - (f_k + Δ), 0).

        Only the newest M Δ are read. A path whose verdict said continue holds M
        + 1 of them at its next verdict: the stage opened in the round it was
        judged in, and that round folded one sample more than a restarted path's
        probe round did. Reading the newest M is what gives the two the same
        count, and the window already spends the extra round on the verdict
        itself (`_stage_window`), so nothing here waits a round longer to collect
        it.

        The next value is f_k + Δ, so the KDE's centres are f_k + Δ_j while its
        bandwidth comes from the Δ themselves - no single Gaussian is assumed, so
        a skewed Δ stream is read as it is.
        """
        # The newest M: `[-0:]` is the whole list, so the empty pool is spelled
        # out instead of borrowed from the slice.
        m = self._sample_count()
        deltas = self.deltas.get(i, [])
        deltas = deltas[-m:] if m else []
        if not deltas:
            return 0.0
        h = _scott_mad_bandwidth(deltas)
        total = sum(_gaussian_ei(h, best_fun_raw, f_k + delta) for delta in deltas)
        return total / len(deltas)

    def _restart_payoff(self, best_fun_raw: float) -> float:
        """Return the expected improvement of any path restarting (global).

        One mean over the whole pool, so the restart side reads the same number
        of samples the window gives the continue side. The finite group-mins are
        read as a scott_mad KDE (one Gaussian kernel per sample), the bandwidth
        taken from the finite subset alone so a zero term cannot widen it. Every
        other sample is a zero term of that same mean: a `+inf` is a dead start,
        a legal observation whose own improvement is exactly 0, and a `nan` or a
        `-inf` is an event outside the finite set - like `+inf` it says nothing
        about how much a restart would buy, so it buys 0 as well. Nothing is
        dropped from the denominator for being unreadable: the read is the
        finite mean scaled by the finite fraction |finite| / M, a constant in
        [0, 1] that can only lower the restart's price.

        Counting every sample is what keeps the two sides comparable: a pool of
        no-ops would otherwise read as a smaller, sharper pool and outprice a
        real stage.

        An empty pool and a pool with no finite sample both read 0.0 - there is
        no evidence for the mean to be positive about. M = 1 (or a constant
        finite pool) → h = 0 → degenerate max(f⋆ - group_min, 0).
        """
        finite = [v for v in self.group_mins if math.isfinite(v)]
        if not finite:
            return 0.0
        h = _scott_mad_bandwidth(finite)
        # A non-finite sample is the zero term of this same mean: restarting onto
        # it buys nothing, so it lowers the price without widening the bandwidth.
        total = sum(_gaussian_ei(h, best_fun_raw, group_min) for group_min in finite)
        return total / len(self.group_mins)

    def _stage_window(self) -> int:
        """Return how many rounds a stage spans: M folds plus its opening round.

        Read fresh at each verdict rather than fixed per stage, because the pool
        only grows - samples are appended, never dropped or aged out - so `M` at
        the verdict is an upper bound on the stage that produced it, and matched
        counts is the property the verdict needs. Nothing pads the read itself:
        the KDE weights every sample equally, so extra folds are extra evidence
        and no fold buys more than its weight.

        The `+ 1` is the round the stage opened in, which every stage spends
        without folding - the restarted path on its probe (`restart`), the
        continuing path on the verdict that closed the stage before. Every path
        that opens a stage in round T is therefore due together in round T + M,
        each reading M samples (`_continue_payoff` keeps the newest M for exactly
        that reason). Without the `+ 1` the restarted path's stage would open one
        round behind its cohort's and fall another round back at every restart.

        Every pooled group-min is a sample, whatever it is worth: the read is one
        mean over the whole pool and a non-finite sample is a zero term of it,
        not a gap in it (`_restart_payoff`). That is what spaces a stalled
        flock's re-probes, and it holds for any dead pool rather than only a
        `+inf` one: each probe appends a sample, so the next ask on that path
        comes a window further out - the interval grows 1, 2, 3, ... and the
        probe rate falls to 0.

        The smallest window is 1 (an empty pool), and `should_restart` answers
        that as cold start before this is read. M = 1 is the one-sample regime -
        one fold has no spread (MAD = 0, so h = 0), and a pool of one prices
        `EI_restart = max(f⋆ - group_min, 0)`, which is 0 whenever that sample is
        the best so far. Both sides then read 0 and the verdict ties, so the path
        probes once; the probe is what buys the second sample, and from M = 2 on
        the spread decides the verdict on its own.

        This method is the seam a rule variant moves (a capped window, a forced
        cadence for an experiment); the rule itself takes no knob, so its own
        cadence can only follow the evidence.
        """
        return self._sample_count() + 1

    def _sample_count(self) -> int:
        """Return M, the pool's length: the samples a verdict reads on either side.

        The window and the continue side's slice are both read off this one
        count, so the two sides of a verdict cannot drift apart: the restart side
        means over all M, the continue side over its newest M.
        """
        return len(self.group_mins)


def _gaussian_ei(sigma: float, best_fun_raw: float, mu: float) -> float:
    """Return the expected improvement over the raw best for X ~ N(mu, sigma²).

    EI = sigma·(z·Φ(z) + φ(z)) with z = (best_fun_raw - mu)/sigma; sigma = 0
    degenerates exactly to max(best_fun_raw - mu, 0). The scott_mad KDE on both
    sides decomposes into one call per sample: EI_continue at mu = f_k + Δ_j and
    EI_restart at mu = r_i.
    """
    if sigma == 0.0:
        return max(best_fun_raw - mu, 0.0)
    z = (best_fun_raw - mu) / sigma
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
