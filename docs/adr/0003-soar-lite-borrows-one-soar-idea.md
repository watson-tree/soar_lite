# "soar-lite" borrows one SOAR idea (sample-estimated restart payoff); it is not a SOAR port

The package name honours Mathesen et al. 2021 ("SOAR") but the "-lite" suffix is
the decision: this library keeps **one** idea from that framework — a restart's
payoff is *measured* by the minimum of freshly sampled starts, not *modelled*
by a fitted surrogate — and realizes it minimally. The surrogate/ensemble model,
the statistical acceptance test, multi-fidelity, and SOAR's interfaces are
deliberately **not** brought over.

The trade-off: borrowing the name invites readers to expect a SOAR
reimplementation, but a full adaptive-restart framework would re-introduce the
expensive surrogate fitting this package exists to avoid (high-dimensional and
budget-bound). The name plus an explicit "difference from SOAR" section on the
`How it works` page is cheaper than renaming to an opaque novelty, and it pays
the attribution honestly.

Consequence: do not "finish" the SOAR port under this name, do not add a
surrogate model and call it SOAR-compatible, and do not rename toward fuller
compatibility. The relationship is attribution + distinction, and that page's
SOAR section is the canonical statement of it.