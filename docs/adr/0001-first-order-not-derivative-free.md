# soar-lite is first-order: value + gradient, no Hessian, no zero-order fallback

The README once described this package as "derivative-free", but the code has
always consumed gradients returned by `evaluate_batch`. We decided the package
is **first-order** and will stay that way: the caller returns `(f, ∇f)`
together, and there is no Hessian, no curvature model, and no finite-difference
gradient fallback.

The trade-off was real, between three ranks:

- **zero-order** (value only): rejected — on the dfbench reference it was
  beaten by random restart alone, so a restart-based method gains nothing by
  dropping the gradient;
- **first-order** (value + gradient): chosen — the cheap batched gradient is
  the premise the diagonal-preconditioned step is built on;
- **second-order** (Hessian): rejected — too expensive to carry for the
  multi-path budget, and it would not be a diagonal method.

Consequence: a future "graceful degradation" to derivative-free (e.g. a
`fd_grad` option) or a curvature model would reverse this decision, so it must
be re-litigated, not silently added. A doc that reads "derivative-free" is a
bug to fix, not a feature to implement.