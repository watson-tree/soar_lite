# The engine owns the LocalSolver/RestartRule protocols; plugins depend on the engine

`engine.py` defines both injected-interface protocols (`LocalSolver`,
`RestartRule`); `solver.py` and `restart.py` import them from the engine, and
the concrete defaults (`DiagonalPreconditioner`, `EIRestartRule`) are injected
by lazy import inside `n_path_schedule` when the caller passes `None`. This is
textbook dependency inversion: the engine owns the shape of its seams, and the
plugins depend on the engine, never the reverse.

The alternatives we rejected were:

- a shared `types.py` module that both sides import — extra indirection with no
  second consumer, and it splits the contract away from the thing that enforces
  it;
- **plugin-owned** interfaces — this would point the dependency the wrong way
  (engine importing its own contract from the plugins it runs).

Held by lazy import specifically to avoid a module-level import cycle, since
the engine must not import the plugins at import time.

Consequence: a new solver or restart rule implements the engine's protocol and
is handed in; it must not define the interface the engine should satisfy. Do
not extract the protocols into a neutral module "so both depend on it" — that
undoes the ownership this decision fixes.