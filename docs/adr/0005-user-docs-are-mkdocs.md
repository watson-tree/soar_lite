# User docs are MkDocs + mkdocstrings, not Sphinx

The user documentation site is MkDocs with the `mkdocstrings` plugin, written in
Markdown and generated from the docstrings rather than duplicated by hand. Site
sources live in `docs/` beside the internal notes; the internal notes are
excluded from the build.

The decision turns on one asymmetry. The docstrings are already Markdown, and the
editor's hover renders them as Markdown. `mkdocstrings` (through Griffe) parses
the Google-style sections and renders their bodies as Markdown, so hover and site
read the same text. Sphinx renders docstring bodies as reST; Markdown docstrings
need `myst-parser` plus the third-party `sphinx-autodoc2`. This repository's own
notes are Markdown too, so reST would be the odd one out.

Two consequences worth carrying forward. The theme is the built-in `readthedocs`
one: Material for MkDocs is in maintenance mode (critical fixes only, end of life
2027-05-05) and its successor Zensical reads the same `mkdocs.yml`, so the theme
can change later without redoing the site. And "docstrings are Markdown" means
Markdown rendering rules apply: a line starting with `>` is a block quote, so
ranges and constraints stay inline rather than starting a continuation line.
