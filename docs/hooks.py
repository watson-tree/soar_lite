"""Build hook: mark the Python blocks the theme's highlighter would guess at.

Fenced blocks carry their language from the fence, but the blocks mkdocstrings
generates - signatures and docstring examples - carry none, so highlight.js falls
back to guessing and typeset the reference page as PHP and Makefile. Every block
on these pages that arrives unlabelled is Python.

A `<code>` that already named its language is left alone, so fences keep theirs.
"""

import re

_PRE = re.compile(r"<pre\b[^>]*>.*?</pre>", re.DOTALL)
_UNLABELED = re.compile(r"<code\b(?![^>]*\bclass=)")


def _label(block: re.Match[str]) -> str:
    return _UNLABELED.sub('<code class="language-python"', block.group(0))


def on_page_content(html: str, **_kwargs: object) -> str:
    """Add `language-python` to every code block that arrived without one."""
    return _PRE.sub(_label, html)
