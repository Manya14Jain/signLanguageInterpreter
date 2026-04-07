"""
Word suggestion engine — prefix-based lookup against the system dictionary.
Uses /usr/share/dict/words (macOS/Linux) with a compact fallback list.
"""

import os
import bisect

# ── Compact fallback (~300 common words) ─────────────────────────────────────
_FALLBACK = sorted(set("""
able about above action add after again age air all allow almost alone
already also always among and another answer any appear area around ask
away back ball based because become before begin behind believe best
better between big black blue body book both bring call came can care
cause change child city clear close come common consider control could
country create dark day decide deep different door down drive early easy
end enough even ever every example face fact fall family far feel fill
find fire five floor follow food force form free friend front full
future game get give good great green group grow hand happen have head
hear heart here high hold home hope house how hundred idea if important
include interest into job keep kind know land language large last late
lead learn leave left less level like little live long look love main
make man may mean member mind month more most move much must name near
need never next nice night nothing number often open order other our out
over own page part past people person place plan plant play possible
power present problem real reason red remain remember result right road
room run same say school seem set show side since sky sleep slow small
social some sound speak stand start state stay still stop story strong
student sun take talk teach team tell then thing think three through
time together top tree true turn under very view visit voice wait walk
want water well when where while white who will within without woman
wonder word work world write year young
""".split()))


def _load_words() -> list[str]:
    for path in ('/usr/share/dict/words', '/usr/dict/words'):
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8', errors='ignore') as f:
                    return sorted(set(
                        w.strip().lower() for w in f
                        if w.strip().isalpha() and 2 <= len(w.strip()) <= 13
                    ))
            except OSError:
                pass
    return _FALLBACK


class WordEngine:
    """Efficient prefix-based word suggestion using bisect O(log N + k)."""

    def __init__(self):
        self._words = _load_words()

    def suggest(self, prefix: str, n: int = 6) -> list[str]:
        """Return up to n dictionary words beginning with prefix."""
        if not prefix:
            return []
        p = prefix.lower().strip()
        if not p.isalpha():
            return []
        idx = bisect.bisect_left(self._words, p)
        results: list[str] = []
        for word in self._words[idx: idx + 800]:
            if word.startswith(p):
                results.append(word)
                if len(results) >= n:
                    break
            elif word >= p[0:1] + 'z' * 14:
                break
        return results

    def apply(self, current: str, suggestion: str) -> str:
        """Replace the last partial word in current with suggestion + space."""
        if not current or current.endswith(' '):
            return current + suggestion + ' '
        parts = current.rsplit(' ', 1)
        prefix = parts[0] + ' ' if len(parts) > 1 else ''
        return prefix + suggestion + ' '
