"""
ASL Gesture Engine v4 — Full A–Z (excluding J & Z which require motion)
==========================================================================
All 24 static signs supported with improved disambiguation.

Key technique: angle-at-joint (PIP/DIP) + tip-to-wrist ratio + direction
vectors. Each "ambiguous cluster" has dedicated disambiguation logic:

  Cluster 1 — Closed-fist variants : A  E  M  N  O  C  S  T
  Cluster 2 — Index-only            : D  G  Q  X
  Cluster 3 — Index+Middle          : H  K  R  U  V
  Cluster 4 — Distinct shapes       : B  F  I  L  W  X  Y  P
"""

import numpy as np
from collections import deque

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20


class GestureEngine:
    def __init__(self, buffer_size: int = 20):
        self.buffer = deque(maxlen=buffer_size)
        self._no_hand = 0

    # ── Low-level geometry ────────────────────────────────────────────────────

    def _pt(self, lm, i):
        return np.asarray(lm[i], dtype=np.float32)

    def _dist(self, lm, a, b):
        return float(np.linalg.norm(self._pt(lm, a) - self._pt(lm, b)))

    def _palm(self, lm):
        return max(self._dist(lm, WRIST, MIDDLE_MCP), 1e-6)

    def _angle(self, lm, a, v, c):
        """Angle at vertex v.  180 = straight,  ~90 = sharply bent."""
        ba = self._pt(lm, a) - self._pt(lm, v)
        bc = self._pt(lm, c) - self._pt(lm, v)
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom < 1e-6:
            return 180.0
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))

    def _ratio(self, lm, tip, pip):
        """dist(tip,wrist)/dist(pip,wrist). >EXT_TH → extended."""
        return self._dist(lm, tip, WRIST) / max(self._dist(lm, pip, WRIST), 1e-6)

    def _spread(self, lm, a, b):
        return self._dist(lm, a, b) / self._palm(lm)

    def _unit(self, lm, a, b):
        v = self._pt(lm, b) - self._pt(lm, a)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    # ── Feature bank ──────────────────────────────────────────────────────────

    def _features(self, lm):
        EXT = 1.10;  CURL = 0.95

        rI = self._ratio(lm, INDEX_TIP,  INDEX_PIP)
        rM = self._ratio(lm, MIDDLE_TIP, MIDDLE_PIP)
        rR = self._ratio(lm, RING_TIP,   RING_PIP)
        rPk = self._ratio(lm, PINKY_TIP, PINKY_PIP)

        I  = rI  > EXT;   M  = rM  > EXT
        R  = rR  > EXT;   Pk = rPk > EXT

        Ic = rI < CURL;   Mc = rM < CURL
        Rc = rR < CURL;   Pkc = rPk < CURL

        T = (self._dist(lm, THUMB_TIP, INDEX_MCP)
             > self._dist(lm, THUMB_MCP, INDEX_MCP) * 0.87)

        # PIP angles (180=straight, 90=bent)
        aIp  = self._angle(lm, INDEX_MCP,  INDEX_PIP,  INDEX_DIP)
        aMp  = self._angle(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP)
        aRp  = self._angle(lm, RING_MCP,   RING_PIP,   RING_DIP)
        aPkp = self._angle(lm, PINKY_MCP,  PINKY_PIP,  PINKY_DIP)
        aId  = self._angle(lm, INDEX_PIP,  INDEX_DIP,  INDEX_TIP)

        sIM  = self._spread(lm, INDEX_TIP,  MIDDLE_TIP)
        sMR  = self._spread(lm, MIDDLE_TIP, RING_TIP)
        sIT  = self._spread(lm, INDEX_TIP,  THUMB_TIP)

        iv = self._unit(lm, INDEX_MCP,  INDEX_TIP)
        mv = self._unit(lm, MIDDLE_MCP, MIDDLE_TIP)
        tv = self._unit(lm, THUMB_MCP,  THUMB_TIP)

        idx_up    = iv[1] < -0.42
        idx_horiz = abs(iv[0]) > abs(iv[1]) * 0.80
        idx_down  = iv[1]  >  0.38
        mid_horiz = abs(mv[0]) > abs(mv[1]) * 0.80
        thm_lat   = abs(tv[0]) > abs(tv[1]) * 0.70

        # Semi-flexed / curved state — not fully up, not fully curled
        def _curved(tip, pip, mcp):
            return lm[pip][1] < lm[tip][1] < lm[mcp][1]

        Icv  = _curved(INDEX_TIP,  INDEX_PIP,  INDEX_MCP)
        Mcv  = _curved(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        Rcv  = _curved(RING_TIP,   RING_PIP,   RING_MCP)
        Pkcv = _curved(PINKY_TIP,  PINKY_PIP,  PINKY_MCP)

        pcx  = (lm[INDEX_MCP][0] + lm[PINKY_MCP][0]) / 2.0
        thm_side = abs(lm[THUMB_TIP][0] - pcx) > 0.09

        # Thumb vertically between INDEX_PIP and INDEX_MCP rows → S / A
        thm_fist = (lm[INDEX_PIP][1] - 0.04
                    < lm[THUMB_TIP][1]
                    < lm[INDEX_MCP][1] + 0.04)

        # Thumb x between INDEX_MCP and MIDDLE_MCP → T
        thm_btw = (min(lm[INDEX_MCP][0], lm[MIDDLE_MCP][0]) - 0.025
                   < lm[THUMB_TIP][0]
                   < max(lm[INDEX_MCP][0], lm[MIDDLE_MCP][0]) + 0.025)

        thm_pip = self._spread(lm, THUMB_TIP, INDEX_PIP) < 0.38
        I_hook  = aId < 148 and rI > 0.88 and not I

        return dict(
            I=I, M=M, R=R, Pk=Pk, T=T,
            Ic=Ic, Mc=Mc, Rc=Rc, Pkc=Pkc,
            aIp=aIp, aMp=aMp, aRp=aRp, aPkp=aPkp, aId=aId,
            sIM=sIM, sMR=sMR, sIT=sIT,
            idx_up=idx_up, idx_horiz=idx_horiz, idx_down=idx_down,
            mid_horiz=mid_horiz, thm_lat=thm_lat,
            thm_side=thm_side, thm_fist=thm_fist,
            thm_btw=thm_btw, thm_pip=thm_pip, I_hook=I_hook,
            Icv=Icv, Mcv=Mcv, Rcv=Rcv, Pkcv=Pkcv,
        )

    def get_finger_states_named(self, lm):
        f = self._features(lm)
        return {'thumb': f['T'], 'index': f['I'], 'middle': f['M'],
                'ring': f['R'], 'pinky': f['Pk']}

    # ── Classification tree ───────────────────────────────────────────────────

    def _classify(self, lm):  # noqa: C901
        f  = self._features(lm)
        I  = f['I'];   M  = f['M'];   R  = f['R'];   Pk = f['Pk'];  T = f['T']
        sIM = f['sIM']; sMR = f['sMR']; sIT = f['sIT']

        # ═══════════════════════════════════════════════════════════
        # B — all four fingers up
        # ═══════════════════════════════════════════════════════════
        if I and M and R and Pk:
            return 'B', 0.91

        # ═══════════════════════════════════════════════════════════
        # W — three fingers spread (index + middle + ring)
        # ═══════════════════════════════════════════════════════════
        if I and M and R and not Pk:
            return 'W', (0.91 if sIM > 0.40 and sMR > 0.34 else 0.79)

        # ═══════════════════════════════════════════════════════════
        # Cluster: Index + Middle  →  H / K / R / U / V
        # ═══════════════════════════════════════════════════════════
        if I and M and not R and not Pk:
            # H: both pointing sideways
            if f['idx_horiz'] and f['mid_horiz']:
                return 'H', 0.87
            # K: index+middle up, thumb near index PIP
            if T and f['thm_pip']:
                return 'K', 0.88
            # R: tips very close (crossed fingers)
            if not T and sIM < 0.20:
                return 'R', 0.83
            # V: peace / wide spread
            if sIM > 0.52:
                return 'V', 0.93
            # U: default — together and vertical
            return 'U', 0.87

        # ═══════════════════════════════════════════════════════════
        # L — thumb + index only
        # ═══════════════════════════════════════════════════════════
        if T and I and not M and not R and not Pk:
            return 'L', 0.94

        # ═══════════════════════════════════════════════════════════
        # Y — thumb + pinky
        # ═══════════════════════════════════════════════════════════
        if T and not I and not M and not R and Pk:
            return 'Y', 0.94

        # ═══════════════════════════════════════════════════════════
        # I — pinky only
        # ═══════════════════════════════════════════════════════════
        if not T and not I and not M and not R and Pk:
            return 'I', 0.93

        # ═══════════════════════════════════════════════════════════
        # Cluster: Index dominant  →  D / G / Q / X
        # ═══════════════════════════════════════════════════════════
        if not M and not R and not Pk and (I or f['I_hook']):
            # X: hooked index (DIP bent, partial extension)
            if f['I_hook']:
                return 'X', 0.84
            # G: index + thumb both horizontal (pointing gun sideways)
            if f['idx_horiz'] and T and f['thm_lat']:
                return 'G', 0.83
            # Q: index + thumb pointing downward
            if f['idx_down'] and T:
                return 'Q', 0.77
            # D: index up, others curl toward thumb
            if f['idx_up']:
                return 'D', (0.90 if sIT < 0.68 else 0.77)
            return 'D', 0.70

        # ═══════════════════════════════════════════════════════════
        # F — middle + ring + pinky up, index+thumb pinch
        # ═══════════════════════════════════════════════════════════
        if not I and M and R and Pk:
            if sIT < 0.40:
                return 'F', 0.91
            if sIT < 0.58:
                return 'F', 0.76
            # If index-thumb not close, it's unclear — let it fall to None
            return None, 0.0

        # ═══════════════════════════════════════════════════════════
        # P — like K but entire hand tilted downward
        # ═══════════════════════════════════════════════════════════
        if T and I and M and not R and not Pk and f['idx_down']:
            if f['thm_pip']:
                return 'P', 0.80
            return 'P', 0.70

        # ═══════════════════════════════════════════════════════════
        # Cluster: Closed / curved  →  A  C  E  M  N  O  S  T
        # ═══════════════════════════════════════════════════════════
        if not I and not M and not R and not Pk:

            all_curved = f['Icv'] and f['Mcv'] and f['Rcv'] and f['Pkcv']

            # ── O: all fingers semi-flexed + thumb-index pinch ──────────────
            if all_curved and sIT < 0.40:
                return 'O', 0.86

            # ── C: all fingers semi-flexed, open (thumb farther away) ───────
            if all_curved and sIT > 0.50:
                return 'C', 0.84

            # ── T: thumb sandwiched between index and middle MCPs ────────────
            if T and f['thm_btw'] and not f['thm_side']:
                return 'T', 0.83

            # ── E: ALL four PIP joints sharply bent, thumb tucked under ──────
            # (fingernails face roughly toward palm, no thumb extension)
            all_deep = (f['aIp'] < 126 and f['aMp'] < 126
                        and f['aRp'] < 126 and f['aPkp'] < 126)
            if all_deep and not T:
                return 'E', 0.85

            # ── N vs M: fingers folded over thumb ────────────────────────────
            # M: index + middle + ring all PIP bent  (<145°) over thumb
            # N: only index + middle bent; ring pip is notably straighter
            if T and not f['thm_side'] and not f['thm_btw']:
                three_bent = (f['aIp'] < 145
                              and f['aMp'] < 145
                              and f['aRp'] < 145)
                two_bent   = (f['aIp'] < 145
                              and f['aMp'] < 145
                              and f['aRp'] >= 145)   # ring NOT deeply bent
                if three_bent:
                    return 'M', 0.78
                if two_bent:
                    return 'N', 0.76

            # ── S vs A: fist — thumb position is the key ─────────────────────
            # S: thumb wraps across the FRONT of the knuckles (in fist zone)
            # A: thumb sticks OUT TO THE SIDE
            if T:
                if f['thm_side']:
                    return 'A', 0.85   # thumb outward → A
                if f['thm_fist']:
                    return 'S', 0.81   # thumb across fist front → S
                return 'A', 0.68       # ambiguous → default to A

            # No thumb and not deeply bent enough for E
            return 'S', 0.62          # closed fist, no thumb → S-like

        # ═══════════════════════════════════════════════════════════
        # Fallback
        # ═══════════════════════════════════════════════════════════
        return None, 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            self._no_hand += 1
            if self._no_hand >= 4:
                self.buffer.clear()
            return None, 0.0, {}

        self._no_hand = 0
        fs = self.get_finger_states_named(landmarks)
        letter, raw_conf = self._classify(landmarks)

        self.buffer.append(letter)
        n = len(self.buffer)
        if n >= 5:
            weights = [1.0 + i / n for i in range(n)]
            totals: dict = {}
            for i, ltr in enumerate(self.buffer):
                totals[ltr] = totals.get(ltr, 0.0) + weights[i]
            best  = max(totals, key=totals.get)  # type: ignore[arg-type]
            stab  = totals[best] / sum(weights)
            if stab >= 0.45 and best is not None:
                return best, min(raw_conf * (0.35 + stab * 0.65), 0.99), fs

        return letter, raw_conf * 0.50, fs

    def reset(self):
        self.buffer.clear()
        self._no_hand = 0
