"""Alpha-composited PNG sprites, for drawing artwork into the camera frame.

Kept apart from game.py for the same reason the game is kept apart from
jump_game.py: none of this needs a GPU or a webcam, so it can be exercised
against a blank numpy array in a test.

Everything here assumes **BGRA with straight (non-premultiplied) alpha**,
which is what `cv2.imread(..., IMREAD_UNCHANGED)` hands back for a normal
RGBA PNG export. The alpha is the whole point: these are drawn over a live
camera image, so a sprite with a baked-in background reads as a sticker.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# The artwork lives at the top of the repo, not beside the code: it is data,
# not source, and it is what someone dropping in a new logo goes looking for.
# Hence parent.parent -- this file sits in src/.
#
# JUMP_ASSETS_DIR overrides that, for the same reason JUMP_FONT exists in
# gameover.py: once installed, the modules live in site-packages and the
# artwork does not, so parent.parent no longer points anywhere useful. The
# packaged wrapper sets it; from a checkout it is unset and nothing changes.
ASSETS = Path(os.environ.get("JUMP_ASSETS_DIR") or Path(__file__).resolve().parent.parent / "images")


def blit(frame, premul, inv, cx, cy):
    """Composite a premultiplied sprite centred on (cx, cy) in `frame` (BGR).

    `premul` is the colour already multiplied by its own alpha and `inv` is
    (255 - alpha) widened to three channels, both built once per size by
    Sprite.sized(). That reduces the per-frame work to dst = dst*inv/255 +
    premul: two SIMD passes instead of a float32 round trip, measured 0.30 ms
    against 3.47 ms on the Jetson at the size the obstacles now reach. The
    integer rounding costs at most 1/255 per channel against the float
    result, which is not visible.

    Clipped against the frame edges rather than wrapped or skipped, so a
    sprite can walk off the side of the screen the way the obstacles do.
    OpenCV writes correctly through the non-contiguous sub-rect view.
    """
    sh, sw = premul.shape[:2]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + sw), min(frame.shape[0], y0 + sh)
    if fx0 >= fx1 or fy0 >= fy1:
        return  # entirely off-frame

    sy, sx = slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0)
    dst = frame[fy0:fy1, fx0:fx1]
    cv2.multiply(dst, inv[sy, sx], dst=dst, scale=1.0 / 255.0)
    cv2.add(dst, premul[sy, sx], dst=dst)


def _outlined(bgra, r):
    """Add an `r`-pixel dark halo around the artwork.

    Without it a sprite competes with whatever the camera happens to be
    looking at, and the pale half of a two-tone logo disappears against a
    bright wall. Same trick as the black under-stroke on the stick figure's
    limbs, just done through the alpha channel instead of a second draw.
    """
    if r < 1:
        return bgra
    pad = r + 1
    out = cv2.copyMakeBorder(bgra, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    halo = cv2.dilate(out[:, :, 3], k)

    # Composite the artwork over an opaque-black layer masked by the halo.
    # Straight alpha over black is just rgb * a_src / a_out.
    a_s = out[:, :, 3].astype(np.float32) / 255.0
    a_h = halo.astype(np.float32) / 255.0
    a_out = a_s + a_h * (1.0 - a_s)
    scale = np.divide(a_s, a_out, out=np.zeros_like(a_s), where=a_out > 0)
    res = np.empty_like(out)
    res[:, :, :3] = (out[:, :, :3].astype(np.float32) * scale[:, :, None]).astype(np.uint8)
    res[:, :, 3] = (a_out * 255).astype(np.uint8)
    return res


def _trim(bgra):
    """Crop away fully transparent margins.

    Sizes are quoted as the height of the *picture*, so a margin an export
    happened to leave would silently shrink one asset against another -- and,
    for an obstacle, float it off the ground line it is supposed to sit on.
    """
    ys, xs = np.nonzero(bgra[:, :, 3])
    if len(ys) == 0:
        return bgra
    return bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class Sprite:
    """One PNG, resized on demand and cached by the size it was asked for.

    Cached because obstacle sizes are drawn from a *continuous* range, so
    without one every frame would re-resize every sprite on screen. Keying on
    whole pixels of height bounds the cache: the game only ever asks for
    heights across a ~30 px span, plus the one the head is drawn at.
    """

    OUTLINE = 0.045  # halo radius, in sprite heights

    def __init__(self, bgra, name=None):
        self.name = name  # the images/ filename, for taunts.json to match on
        # A PNG with no alpha is a supported case, not merely a tolerated
        # one: some logos only exist matted onto a solid background, and
        # keying that background out can do more harm than leaving it. Red
        # Hat is the example -- its white is load-bearing, filling the face
        # silhouette under the hat and the counters of the letters in
        # "redhat", none of which is reachable from the border. Keying by
        # connectivity ate the silhouette and hollowed the wordmark out into
        # outlines. Drawn whole it reads as a sticker, which is worse in
        # principle and better on screen; the outline pass frames it.
        if bgra.shape[2] == 3:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
        bgra = _trim(bgra)
        self.src = bgra
        self.aspect = bgra.shape[1] / bgra.shape[0]
        self._cache = {}

    @classmethod
    def load(cls, name):
        """Load images/<name>, or return None if it is not there.

        Missing artwork is not fatal: the assets are big binaries in a repo
        that otherwise holds none, so a fresh checkout may well not have them
        and the game should still be playable in blocks-and-circles form.
        """
        path = ASSETS / name
        # Checked before imread, which warns on stderr for a missing file --
        # and a missing asset is an expected state here, not an error.
        if not path.is_file():
            return None
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if bgra is None else cls(bgra, name)

    def sized(self, h, tint=None):
        """(premul, inv) for a copy `h` px tall (before the halo), for blit().

        `tint` blends the artwork towards a colour -- used to redden the
        avatar on death, which the old flat-coloured head got for free.
        """
        h = max(2, int(round(h)))
        key = (h, tint)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        w = max(2, int(round(h * self.aspect)))
        # INTER_AREA is the one that does not alias when shrinking, and these
        # are only ever shrunk: sources are ~256 px, targets are ~40-70.
        small = cv2.resize(self.src, (w, h), interpolation=cv2.INTER_AREA)
        if tint is not None:
            rgb = small[:, :, :3].astype(np.float32)
            small = small.copy()
            small[:, :, :3] = (rgb * 0.35 + np.float32(tint) * 0.65).astype(np.uint8)
        out = _outlined(small, round(h * self.OUTLINE))
        # Split into the two arrays blit() wants, once, here -- this is the
        # whole point of the cache.
        a = out[:, :, 3:4].astype(np.uint16)
        premul = ((out[:, :, :3].astype(np.uint16) * a) // 255).astype(np.uint8)
        inv = np.repeat((255 - out[:, :, 3:4]), 3, axis=2)
        self._cache[key] = (premul, inv)
        return self._cache[key]

    def draw(self, frame, cx, cy, h, tint=None):
        """Draw centred on (cx, cy), `h` pixels tall."""
        premul, inv = self.sized(h, tint)
        blit(frame, premul, inv, cx, cy)
