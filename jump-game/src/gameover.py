"""The game-over screen: a BUSTED-style panel naming whatever killed you.

Kept out of game.py because it is entirely presentation, and because it is
the only part of the game that wants a real typeface. cv2's Hershey fonts are
thin single-stroke vectors -- fine for a HUD, wrong for a title card -- so
this reaches for PIL and a TTF when it can, and falls back to Hershey when it
cannot. Neither is a hard dependency: a box with no PIL and no font still
gets a readable screen, just a plainer one.

The taunts live in taunts.json rather than here, as a flat list of
{image, title, text}. A flat list rather than a mapping so the same image can
appear as many times as you like with different lines, and one is picked at
random per death.
"""

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

import sprite

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # a readable screen still beats no screen
    Image = None

# JUMP_TAUNTS overrides this once installed, where the modules sit in
# site-packages and taunts.json does not -- same reasoning as sprite.ASSETS.
TAUNTS_PATH = Path(os.environ.get("JUMP_TAUNTS") or Path(__file__).resolve().parent.parent / "taunts.json")

# Colours (BGR).
C_TITLE = (255, 255, 255)
C_TITLE_SHADOW = (50, 50, 200)
C_SUB = (190, 190, 190)
C_SCORE = (120, 220, 255)
C_RULE = (60, 60, 190)
C_SHADOW = (0, 0, 0)


def find_font():
    """A heavy sans TTF, or None to fall back to Hershey.

    Resolved at runtime rather than hardcoded: the obvious path here is a Nix
    store path, which changes on every rebuild of the font package.
    """
    env = os.environ.get("JUMP_FONT")
    if env and Path(env).is_file():
        return env
    fc = shutil.which("fc-match")
    if fc:
        try:
            r = subprocess.run([fc, "-f", "%{file}", "DejaVu Sans:bold"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and Path(r.stdout.strip()).is_file():
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    for pat in ("/run/current-system/sw/share/fonts/**/DejaVuSans-Bold.ttf",
                "/nix/store/*dejavu*/**/DejaVuSans-Bold.ttf",
                "/nix/store/*liberation*/**/LiberationSans-Bold.ttf",
                "/usr/share/fonts/**/DejaVuSans-Bold.ttf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


class Type:
    """Turns strings into alpha masks, at a size that fits the space given."""

    def __init__(self, font_path=None):
        self.path = font_path if font_path is not None else find_font()
        self._faces = {}

    def _face(self, px):
        f = self._faces.get(px)
        if f is None:
            f = self._faces[px] = ImageFont.truetype(self.path, px)
        return f

    def mask(self, text, px, track=0.0, weight=0):
        """An 8-bit coverage mask, cropped to the ink.

        `track` is extra letter-spacing in ems; `weight` dilates the glyphs,
        which fattens a face far more convincingly than a heavier stroke
        setting would, and works the same on either rendering path.
        """
        px = max(8, int(px))
        if self.path is None or Image is None:
            m = self._hershey(text, px, track)
        else:
            f = self._face(px)
            adv = [f.getlength(c) + track * px for c in text]
            img = Image.new("L", (max(1, int(sum(adv)) + px), int(px * 1.9)), 0)
            d = ImageDraw.Draw(img)
            x = px * 0.4
            for c, a in zip(text, adv):
                d.text((x, px * 0.2), c, font=f, fill=255)
                x += a
            m = np.array(img)
        if weight >= 1:
            k = int(weight) | 1  # dilate wants an odd kernel
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return np.zeros((1, 1), np.uint8)
        return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    @staticmethod
    def _hershey(text, px, track):
        font, scale = cv2.FONT_HERSHEY_DUPLEX, px / 26.0
        th = max(1, round(px * 0.07))
        spaced = (" " * max(0, round(track * 4))).join(text) if track else text
        (w, h), base = cv2.getTextSize(spaced, font, scale, th)
        img = np.zeros((h + base + px, w + px), np.uint8)
        cv2.putText(img, spaced, (px // 2, h + px // 2), font, scale, 255, th, cv2.LINE_AA)
        return img

    def measure(self, text, px, track=0.0):
        """Width in pixels, from the font metrics, without rasterising.

        Fitting used to render a string just to discover it was too wide and
        then render it again smaller -- and the flavour line, which can wrap,
        cost up to five rasterisations before it settled. That was 30 ms on
        the frame the player dies on. Metrics answer the same question for
        nothing.
        """
        px = max(8, int(px))
        if self.path is None or Image is None:
            th = max(1, round(px * 0.07))
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, px / 26.0, th)[0][0]
        f = self._face(px)
        return int(sum(f.getlength(c) + track * px for c in text))

    def fitted(self, text, px, max_w, track=0.0, weight_frac=0.0):
        """Like mask(), but sized so it fits `max_w`. Rasterises once.

        Needed because the titles are wildly different lengths -- "YAST!"
        against "DOCKERFILED!" -- and the stage's width is a runtime choice.
        Without it the long ones simply ran off the right of the screen.
        """
        px = max(8, int(px))
        want = self.measure(text, px, track) + 2 * round(px * weight_frac)
        if want > max_w:
            px = max(8, int(px * max_w / want))
        m = self.mask(text, px, track, round(px * weight_frac))
        return m[:, :max_w] if m.shape[1] > max_w else m

    def wrapped(self, text, px, max_w, track=0.0, min_scale=0.78):
        """One line if it fits without shrinking past `min_scale`, else two.

        Shrinking the flavour line to fit is fine up to a point; past it the
        text gets smaller than the score under it and the panel looks broken.
        """
        px = max(8, int(px))
        w = self.measure(text, px, track)
        if w <= max_w or w * min_scale <= max_w:
            return [self.fitted(text, px, max_w, track)]
        words = text.split()
        # Split at the word boundary nearest the middle by character count.
        best, target = 1, len(text) / 2
        for i in range(1, len(words)):
            if abs(len(" ".join(words[:i])) - target) < abs(len(" ".join(words[:best])) - target):
                best = i
        return [self.fitted(" ".join(words[:best]), px, max_w, track),
                self.fitted(" ".join(words[best:]), px, max_w, track)]


class Stamp:
    """A mask in a colour, pre-split into the two arrays sprite.blit wants.

    A drop shadow is baked in rather than drawn as a second pass: the title
    is the largest thing on the screen, and compositing it twice a frame cost
    as much as everything else on the panel put together.
    """

    def __init__(self, mask, colour, shadow=None, offset=0):
        if shadow is not None and offset > 0:
            mask, premul = self._shadowed(mask, colour, shadow, offset)
        else:
            premul = cv2.merge([cv2.convertScaleAbs(mask, alpha=c / 255.0) for c in colour])
        self.mask = mask
        self.premul = premul
        self.inv = cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _shadowed(mask, colour, shadow, off):
        """Composite `mask` over a copy of itself offset down-right.

        Straight 'over' in premultiplied form, which is what blit() consumes
        anyway: premul = top*a + under*(1-a), alpha = a + under_a*(1-a). Done
        with cv2 rather than numpy floats -- this runs on the frame the
        player dies on, and the float version of these same six lines took
        10 ms of it.
        """
        h, w = mask.shape
        hi = np.zeros((h + off, w + off), np.uint8)
        hi[:h, :w] = mask
        lo = np.zeros_like(hi)
        lo[off:, off:] = mask
        lo = cv2.multiply(lo, cv2.bitwise_not(hi), scale=1 / 255.0)  # shadow where no glyph
        alpha = cv2.add(hi, lo)
        premul = cv2.merge([cv2.addWeighted(hi, colour[i] / 255.0, lo, shadow[i] / 255.0, 0)
                            for i in range(3)])
        return alpha, premul

    @property
    def w(self):
        return self.mask.shape[1]

    @property
    def h(self):
        return self.mask.shape[0]

    def draw(self, frame, x, cy, scale=1.0, clip=None):
        """Composite with the left edge at `x` and the middle at `cy`."""
        premul, inv = self.premul, self.inv
        if scale != 1.0:
            size = (max(1, int(self.w * scale)), max(1, int(self.h * scale)))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            premul = cv2.resize(premul, size, interpolation=interp)
            inv = cv2.resize(inv, size, interpolation=interp)
        if clip is not None:
            c = max(1, min(premul.shape[1], int(clip)))
            premul, inv = premul[:, :c], inv[:, :c]
        sprite.blit(frame, premul, inv,
                    x + premul.shape[1] / 2, cy)


class GameOverScreen:
    """Draws the death panel. One instance for the life of the game."""

    # The panel is the right two thirds of the screen, floor to ceiling.
    #
    # It used to be a horizontal band across the middle, and that stopped
    # working the moment the camera panel moved into the top-left corner:
    # the two wanted the same rows, and the band cut the stage in half
    # exactly where the interesting part of it is. Stood up on the right it
    # leaves the whole left third clear -- camera panel above, and below it
    # the avatar frozen against whatever just killed it, which on a death
    # screen is the one thing actually worth looking at.
    #
    # The mugshot still goes inside the panel rather than near the avatar,
    # and for the original reason: the obstacle that killed you is already
    # drawn at the far left, and repeating the same picture a few hundred
    # pixels away reads as a rendering bug.
    PANEL_X = 0.34           # of frame width; the panel runs from here to the edge
    TEXT_X = 0.37            # of frame width
    TEXT_W = 0.60            # of frame width
    LOGO_CY, LOGO_H = 0.70, 0.28   # of frame height
    TITLE_CY, TITLE_PX = 0.215, 0.115
    SUB_Y, SUB_PX = 0.315, 0.040
    SCORE_Y, SCORE_PX = 0.395, 0.048
    PROMPT_Y, PROMPT_PX = 0.900, 0.045

    # Animation, in seconds since death.
    SLAM, SETTLE = 0.13, 0.09
    SUB_AT, SUB_WIPE, SCORE_AT = 0.28, 0.22, 0.50

    # Death repaints the whole stage, so both passes go through lookup
    # tables: same reason the gameplay dim does, and the world table folds
    # the red cast into the same pass by giving each channel its own curve.
    WORLD_DIM, WORLD_RED, BAND_DIM = 0.62, 14, 0.45

    def __init__(self, path=TAUNTS_PATH):
        self.type = Type()
        ramp = np.arange(256)
        self._world_lut = np.zeros((1, 256, 3), np.uint8)
        self._world_lut[0, :, 0] = np.clip(ramp * self.WORLD_DIM, 0, 255)
        self._world_lut[0, :, 1] = self._world_lut[0, :, 0]
        self._world_lut[0, :, 2] = np.clip(ramp * self.WORLD_DIM + self.WORLD_RED, 0, 255)
        # The band's table has the world's already composed into it, so the
        # rows behind the panel are dimmed once rather than twice. Two passes
        # over a 1707x960 frame is 0.75 ms that buys nothing.
        self._band_lut = np.clip(
            self._world_lut.astype(np.float32) * self.BAND_DIM, 0, 255).astype(np.uint8)
        self.taunts = self._load(path)
        self._key = None
        self._parts = {}
        self._prompt_mask = None
        self._prompt_for = None

    @staticmethod
    def _load(path):
        """Read taunts.json, or fall back to one hardcoded line.

        Never fatal: a missing or malformed file costs you the jokes, not the
        game, and the file is the part most likely to be hand-edited.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            out = [e for e in data if isinstance(e, dict) and e.get("title")]
            if out:
                return out
        except (OSError, ValueError) as exc:
            print(f"taunts: {exc}; falling back to a plain screen", flush=True)
        return [{"image": None, "title": "GAME OVER", "text": ""}]

    def pick(self, art_name, rng):
        """A taunt for whatever killed you, chosen through `rng`.

        Entries with no image are the fallback pool, used when the obstacle
        had no artwork or when nobody wrote a line for it yet.
        """
        matches = [e for e in self.taunts if e.get("image") == art_name] if art_name else []
        if not matches:
            matches = [e for e in self.taunts if e.get("image") is None] or self.taunts
        return rng.choice(matches)

    PULSE_STEPS = 6

    def _part(self, name, factory):
        """Rasterise `name` the first time it is actually wanted.

        Deliberately lazy. Building the whole panel in one go cost 70 ms --
        four dropped frames at exactly the moment of death, which is the one
        moment the screen is trying to feel like an impact. The pieces are
        not needed together anyway: the title is due immediately, the flavour
        line at 0.28 s, the score at 0.5 s and the prompt not for 1.6 s, so
        each one is paid for on a different frame and none of them shows.
        """
        got = self._parts.get(name)
        if got is None:
            got = self._parts[name] = factory()
        return got

    def _title(self):
        w, h, taunt = self._w, self._h, self._taunt
        m = self.type.fitted(taunt["title"], self.TITLE_PX * h, int(self.TEXT_W * w),
                             track=0.06, weight_frac=0.045)
        return Stamp(m, C_TITLE, C_TITLE_SHADOW, max(2, round(self.TITLE_PX * h * 0.07)))

    def _subs(self):
        text = self._taunt.get("text", "")
        if not text:
            return []
        ms = self.type.wrapped(text, self.SUB_PX * self._h,
                               int(self.TEXT_W * self._w), track=0.01)
        return [Stamp(m, C_SUB) for m in ms]

    def _score(self):
        m = self.type.fitted(f"SCORE {self._cur_score}     BEST {self._cur_best}",
                             self.SCORE_PX * self._h, int(self.TEXT_W * self._w),
                             track=0.10, weight_frac=0.03)
        return Stamp(m, C_SCORE, C_SHADOW, 3)

    def _pulses(self):
        """Quantised brightnesses, so the pulse is a lookup not a rasterise.

        The mask itself is cached across deaths -- the words never change,
        only the frame size can -- so only the very first death pays for it.
        """
        if self._prompt_mask is None or self._prompt_for != (self._w, self._h):
            self._prompt_mask = self.type.fitted(
                "JUMP TO PLAY AGAIN", self.PROMPT_PX * self._h,
                int(self._w * 0.9), track=0.16, weight_frac=0.03)
            self._prompt_for = (self._w, self._h)
        n = self.PULSE_STEPS
        return [Stamp(self._prompt_mask,
                      tuple(int(c * (0.62 + 0.38 * i / (n - 1))) for c in C_TITLE),
                      C_SHADOW, 3) for i in range(n)]

    def draw(self, frame, taunt, art, score, best, t, lockout):
        """Paint the whole screen. `t` is seconds since the collision."""
        h, w = frame.shape[:2]
        key = (id(taunt), score, best, w, h)
        if key != self._key:
            self._key, self._parts = key, {}
            self._w, self._h, self._taunt = w, h, taunt
            self._cur_score, self._cur_best = score, best
        tx = int(self.TEXT_X * w)

        # 1. Knock the world back further than gameplay already does, and
        #    warm it towards red, so the panel unambiguously owns the screen.
        #    The band, whose job is to give the text one predictable
        #    background because everything behind it is a live room, is the
        #    same pass with a darker table -- so every pixel is touched once.
        x0 = int(self.PANEL_X * w)
        for cols, lut in ((frame[:, :x0], self._world_lut),
                          (frame[:, x0:], self._band_lut)):
            if cols.size:
                cv2.LUT(cols, lut, dst=cols)
        cv2.line(frame, (x0, 0), (x0, h), C_RULE, 3, cv2.LINE_AA)

        # 3. Whatever killed you, large. The point of the screen.
        if art is not None:
            art.draw(frame, (x0 + w) // 2, int(self.LOGO_CY * h),
                     int(self.LOGO_H * h))

        # 4. Title: slams in oversized, overshoots, settles.
        if t < self.SLAM:
            k = 1.0 + 1.15 * (1 - t / self.SLAM) ** 2
        elif t < self.SLAM + self.SETTLE:
            k = 1.0 - 0.06 * (1 - (t - self.SLAM) / self.SETTLE)
        else:
            k = 1.0
        title = self._part("title", self._title)
        # Grow about the title's own centre, not its left edge: anchored
        # left it lunges off the right of the screen and the first frames
        # of the slam show only the first half of the word.
        title.draw(frame, tx + int(title.w * (1 - k) / 2), int(self.TITLE_CY * h), k)

        # Laid out whether or not it is being drawn yet: the score below it
        # has to sit in the same place from the first frame to the last, and
        # the flavour line only animates its clip, never its geometry.
        if t > self.SUB_AT:
            p = min(1.0, (t - self.SUB_AT) / self.SUB_WIPE)
            y = self.SUB_Y * h
            for s in self._part("subs", self._subs):
                s.draw(frame, tx + 4, int(y + s.h / 2), clip=int(s.w * p))
                y += s.h * 1.35

        if t > self.SCORE_AT:
            sc = self._part("score", self._score)
            # SCORE_Y is where the score goes when the flavour line is one
            # line, which every taunt was until "I am because we are..."
            # wrapped to two and landed 3 px inside the score. Each *extra*
            # line pushes the score down by exactly that line's advance.
            #
            # By its own advance rather than by a measured gap under the ink:
            # mask heights vary with ascenders and descenders, so measuring
            # would shift a one-line taunt a few pixels depending on whether
            # it happens to contain a "y".
            #
            # Reading "subs" here is free. SCORE_AT is after SUB_AT, so the
            # flavour line was already rasterised on an earlier frame -- which
            # is the whole point of _part() and why this must not be hoisted
            # out of the `if`.
            extra = sum(s.h * 1.35 for s in self._part("subs", self._subs)[1:])
            sc.draw(frame, tx + 4, int(self.SCORE_Y * h + extra + sc.h / 2))

        if t >= lockout:
            pulses = self._part("pulses", self._pulses)
            n = len(pulses)
            pulse = pulses[min(n - 1, int(abs(np.sin(t * 3.0)) * (n - 0.5)))]
            pulse.draw(frame, x0 + (w - x0 - pulse.w) // 2, int(self.PROMPT_Y * h))
