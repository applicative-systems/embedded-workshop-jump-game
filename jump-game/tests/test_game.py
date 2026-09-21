"""The game rules, hand-stepped. No model, no camera, no window.

`JumpGame.update()` is deterministic given an injected rng, which is what the
module docstring in src/game.py promises and what nothing until now checked.
Obstacles are constructed directly rather than drawn from the spawner, so a
collision test pins the collision and nothing else.

The last group are arithmetic regressions rather than behaviour: they encode
the fairness sums the constants were chosen by, so that retuning SPEED0,
HIT_W_H or TOP_EXTENT without redoing those sums fails loudly here instead of
quietly in someone's face.
"""

import random

import pytest

import coco
from game import JumpGame, Obstacle, Player, StickFigure

ASPECT = 16 / 9
DT = 1 / 60


def game(seed=1, **kw):
    g = JumpGame(rng=random.Random(seed), aspect=ASPECT, **kw)
    g.start()
    g.next_gap = 1e9  # keep the spawner out of the way
    return g


def g_art():
    """The sprite pool, or empty when images/ is not in the checkout."""
    return JumpGame(rng=random.Random(0), aspect=ASPECT).art


def hitbox(g):
    pad = (g.pw - g.hit_w) / 2
    return Player.X + pad, Player.X + g.pw - pad


def ground_obstacle(g, contact_in, w=0.09, h=0.28):
    """A block that first touches the player `contact_in` seconds from now."""
    _px0, px1 = hitbox(g)
    return Obstacle(x=px1 + g.speed * contact_in, w=w, h=h, art_h=h)


def ceiling_bar(g, contact_in, w=0.09, art_h=0.20):
    _px0, px1 = hitbox(g)
    return Obstacle(x=px1 + g.speed * contact_in, w=w, y=g.BAR_GAP,
                    h=g.ground - g.BAR_GAP + 0.04, art_h=art_h)


def step(g, seconds, jump=None, ducking=False, present=True):
    """Run `seconds` of game time, firing `jump` on the first frame only."""
    for _ in range(round(seconds / DT)):
        if g.state == JumpGame.OVER:
            return
        g.update(DT, jump, present, ducking=ducking)
        jump = None


# -- ground obstacles still behave exactly as they did ---------------------

def test_a_ground_obstacle_still_needs_a_jump():
    g = game()
    g.obstacles = [ground_obstacle(g, 0.5)]
    step(g, 1.5)
    assert g.state == JumpGame.OVER
    assert g.score == 0


def test_a_well_timed_jump_clears_a_ground_obstacle():
    g = game()
    g.obstacles = [ground_obstacle(g, 0.5)]
    step(g, 0.4)
    step(g, 1.2, jump=Player.LIFT_MAX)
    assert g.state == JumpGame.PLAYING
    assert g.score == 1


def test_crouching_does_not_help_against_a_ground_obstacle():
    g = game()
    g.obstacles = [ground_obstacle(g, 0.5)]
    step(g, 1.5, ducking=True)
    assert g.state == JumpGame.OVER


def test_the_new_collision_test_is_the_old_one_for_ground_obstacles():
    """The AABB must not have changed what a ground obstacle does.

    A full replay of the recorded session cannot check this -- obstacle
    spawning is random and the trace does not carry the seed, so a replay
    meets different obstacles. What is checkable, and is the actual claim, is
    that for y = 0 the interval test reduces to the scalar height test it
    replaced, for every lift, every obstacle height and both player heights.
    """
    g = game()
    for crouch_k in (0.0, 0.5, 1.0):
        g.crouch_k = crouch_k
        for lift in [i / 200 for i in range(int(0.7 * 200))]:
            g.player.lift = lift
            for h in [i / 200 for i in range(int(0.35 * 200))]:
                o = Obstacle(x=0.0, w=0.09, h=h)
                assert g._hits(o) == (lift < h - g.FORGIVE), (crouch_k, lift, h)


# -- ceiling bars ----------------------------------------------------------

def test_a_crouch_clears_a_bar():
    g = game()
    g.obstacles = [ceiling_bar(g, 0.5)]
    step(g, 1.5, ducking=True)
    assert g.state == JumpGame.PLAYING
    assert g.score == 1
    assert g.ducks_cleared == 1


def test_standing_into_a_bar_dies():
    g = game()
    g.obstacles = [ceiling_bar(g, 0.5)]
    step(g, 1.5)
    assert g.state == JumpGame.OVER
    assert g.taunt is not None


def test_a_death_is_taunted_by_whatever_hit_you():
    """Keyed on the artwork, whichever way the obstacle was arriving.

    Both kinds are drawn from the same sprites, so a bar earns the same
    jokes a ground obstacle does -- which only works because no taunt names
    a verb.
    """
    if not g_art():
        pytest.skip("images/ not present")
    for ceiling in (False, True):
        g = game()
        o = ceiling_bar(g, 0.5) if ceiling else ground_obstacle(g, 0.5)
        o.art = g.art[0]
        g.obstacles = [o]
        step(g, 1.5)
        assert g.state == JumpGame.OVER
        assert g.killer is o.art
        assert g.taunt["image"] == o.art.name


def test_no_taunt_names_a_verb():
    """The obstacle mix is even and either verb can kill you, so a taunt
    that says "you didn't jump" is wrong half the time."""
    g = game()
    for t in g.over.taunts:
        words = (t["title"] + " " + t["text"]).lower()
        for verb in ("jump", "duck", "ceiling"):
            assert verb not in words, (verb, t["title"])


def test_jumping_into_a_bar_dies():
    g = game()
    g.obstacles = [ceiling_bar(g, 0.5)]
    step(g, 0.4)
    step(g, 1.2, jump=Player.LIFT_MAX)
    assert g.state == JumpGame.OVER


def test_a_duck_is_ignored_in_the_air():
    g = game()
    g.update(DT, Player.LIFT_MAX, True)
    assert g.player.airborne
    for _ in range(20):
        g.update(DT, None, True, ducking=True)
        assert not g.crouched


def test_a_jump_beats_a_duck_on_the_same_frame():
    g = game()
    step(g, 0.4, ducking=True)
    assert g.crouched
    g.update(DT, Player.LIFT_MAX, True, ducking=True)
    assert g.player.airborne
    assert not g.crouched
    assert g.crouch_k == 0.0


# -- the hold --------------------------------------------------------------

def test_the_minimum_hold_covers_a_detector_dropout():
    """Two frames of `ducking`, then silence, still clears the bar."""
    g = game()
    g.obstacles = [ceiling_bar(g, 2 * DT + 0.05, w=JumpGame.OBST_WMAX / ASPECT)]
    step(g, 2 * DT, ducking=True)
    assert g.crouched
    step(g, 1.5)
    assert g.state == JumpGame.PLAYING
    assert g.score == 1


def test_the_hold_is_a_floor_and_not_a_latch():
    g = game()
    g.obstacles = [ceiling_bar(g, 0.9)]
    step(g, 2 * DT, ducking=True)
    step(g, 1.5)
    assert g.state == JumpGame.OVER


def test_a_short_dropout_is_debounced_and_a_long_one_is_not():
    def run(dropout):
        g = game()
        g.obstacles = [ceiling_bar(g, 0.5)]
        step(g, 0.45, ducking=True)
        step(g, dropout)
        step(g, 1.0, ducking=True)
        return g.state

    assert run(0.10) == JumpGame.PLAYING
    assert run(0.20) == JumpGame.OVER


def test_an_absent_player_holds_the_crouch():
    """Stepping out of shot is not evidence that you stood up."""
    g = game()
    step(g, 0.5, ducking=True)
    assert g.crouched
    for _ in range(60):
        g.update(DT, None, False, ducking=False)
    assert g.crouched


def test_the_crouch_is_practisable_in_the_lobby():
    g = JumpGame(rng=random.Random(1), aspect=ASPECT)
    assert g.state == JumpGame.WAITING
    for _ in range(30):
        g.update(DT, None, True, ducking=True)
    assert g.crouched


# -- the spawn mix ---------------------------------------------------------

def test_the_mix_is_even_at_every_score():
    g = game()
    for score in (0, 1, 5, 40):
        g.score = score
        rate = sum(g._next_kind(False) for _ in range(20000)) / 20000
        assert rate == pytest.approx(0.5, abs=0.02), score


def test_the_mix_does_not_depend_on_the_previous_obstacle():
    """Streaks are allowed, and that is what makes an even mix possible.

    A no-repeats rule caps bars at 50% only by strict alternation, which
    telegraphs every obstacle.
    """
    g = game()
    after_bar = sum(g._next_kind(True) for _ in range(20000)) / 20000
    after_ground = sum(g._next_kind(False) for _ in range(20000)) / 20000
    assert after_bar == pytest.approx(0.5, abs=0.02)
    assert after_ground == pytest.approx(0.5, abs=0.02)


def test_the_first_obstacle_of_a_run_is_always_a_ground_one():
    """The jump that starts the game is followed by something to jump over."""
    for seed in range(50):
        g = game(seed=seed)
        assert not g.next_is_bar
        g.next_gap = 0.0
        g._advance_obstacles(DT)
        assert not g.obstacles[0].ceiling


def test_no_bars_at_all_with_bars_off():
    g = game(bars=False)
    g.score = 50
    assert not any(g._next_kind(False) for _ in range(2000))
    assert not any(g._next_kind(True) for _ in range(2000))


def spawn_sequence(frames=120000, seed=1):
    """(time, is_bar) for every obstacle a long run would spawn."""
    g = game(seed=seed)
    g.next_gap = 0.0
    t, spawns = 0.0, []
    for i in range(frames):
        g.score = min(40, i // 300)
        before = len(g.obstacles)
        g._advance_obstacles(DT)
        if len(g.obstacles) > before:
            spawns.append((t, g.obstacles[-1].ceiling))
        g.obstacles.clear()  # nothing ever reaches the player
        t += DT
    return spawns


def test_standing_up_into_a_jump_always_gets_the_wider_gap():
    """The one transition GAP_FLOOR does not cover: crouch -> jump."""
    spawns = spawn_sequence()
    checked = 0
    for i in range(len(spawns) - 1):
        if spawns[i][1] and not spawns[i + 1][1]:
            checked += 1
            gap = spawns[i + 1][0] - spawns[i][0]
            assert gap >= JumpGame.BAR_SPACING - 1e-9, gap
    assert checked > 100, "too few crouch -> jump transitions to test"


def test_the_other_transitions_are_not_needlessly_widened():
    """GAP_TIGHTEN must still bite; at an even mix it would not if every gap
    next to a bar were widened."""
    spawns = spawn_sequence()
    gaps = [spawns[i + 1][0] - spawns[i][0] for i in range(len(spawns) - 1)
            if not (spawns[i][1] and not spawns[i + 1][1])]
    assert min(gaps) < JumpGame.BAR_SPACING - 0.05


def test_the_long_run_mix_is_even():
    spawns = spawn_sequence()
    bars = sum(1 for _t, c in spawns if c)
    assert bars / len(spawns) == pytest.approx(0.5, abs=0.03)


# -- the arithmetic the constants were chosen by ---------------------------

def test_a_crouch_fits_under_a_bar_and_standing_does_not():
    g = game()
    assert JumpGame.H_CROUCH <= g.BAR_GAP + g.FORGIVE < JumpGame.H_STAND
    crouch_margin = g.BAR_GAP + g.FORGIVE - JumpGame.H_CROUCH
    kill_margin = JumpGame.H_STAND - g.BAR_GAP - g.FORGIVE
    assert crouch_margin >= 0.04, crouch_margin
    assert kill_margin >= 0.04, kill_margin


@pytest.mark.parametrize("aspect, floor", [(16 / 9, 1.6), (4 / 3, 1.25)])
def test_the_hold_outlasts_the_widest_obstacle_at_the_opening_speed(aspect, floor):
    """The jump arc keeps a 1.64x margin over its worst crossing; match it.

    OBST_WMAX, not some narrower bar-specific width: a ceiling bar is drawn
    from the same sprites and sized by the same rule as a ground obstacle, so
    it can be as wide as any of them. 4:3 (`--stage off`) is the looser case,
    since a width quoted in frame heights is a larger fraction of a narrower
    stage and therefore crosses slower.
    """
    hit_w = Player.HIT_W_H / aspect
    crossing = (JumpGame.OBST_WMAX / aspect + hit_w) / JumpGame.SPEED0
    assert JumpGame.DUCK_MIN / crossing >= floor, crossing


def test_peak_max_is_unmoved_by_the_crouch():
    g = JumpGame(ground=0.96, aspect=ASPECT)
    assert g.peak_max == pytest.approx(Player.PEAK_MAX)
    # and the arc still leaves a standing head in frame
    assert g.peak_max + JumpGame.H_STAND < g.ground


def test_the_drawn_height_is_the_lethal_height():
    """The invariant: one number feeds both the figure and the collision."""
    g = game()
    for k in (0.0, 0.34, 1.0):
        g.crouch_k = k
        extent = g.player_h / StickFigure.HEIGHT
        assert extent * StickFigure.HEIGHT == pytest.approx(g.player_h)
    g.crouch_k = 0.0
    assert g.player_h == pytest.approx(JumpGame.H_STAND)
    g.crouch_k = 1.0
    assert g.player_h == pytest.approx(JumpGame.H_CROUCH)


# -- both kinds are the same picture --------------------------------------

def test_both_kinds_are_sized_by_the_same_rule():
    """Only the height off the ground differs, which is the whole design."""
    g = game()
    for seed in range(200):
        g.rng = random.Random(seed)
        ground = g._spawn(ceiling=False)
        g.rng = random.Random(seed)
        bar = g._spawn(ceiling=True)
        assert bar.art is ground.art
        assert bar.w == pytest.approx(ground.w)
        assert bar.art_h == pytest.approx(ground.art_h)
        assert ground.y == 0.0 and bar.y == g.BAR_GAP


def test_a_bar_runs_off_the_top_of_the_frame():
    """Its box must cover every height a jump can reach, or the drawing lies."""
    g = game()
    bar = g._spawn(ceiling=True)
    assert bar.y + bar.h > g.peak_max + JumpGame.H_STAND
    assert bar.art_h < bar.h  # there is wall above the picture


def test_a_ground_obstacle_is_all_picture():
    g = game()
    o = g._spawn(ceiling=False)
    assert o.art_h == pytest.approx(o.h)


# -- the death screen is a photograph, not a mirror ------------------------

def fake_pose(bend):
    """One person; `bend` pixels of stoop, lowering the head and shoulders.

    A *shape* change, not a translation. The figure normalises to the hip
    origin and divides by its own rolling scale, so shifting every keypoint
    and the box by the same amount produces a byte-identical pose -- which is
    correct behaviour and useless for telling live from frozen.
    """
    kp = [[500.0, 500.0, 0.9] for _ in range(coco.N)]
    for j, y in ((coco.NOSE, 200.0 + bend),
                 (coco.LEFT_SHOULDER, 300.0 + bend), (coco.RIGHT_SHOULDER, 300.0 + bend),
                 (coco.LEFT_ELBOW, 420.0 + bend), (coco.RIGHT_ELBOW, 420.0 + bend),
                 (coco.LEFT_HIP, 520.0), (coco.RIGHT_HIP, 520.0),
                 (coco.LEFT_KNEE, 700.0), (coco.RIGHT_KNEE, 700.0),
                 (coco.LEFT_ANKLE, 880.0), (coco.RIGHT_ANKLE, 880.0)):
        kp[j] = [500.0, y, 0.9]

    class P:
        xy = [[(k[0], k[1]) for k in kp]]
        kp_conf = [[k[2] for k in kp]]
        box_xyxy = [[440.0, 150.0 + bend, 560.0, 900.0]]
        box_conf = [0.9]

    return P()


def test_the_pose_freezes_at_the_collision():
    """The player walks off and waves; the avatar stays where it died."""
    g = game()
    for _ in range(20):
        g.set_pose(fake_pose(0.0), 0)
    g.obstacles = [ground_obstacle(g, 0.2)]
    step(g, 1.0)
    assert g.state == JumpGame.OVER

    frozen = [tuple(p) for p in g.figure.pts]
    for _ in range(60):
        g.set_pose(fake_pose(160.0), 0)
        g.update(DT, None, True)
    assert [tuple(p) for p in g.figure.pts] == frozen


def test_the_arc_freezes_at_the_collision():
    """Dying in mid-air leaves the avatar in mid-air."""
    g = game()
    g.obstacles = [ceiling_bar(g, 0.5)]
    step(g, 0.4)
    step(g, 1.2, jump=Player.LIFT_MAX)
    assert g.state == JumpGame.OVER
    assert g.player.lift > 0.0, "should have died in the air"
    lift = g.player.lift
    step(g, 1.0)
    assert g.player.lift == lift


def test_the_pose_resumes_on_the_next_run():
    g = game()
    for _ in range(20):
        g.set_pose(fake_pose(0.0), 0)
    g.obstacles = [ground_obstacle(g, 0.2)]
    step(g, 1.0)
    assert g.state == JumpGame.OVER
    # Not step(), which bails out the moment the run is over -- the lockout
    # has to be waited out inside the OVER state.
    for _ in range(int((JumpGame.OVER_LOCKOUT + 0.1) / DT)):
        g.update(DT, None, True)
    g.update(DT, Player.LIFT_MAX, True)
    assert g.state == JumpGame.PLAYING
    frozen = [tuple(p) for p in g.figure.pts]
    g.set_pose(fake_pose(160.0), 0)
    assert [tuple(p) for p in g.figure.pts] != frozen


# -- the fixed timestep ----------------------------------------------------

def narrowest_width(aspect=ASPECT):
    """The thinnest obstacle _spawn can build, in stage widths."""
    g = JumpGame(rng=random.Random(0), aspect=aspect)
    if not g.art:
        return JumpGame.OBST_WMAX * 0.58 / aspect
    best = None
    for art in g.art:
        hi = min(JumpGame.OBST_H[1], JumpGame.OBST_WMAX / art.aspect)
        lo = hi * JumpGame.OBST_H[0] / JumpGame.OBST_H[1]
        w = lo * art.aspect / aspect
        best = w if best is None else min(best, w)
    return best


def test_the_substep_cannot_tunnel_the_narrowest_obstacle():
    """The collision test is a discrete overlap check, so this is real.

    The old `min(dt, 0.05)` clamp cleared the narrowest window by 1.57x and
    only by luck -- one skinnier logo in ART and it would have started letting
    obstacles through at top speed.
    """
    window = narrowest_width() + Player.HIT_W_H / ASPECT
    travel = JumpGame.SPEED_MAX * JumpGame.STEP
    assert window / travel >= 4, (window, travel)


def test_a_long_frame_still_kills_you():
    """A stalled frame must not walk an obstacle straight through the player."""
    g = game()
    g.player.lift = 0.0
    g.obstacles = [ground_obstacle(g, 0.0, w=narrowest_width(), h=0.28)]
    g.speed = JumpGame.SPEED_MAX
    g.update(0.2, None, True)
    assert g.state == JumpGame.OVER


def test_no_simulated_time_is_lost():
    """The clamp used to swallow 8-30% of wall clock; the accumulator carries
    the remainder instead."""
    g = game()
    g.obstacles = []
    g.next_gap = 1e9
    dt, n = 0.0716, 100          # 14 fps, the rate the clamp hurt most
    # Far enough away that it never reaches the player: a death stops the
    # substep loop, which would measure the distance to the hitbox instead of
    # the distance travelled.
    o = ground_obstacle(g, 50.0)
    g.obstacles = [o]
    x0 = o.x
    for _ in range(n):
        g.update(dt, None, True)
    moved = x0 - o.x
    assert moved == pytest.approx(n * dt * g.speed, abs=g.speed * JumpGame.STEP)


def test_time_beyond_max_catchup_is_discarded_not_banked():
    """A two-second stall must not then sprint the world through the player."""
    g = game()
    g.obstacles = []
    g.next_gap = 1e9
    o = ground_obstacle(g, 50.0)
    g.obstacles = [o]
    x0 = o.x
    g.update(2.0, None, True)
    assert x0 - o.x == pytest.approx(JumpGame.MAX_CATCHUP * g.speed,
                                     abs=g.speed * JumpGame.STEP)
    assert g._accum < JumpGame.STEP     # no debt carried forward


def test_a_takeoff_on_a_tick_too_short_to_step_is_not_lost():
    """The case a naive accumulator drops on the floor."""
    g = game()
    g._accum = 0.0
    g.update(JumpGame.STEP / 4, Player.LIFT_MAX, True)
    assert not g.player.airborne, "no substep should have run yet"
    g.update(JumpGame.STEP, None, True)
    assert g.player.airborne, "the held takeoff must fire on the next substep"


def test_a_held_duck_integrates_the_same_at_any_frame_rate():
    """Levels are sampled, not counted -- which is what lets the render loop
    run at 60 Hz on poses arriving at 30."""
    def crouch_after(dt, seconds):
        g = game()
        for _ in range(round(seconds / dt)):
            g.update(dt, None, True, ducking=True)
        return g.crouched, round(g.crouch_k, 3)

    assert crouch_after(1 / 16, 0.5) == crouch_after(1 / 120, 0.5)
