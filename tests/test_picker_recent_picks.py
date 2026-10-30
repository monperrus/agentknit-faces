"""The route picker lists the five most-picked routes of the last 7 days first.

The picker used to look identical on every start: gateway first, then the
whole catalog alphabetically, so the three or four routes actually used were
buried in a wall of ids. It now remembers picks (one timestamp each, in
machine-local state) and promotes the five most-picked routes of the last
week; everything else keeps its place.

What has to hold: the window is 7 days (an older pick does not count), at
most five routes are promoted, ties go to the route used most recently, a
promoted route stays reachable even if a newer version shadows it, and a
state file that is missing, corrupt, or not writable never breaks picking.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTKNIT = Path.home() / "workspace" / "prototypes" / "agentknit"
if str(AGENTKNIT) not in sys.path:
    sys.path.insert(0, str(AGENTKNIT))

DAY = 86400.0
NOW = 1_800_000_000.0  # fixed clock, so window edges are exact


@pytest.fixture(scope="module")
def face():
    path = REPO_ROOT / "agent-kimi-k3-tui-superlean.py"
    spec = importlib.util.spec_from_file_location("_face_picks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the developer's real pick history."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


def models(*names: str) -> list[dict]:
    return [{"kind": "model", "entry_id": "e", "model": name} for name in names]


def key_of(face, route: dict) -> str:
    return face._route_key(route)


def rank_names(face, routes: list[dict], picks: dict[str, list[float]]) -> list[str]:
    rank = face._recent_pick_rank(routes, picks, now=NOW)
    by_position = sorted(rank, key=rank.get)
    return [position.split(":")[-1] for position in by_position]


def test_most_picked_first_and_the_rest_in_catalog_order(face):
    routes = models("a", "b", "c", "d", "e", "f", "g")
    picks = {
        key_of(face, routes[2]): [NOW - DAY] * 3,  # c: 3 picks
        key_of(face, routes[5]): [NOW - 2 * DAY] * 2,  # f: 2 picks
        key_of(face, routes[0]): [NOW - 3 * DAY],  # a: 1 pick
    }
    assert rank_names(face, routes, picks) == ["c", "f", "a"]
    order = face._picker_order(routes, list(range(len(routes))),
                               face._recent_pick_rank(routes, picks, now=NOW))
    assert order == [2, 5, 0, 1, 3, 4, 6]


def test_picks_older_than_a_week_do_not_count(face):
    routes = models("old", "fresh")
    picks = {
        key_of(face, routes[0]): [NOW - 8 * DAY] * 9,
        key_of(face, routes[1]): [NOW - 6 * DAY],
    }
    assert rank_names(face, routes, picks) == ["fresh"]


def test_window_boundary_is_seven_days(face):
    routes = models("edge", "just-out")
    picks = {
        key_of(face, routes[0]): [NOW - 7 * DAY],  # inside
        key_of(face, routes[1]): [NOW - 7 * DAY - 1],  # one second outside
    }
    assert rank_names(face, routes, picks) == ["edge"]


def test_ties_go_to_the_route_picked_most_recently(face):
    routes = models("older", "newer")
    picks = {
        key_of(face, routes[0]): [NOW - 5 * DAY],
        key_of(face, routes[1]): [NOW - 1 * DAY],
    }
    assert rank_names(face, routes, picks) == ["newer", "older"]


def test_at_most_five_routes_are_promoted(face):
    routes = models(*"abcdefg")
    picks = {key_of(face, route): [NOW - DAY] * count
             for count, route in enumerate(routes, start=1)}
    rank = face._recent_pick_rank(routes, picks, now=NOW)
    assert len(rank) == face.PICK_TOP == 5
    # the two least-picked routes stay in catalog order at the end
    order = face._picker_order(routes, list(range(len(routes))), rank)
    assert order == [6, 5, 4, 3, 2, 0, 1]


def test_history_of_routes_not_in_the_catalog_does_not_take_a_slot(face):
    routes = models(*"abcdef")
    picks = {"model:gone:retired": [NOW - DAY] * 99}
    picks.update({key_of(face, route): [NOW - DAY] * count
                  for count, route in enumerate(routes, start=1)})
    rank = face._recent_pick_rank(routes, picks, now=NOW)
    assert "model:gone:retired" not in rank
    assert len(rank) == 5


def test_record_and_load_round_trip(face, state_home: Path):
    route = {"kind": "superleanai"}
    face._record_pick(route, now=NOW - DAY)
    face._record_pick(route, now=NOW)
    assert face._load_picks(now=NOW) == {"superleanai": [NOW - DAY, NOW]}
    stored = json.loads((state_home / "agentknit-faces" / "route-picks.json").read_text())
    assert stored == {"superleanai": [NOW - DAY, NOW]}


def test_record_drops_history_older_than_the_retention(face, state_home: Path):
    route = {"kind": "superleanai"}
    face._record_pick(route, now=NOW - 40 * DAY)
    face._record_pick(route, now=NOW)
    assert face._load_picks(now=NOW) == {"superleanai": [NOW]}


def test_missing_or_corrupt_history_reads_as_empty(face, state_home: Path):
    assert face._load_picks(now=NOW) == {}
    path = face._picks_path()
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    assert face._load_picks(now=NOW) == {}
    path.write_text(json.dumps(["wrong", "shape"]))
    assert face._load_picks(now=NOW) == {}


def test_unwritable_state_dir_never_breaks_picking(face, state_home: Path):
    state_home.mkdir(parents=True)
    state_home.chmod(0o500)
    try:
        face._record_pick({"kind": "superleanai"}, now=NOW)  # must not raise
    finally:
        state_home.chmod(0o700)


def _render(face, routes: list[dict]) -> list[str]:
    """Open the picker headless and return the visible labels, top to bottom."""

    async def scenario() -> list[str]:
        from textual.app import App

        app = App()
        async with app.run_test() as pilot:
            screen = face.ProviderPickerScreen(routes)
            await app.push_screen(screen)
            await pilot.pause()
            return [str(item.query_one(face.Label).render())
                    for item in screen.query_one(face.ListView).children]

    return asyncio.run(scenario())


def test_screen_puts_recent_picks_first_with_a_star(face):
    routes = models("a", "b", "c", "d")
    now = time.time()
    for stamp in (now - 2 * DAY, now - DAY, now):
        face._record_pick(routes[2], now=stamp)
    face._record_pick(routes[1], now=now - DAY)
    assert _render(face, routes) == ["★ e · c", "★ e · b", "e · a", "e · d"]


def test_screen_promotes_a_picked_route_that_is_shadowed(face):
    """Picking an older id keeps it visible; unpicked shadowed ids stay hidden."""
    picked = dict(models("old")[0], shadowed=True)
    unpicked = dict(models("older")[0], shadowed=True)
    routes = [picked, unpicked, *models("new", "newer")]
    face._record_pick(picked, now=time.time())
    labels = _render(face, routes)
    assert labels == ["★ e · old", "e · new", "e · newer"]
