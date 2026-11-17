# [AI-SCAFFOLD] Pipeline fixes implemented with AI assistance; include in course disclosure.
# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
import pandas as pd

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
DATA_DIR = _section_01.DATA_DIR
SEP_COL = _section_01.SEP_COL

# %% [markdown]
# ## 2. Load the data (WR-only files)
#
# Everything downstream uses WR-only tables, so the big competition CSVs are needed only once. The cell below **builds** `wr_only_players.csv`, `wr_only_results.csv`, `wr_only_trk.csv` and `wr_only_play.csv` from the full files if they do not exist yet, and otherwise just loads them. Delete the WR-only files to force a rebuild.
#
# Notes:
# - WRs are defined by `nfl_position == "WR"` in `players.csv`. About 106 of them have Combine tracking.
# - Combine tracking has `entity_type` `PLAYER` and `BALL` rows. Ball rows carry the player's `nfl_id` (they survive the WR filter), so ball frames can be joined on `(event_id, nfl_id)`. They cover only about a third of attempts, so v1 does not use them.
# - `time` is parsed to a datetime once, here.
# - **Do not load the `game_tracking_*.csv` files.** They are huge, and you do not need them: separation is already in `player_play.csv`.

# %%
WR_FILES = {k: DATA_DIR / f"wr_only_{k}.csv" for k in ["players", "results", "trk", "play"]}

def build_wr_files():
    """One-time: needs the FULL competition CSVs in DATA_DIR. Skipped when the WR-only files already exist."""
    players_full = pd.read_csv(DATA_DIR / "players.csv")
    ids = set(players_full.loc[players_full["nfl_position"] == "WR", "nfl_id"])
    sources = {"players": "players.csv", "results": "combine_results.csv",
               "trk": "combine_tracking.csv", "play": "player_play.csv"}
    for key, fname in sources.items():
        df = pd.read_csv(DATA_DIR / fname)
        df[df["nfl_id"].isin(ids)].to_csv(WR_FILES[key], index=False)
        print("wrote", WR_FILES[key])

if not all(p.exists() for p in WR_FILES.values()):
    build_wr_files()

wr_only_players = pd.read_csv(WR_FILES["players"])
wr_only_results = pd.read_csv(WR_FILES["results"])
wr_only_trk     = pd.read_csv(WR_FILES["trk"], parse_dates=["time"])
wr_only_play    = pd.read_csv(WR_FILES["play"], low_memory=False)

# Later sections (5-14) use these shorter names for the same WR-only tables.
players, results, play = wr_only_players, wr_only_results, wr_only_play
wr_ids = set(wr_only_players["nfl_id"])

# %%
# Sanity checks: fix any VERIFY item here before going further.
for name, df in [("players", wr_only_players), ("results", wr_only_results),
                 ("trk", wr_only_trk), ("play", wr_only_play)]:
    print(f"{name:8s} {df.shape}")

print("\nWRs in players:", len(wr_ids))
print("WRs in results:", wr_only_results["nfl_id"].nunique())
print("WRs with Combine tracking:", wr_only_trk["nfl_id"].nunique())
print("WRs with no tracking:", sorted(wr_ids - set(wr_only_trk["nfl_id"].dropna())))

print("\nentity_type:", wr_only_trk["entity_type"].value_counts(dropna=False).to_dict())
print("null nfl_id in tracking:", int(wr_only_trk["nfl_id"].isna().sum()))
print("time dtype:", wr_only_trk["time"].dtype)

need = {"play": [SEP_COL, "route_ran", "nfl_id"],
        "trk": ["event_id", "nfl_id", "entity_type", "time", "drill_type", "drill_name", "attempt",
                "draft_year", "x", "y", "s", "a", "dir"],
        "results": ["nfl_id", "draft_year", "combine_position"]}
for key, cols in need.items():
    df = {"play": wr_only_play, "trk": wr_only_trk, "results": wr_only_results}[key]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {key}: {missing}")

# %%
# Regular-season outcomes only; competition files may live beside the WR extracts.
REGULAR_SEASON_ONLY = True
GAMES_PATH = next((p for p in [DATA_DIR / "games.csv",
    DATA_DIR.parent / "nfl-big-data-bowl-2027" / "games.csv"] if p.exists()), None)
if REGULAR_SEASON_ONLY:
    if GAMES_PATH is None:
        raise FileNotFoundError("Regular-season filtering requires games.csv in data/ or nfl-big-data-bowl-2027/.")
    games = pd.read_csv(GAMES_PATH)
    if not {"game_id", "season_type"}.issubset(games.columns):
        raise ValueError("games.csv requires game_id and season_type")
    if games["game_id"].duplicated().any():
        raise ValueError("games.csv contains duplicate game IDs")
    unknown = set(wr_only_play["game_id"]) - set(games["game_id"])
    if unknown:
        raise ValueError(f"Play rows reference {len(unknown)} unknown game IDs")
    reg_games = set(games.loc[games["season_type"] == "REG", "game_id"])
    wr_only_play = wr_only_play[wr_only_play["game_id"].isin(reg_games)].copy()
    play = wr_only_play
    print("play rows after regular-season filter:", len(play))
