import os
import subprocess

BASE_DIR = os.path.expanduser("/home/adamz/Documents/praca-magisterska")
MY_ENGINES_DIR = os.path.join(BASE_DIR, "data/engines/research")
ANCHOR_EXEC = os.path.join(BASE_DIR, "stockfish_anchor")
BOOK_PATH = os.path.join(BASE_DIR, "UHO_4060_v4.epd")
PGN_OUT = os.path.join(MY_ENGINES_DIR, "tournament_results.pgn")
CUTECHESS_EXEC = "./cutechess-cli"

ANCHOR_ELOS = [1000, 1200, 1400, 1600, 1800]

cmd = [CUTECHESS_EXEC]

# 1. Autorskie silniki
my_engines = sorted(os.listdir(MY_ENGINES_DIR))
for engine in my_engines:
    engine_path = os.path.join(MY_ENGINES_DIR, engine)
    if os.path.isfile(engine_path) and not engine.endswith(".pgn") and not engine.endswith(".txt"):
        cmd.extend(["-engine", f"name={engine}", f"cmd={engine_path}"])

# 2. Silniki kotwiczące
for elo in ANCHOR_ELOS:
    cmd.extend([
        "-engine", f"name=Anchor_SF_{elo}", f"cmd={ANCHOR_EXEC}",
        "option.UCI_LimitStrength=true", f"option.UCI_Elo={elo}"
    ])

# 3. Poprawny zapis dla książki otwarć i limitu węzłów w Cutechess v1.2+
# 3. Dodanie tc=inf obok nodes=10000 ucisza ostrzeżenie o czasie
cmd.extend([
    "-each", "proto=uci", "tc=inf", "nodes=10000",
    "-openings", f"file={BOOK_PATH}", "format=epd", "order=random",
    "-games", "2",
    "-rounds", "40",
    "-repeat", "2",
    "-concurrency", "10",
    "-pgnout", PGN_OUT
])

print("Uruchamianie turnieju Cutechess...")
print("Wykonuję komendę:\n" + " ".join(cmd) + "\n")
print("--- START TURNIEJU ---\n")

subprocess.run(cmd)

print(f"\nTurniej zakończony! Wyniki zapisano do: {PGN_OUT}")