import os
import subprocess

# Ścieżki bazowe
BASE_DIR = os.path.expanduser("/home/adamz/Documents/praca-magisterska")
WEIGHTS_DIR = os.path.join(BASE_DIR, "data/nnue/research")  # Tutaj leżą pliki .nnue
STOCKFISH_EXEC = os.path.join(BASE_DIR, "stockfish_anchor")     # Główny binarek Stockfisha
BOOK_PATH = os.path.join(BASE_DIR, "UHO_4060_v4.epd")
PGN_OUT = os.path.join(WEIGHTS_DIR, "tournament_results.pgn")
CUTECHESS_EXEC = "./cutechess-cli"

ANCHOR_ELOS = [1350, 1500, 1650, 1800]

cmd = [CUTECHESS_EXEC]

# 1. Autorskie epoki - wczytywanie wag .nnue przez opcję UCI EvalFile
nnue_files = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".nnue") or "epoch" in f])

for weight_file in nnue_files:
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    
    # Pomijamy pliki wyników i logów, przetwarzamy tylko wagi
    if os.path.isfile(weight_path) and not weight_file.endswith((".pgn", ".txt", ".py")):
        # Nazwa silnika w turnieju bez rozszerzenia .nnue
        engine_name = os.path.splitext(weight_file)[0]
        
        cmd.extend([
            "-engine", f"name={engine_name}",
            f"cmd={STOCKFISH_EXEC}",
            f"option.EvalFileSmall={weight_path}"  # Wskazanie konkretnego pliku sieci dla danej epoki
        ])

# 2. Silniki kotwiczące z poprawnie rozdzielonymi opcjami UCI
for elo in ANCHOR_ELOS:
    cmd.extend([
        "-engine", f"name=Anchor_SF_{elo}",
        f"cmd={STOCKFISH_EXEC}",
        "option.UCI_LimitStrength=true",  # Włączenie limitowania siły
        f"option.UCI_Elo={elo}"           # Ustawienie docelowego Elo
    ])

# 3. Parametry turniejowe
cmd.extend([
    "-each", "proto=uci", "tc=inf", "nodes=10000",
    "-openings", f"file={BOOK_PATH}", "format=epd", "order=random",
    "-games", "2",
    "-rounds", "40",
    "-repeat", "2",
    "-concurrency", "10",
    "-pgnout", PGN_OUT
])

print("Uruchamianie turnieju Cutechess z ewaluacją plików NNUE...")
print("Wykonuję komendę:\n" + " ".join(cmd) + "\n")
print("--- START TURNIEJU ---\n")

subprocess.run(cmd)

print(f"\nTurniej zakończony! Wyniki zapisano do: {PGN_OUT}")