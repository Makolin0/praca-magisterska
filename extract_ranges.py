import chess.pgn
import os
import sys

PGN_FILE = "lichess_2013-01.pgn"
TARGET_GAMES = 500

ranges = {
    "test_1200_1500.pgn": (1200, 1500),
    "test_1500_1800.pgn": (1500, 1800),
    "test_0_1800.pgn": (0, 1800)
}

counts = {name: 0 for name in ranges}
out_files = {name: open(name, "w") for name in ranges}

if not os.path.exists(PGN_FILE):
    print(f"Błąd: Plik {PGN_FILE} nie istnieje. Prawdopodobnie trzeba uruchomić download_and_filter.py najpierw.")
    sys.exit(1)

print("Rozpoczęto filtrowanie gier dla przedziałów: 1200-1500, 1500-1800 oraz 0-1800...")

with open(PGN_FILE, "r") as pgn:
    while True:
        # Check if all targets met
        if all(c >= TARGET_GAMES for c in counts.values()):
            break
            
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
            
        white_elo = game.headers.get("WhiteElo", "?")
        black_elo = game.headers.get("BlackElo", "?")
        termination = game.headers.get("Termination", "Normal")
        
        if white_elo != "?" and black_elo != "?" and termination == "Normal":
            try:
                we = int(white_elo)
                be = int(black_elo)
                avg_elo = (we + be) / 2
                
                ply_count = sum(1 for _ in game.mainline_moves())
                
                if ply_count >= 40:
                    game_str = str(game) + "\n\n"
                    for name, (elo_min, elo_max) in ranges.items():
                        if counts[name] < TARGET_GAMES and elo_min <= avg_elo <= elo_max:
                            out_files[name].write(game_str)
                            counts[name] += 1
                            if counts[name] % 100 == 0:
                                print(f"[{name}] Znaleziono {counts[name]}/{TARGET_GAMES}")
            except ValueError:
                pass

for f in out_files.values():
    f.close()
    
print("Zakończono wyodrębnianie plików PGN dla nowych przedziałów!")
