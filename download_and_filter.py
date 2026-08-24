import subprocess
import urllib.request
import os
import chess.pgn

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
ZST_FILE = "lichess_2013-01.pgn.zst"
PGN_FILE = "lichess_2013-01.pgn"
OUT_FILE = "test_1200.pgn"
TARGET_GAMES = 500

if not os.path.exists(ZST_FILE):
    print(f"Downloading {URL}...")
    urllib.request.urlretrieve(URL, ZST_FILE)

if not os.path.exists(PGN_FILE):
    print("Decompressing with zstd...")
    subprocess.run(["zstd", "-d", ZST_FILE, "-o", PGN_FILE], check=True)

print(f"Filtering games (Elo ~1200, min 40 plies, normal termination). Target: {TARGET_GAMES}...")
saved = 0
with open(PGN_FILE, "r") as pgn, open(OUT_FILE, "w") as out:
    while saved < TARGET_GAMES:
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
                
                if 1150 <= avg_elo <= 1250 and ply_count >= 40:
                    out.write(str(game) + "\n\n")
                    saved += 1
                    if saved % 50 == 0:
                        print(f"Found {saved}/{TARGET_GAMES} valid games...")
            except ValueError:
                pass

print(f"Done! Saved {saved} games to {OUT_FILE}.")
