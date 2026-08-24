import os
import subprocess
import re
import json
BASE_DIR = os.path.expanduser("/home/adamz/Documents/praca-magisterska")

# ======================================================================
#                        TOURNAMENT CONFIGURATION
# ======================================================================

# MODE TOGGLE
# "anchors"          -> All .nnue nets play against Stockfish anchors in one giant tournament
# "separate_anchors" -> Each .nnue net runs its own independent tournament against Stockfish anchors
# "adjacent"         -> Nets play only their immediate neighbour (e.g. epoch 10 vs 20, 20 vs 30)
# "round_robin"      -> Every .nnue net plays against every other .nnue net
COMPARE_MODE = "separate_anchors"  

WEIGHTS_DIR = os.path.join(BASE_DIR, "nnue/research")  # Tutaj leżą pliki .nnue

# SPEED OPTIMIZATIONS
# To make the tournament fast, we reduce the nodes-per-move constraint and total games.
# 'tc=inf' turns off the absolute clock so Python doesn't lose on time due to startup/overhead.
# 'nodes=500' restricts exactly how many nodes are searched per move (very fast).
ROUNDS       = "50"    # Number of rounds (Total games = ROUNDS * 2 for colors)
CONCURRENCY  = "8"    # How many games run at the same time
NODES        = "2000"  # Nodes to analyze per move.

# ======================================================================

# Ścieżki bazowe
STOCKFISH_EXEC = os.path.join(BASE_DIR, "stockfish_anchor")
BOOK_PATH = os.path.join(BASE_DIR, "UHO_4060_v4.epd")
PGN_OUT = os.path.join(WEIGHTS_DIR, f"tournament_results_{COMPARE_MODE}.pgn")
JSON_OUT = os.path.join(WEIGHTS_DIR, f"tournament_results_{COMPARE_MODE}.json")
CUTECHESS_EXEC = "./cutechess-cli"

PYTHON_EXEC = os.path.join(BASE_DIR, ".venv/bin/python3")
UCI_ENGINE_SCRIPT = os.path.join(BASE_DIR, "uci_engine_optimized.py")

ANCHOR_ELOS = [1350, 1500, 1800]

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def build_engine_arg(name, path):
    return [
        "-engine", f"name={name}",
        f"cmd={PYTHON_EXEC}", f"arg={UCI_ENGINE_SCRIPT}", "arg=--net", f"arg={path}"
    ]

def build_anchor_arg(elo):
    return [
        "-engine", f"name=Anchor_SF_{elo}",
        f"cmd={STOCKFISH_EXEC}", "option.UCI_LimitStrength=true", f"option.UCI_Elo={elo}"
    ]

def parse_cutechess_output(output_text):
    """Extracts final Elo diff, error margin, and game statistics using regex."""
    lines = output_text.strip().split("\n")
    
    result = {
        "elo_diff": 0.0,
        "error_margin": 0.0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "los_percent": 0.0,
        "draw_ratio_percent": 0.0,
        "raw_output": lines[-30:] if len(lines) >= 30 else lines
    }

    score_matches = re.findall(r"Score of .*? vs .*?:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", output_text)
    if score_matches:
        final_score = score_matches[-1]
        result["wins"] = int(final_score[0])
        result["losses"] = int(final_score[1])
        result["draws"] = int(final_score[2])

    elo_match = re.search(
        r"Elo difference:\s*([+-]?\d+\.?\d*)\s*[+-/]+\s*(\d+\.?\d*),\s*LOS:\s*(\d+\.?\d*)\s*%,\s*DrawRatio:\s*(\d+\.?\d*)\s*%", 
        output_text
    )
    
    if elo_match:
        result["elo_diff"] = float(elo_match.group(1))
        result["error_margin"] = float(elo_match.group(2))
        result["los_percent"] = float(elo_match.group(3))
        result["draw_ratio_percent"] = float(elo_match.group(4))
        
    return result

def run_cutechess(engines, tournament_type):
    cmd = [CUTECHESS_EXEC]
    for eng in engines:
        cmd.extend(eng)
        
    cmd.extend([
        "-each", "proto=uci", "tc=inf", f"nodes={NODES}",
        "-openings", f"file={BOOK_PATH}", "format=epd", "order=random",
        "-games", "2",
        "-rounds", ROUNDS,
        "-repeat", "2",
        "-concurrency", CONCURRENCY,
        "-tournament", tournament_type,
        "-pgnout", PGN_OUT
    ])
    
    print("Wykonuję komendę:\n" + " ".join(cmd) + "\n")
    
    # Run with Popen to capture output while still printing it to terminal live
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
        
    process.wait()
    output_text = "".join(output_lines)
    return parse_cutechess_output(output_text)

# ─── ZBIERANIE SIECI ──────────────────────────────────────────
nnue_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".nnue") or "epoch" in f]
nnue_files = [f for f in nnue_files if not f.endswith((".pgn", ".txt", ".py"))]
nnue_files.sort(key=natural_sort_key)

if not nnue_files:
    print(f"Nie znaleziono plików .nnue w {WEIGHTS_DIR}")
    exit(1)

print(f"--- START TURNIEJU (TRYB: {COMPARE_MODE}) ---\n")

if COMPARE_MODE == "anchors":
    engines = []
    for f in nnue_files:
        engines.append(build_engine_arg(os.path.splitext(f)[0], os.path.join(WEIGHTS_DIR, f)))
    for elo in ANCHOR_ELOS:
        engines.append(build_anchor_arg(elo))
        
    metrics = run_cutechess(engines, "gauntlet")
    results_report = {"anchors_tournament": metrics}

elif COMPARE_MODE == "separate_anchors":
    results_report = []
    for f in nnue_files:
        net_name = os.path.splitext(f)[0]
        print(f"\n>> Turniej kotwiczący dla sieci: {net_name}")
        
        # In a gauntlet, the FIRST engine plays against the rest
        engines = [build_engine_arg(net_name, os.path.join(WEIGHTS_DIR, f))]
        
        for elo in ANCHOR_ELOS:
            engines.append(build_anchor_arg(elo))
            
        metrics = run_cutechess(engines, "gauntlet")
        results_report.append({
            "net": net_name,
            "metrics": metrics
        })

elif COMPARE_MODE == "round_robin":
    engines = []
    for f in nnue_files:
        engines.append(build_engine_arg(os.path.splitext(f)[0], os.path.join(WEIGHTS_DIR, f)))
        
    metrics = run_cutechess(engines, "round-robin")
    results_report = {"round_robin_tournament": metrics}

elif COMPARE_MODE == "adjacent":
    if len(nnue_files) < 2:
        print("Za mało sieci do trybu adjacent (minimum 2).")
        exit(1)
        
    results_report = []
    for i in range(len(nnue_files) - 1):
        file_a, file_b = nnue_files[i], nnue_files[i+1]
        name_a = os.path.splitext(file_a)[0]
        name_b = os.path.splitext(file_b)[0]
        
        print(f"\n>> Mecz {i+1}/{len(nnue_files)-1}: {name_a} vs {name_b}")
        
        eng_a = build_engine_arg(name_a, os.path.join(WEIGHTS_DIR, file_a))
        eng_b = build_engine_arg(name_b, os.path.join(WEIGHTS_DIR, file_b))
        
        metrics = run_cutechess([eng_a, eng_b], "round-robin")
        results_report.append({
            "match": f"{name_a}_vs_{name_b}",
            "metrics": metrics
        })

with open(JSON_OUT, "w") as f:
    json.dump(results_report, f, indent=4)

print(f"\nTurniej zakończony! Wyniki zapisano do: {PGN_OUT}")