import os
import subprocess
import re
import json
from pathlib import Path

# ==========================================
#               CONFIGURATION
# ==========================================

# ==========================================
#              MODE SWITCH
# ==========================================
COMPARE_MODE = "adjacent"  
# Options:
# "baseline"   -> each net vs baseline
# "round_robin" -> all nets vs each other
# "adjacent"    -> sequential pairing (e.g., e10 vs e20, e20 vs e30...)
# ==========================================


# Paths to your core directories
STOCKFISH_SRC_DIR = "/home/adamz/Documents/praca-magisterska/stockfish-source/src"  # Must point to Stockfish's 'src' directory
CUTECHESS_CLI_PATH = "/home/adamz/Documents/praca-magisterska/cutechess-cli"          # Change to absolute path if not in system PATH
NEW_NETS_DIR = "/home/adamz/Documents/praca-magisterska/data/nnue/1200_under" # Directory containing the .nnue files to test
OUTPUT_JSON_PATH = "/home/adamz/Documents/praca-magisterska/data/engines/1200_under/tournament_results.json"

# GLOBAL PERSISTENT BINARY CACHE DIRECTORY
COMPILED_BIN_DIR = "/home/adamz/Documents/praca-magisterska/data/engines/1200_under"
os.makedirs(COMPILED_BIN_DIR, exist_ok=True)


# Hardware / Compilation settings
CPU_ARCH = "x86-64-bmi2"                      # e.g., x86-64-bmi2, x86-64-avx2, apple-silicon
CONCURRENCY = 10                               # Number of concurrent games cutechess should run

# Cutechess match settings
OPENING_BOOK_PATH = "/home/adamz/Documents/praca-magisterska/UHO_4060_v4.epd"  # Path to an .epd or .pgn opening book
GAME_COUNT = 400                              # 400-1000 games recommended for stable Elo
TIME_CONTROL = "10+0.1"                       # Time control format: base+increment

# Baseline networks to compare your new nets against
BASELINE_NETS = [
    # {
    #     "name": "SF_Baseline_1",
    #     "path": "/path/to/baselines/nn-111111111111.nnue",
    #     "elo": 3500  # Known anchor Elo
    # },
    # {
    #     "name": "SF_Baseline_2",
    #     "path": "/path/to/baselines/nn-222222222222.nnue",
    #     "elo": 3550
    # }
]
# ==========================================

def natural_sort_key(s):
    """Helper to sort strings containing numbers naturally (e.g., epoch_20 before epoch_100)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def run_adjacent_comparison(compiled_nets, sorted_net_files):
    """Pairs sorted nets sequentially: net[i] vs net[i+1]."""
    results_report = {}
    
    # Extract names in their naturally sorted order
    sorted_names = [f.stem for f in sorted_net_files if f.stem in compiled_nets]
    
    if len(sorted_names) < 2:
        print("Not enough successfully compiled nets to perform adjacent comparison.")
        return results_report

    print(f"Chaining adjacent matches across {len(sorted_names)} networks...")
    
    for i in range(len(sorted_names) - 1):
        name_a = sorted_names[i]
        name_b = sorted_names[i + 1]
        
        binary_a = compiled_nets[name_a]
        binary_b = compiled_nets[name_b]
        
        match_metrics = run_tournament(
            binary_a,
            binary_b,
            name_a=name_a,
            name_b=name_b
        )
        
        # Store results clearly mapping the progression
        match_key = f"{name_a}_vs_{name_b}"
        results_report[match_key] = {
            "engine_a": name_a,
            "engine_b": name_b,
            "match_metrics": match_metrics
        }
        
    return results_report

def run_round_robin(compiled_nets):
    """Each engine plays against every other engine."""
    results_report = {name: [] for name in compiled_nets.keys()}

    net_names = list(compiled_nets.keys())

    for i in range(len(net_names)):
        for j in range(i + 1, len(net_names)):
            name_a = net_names[i]
            name_b = net_names[j]

            binary_a = compiled_nets[name_a]
            binary_b = compiled_nets[name_b]

            match_metrics = run_tournament(
                binary_a,
                binary_b,
                name_a=name_a,
                name_b=name_b
            )

            results_report[name_a].append({
                "opponent": name_b,
                "match_metrics": match_metrics
            })

            results_report[name_b].append({
                "opponent": name_a,
                "match_metrics": match_metrics
            })

    return results_report

def compile_all_nets(net_files):
    """Compile all networks once and return {name: binary_path}."""
    compiled = {}

    for net_file in net_files:
        net_name = net_file.stem
        binary = compile_stockfish(str(net_file), f"sf_{net_name}")

        if binary:
            compiled[net_name] = binary
        else:
            print(f"Skipping {net_name} due to compilation error.")

    return compiled

def compile_stockfish(net_path, binary_name):
    """
    Compiles Stockfish with a given NNUE, but uses caching:
    if binary already exists in COMPILED_BIN_DIR -> reuse it instantly.
    """
    abs_net_path = os.path.abspath(net_path)
    cached_binary = os.path.join(COMPILED_BIN_DIR, binary_name)

    if os.path.exists(cached_binary):
        print(f"[CACHE HIT] Found existing compiled binary: {binary_name}")
        return cached_binary

    print(f"[COMPILING] {os.path.basename(net_path)} -> {binary_name} (This may take a while...)")

    try:
        subprocess.run(
            ["make", "clean"],
            cwd=STOCKFISH_SRC_DIR,
            check=True,
            stdout=subprocess.DEVNULL
        )

        compile_cmd = [
            "make",
            f"-j{os.cpu_count()}",
            "profile-build",
            "COMP=gcc",
            f"ARCH={CPU_ARCH}",
            f"EVALFILE={abs_net_path}"
        ]

        subprocess.run(
            compile_cmd,
            cwd=STOCKFISH_SRC_DIR,
            check=True,
            stdout=subprocess.DEVNULL
        )

        os.rename(
            os.path.join(STOCKFISH_SRC_DIR, "stockfish"),
            cached_binary
        )

        return cached_binary

    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {binary_name}: {e}")
        return None
    
def parse_cutechess_output(output_text):
    """Extracts final Elo diff, error margin, and game statistics using regex."""
    lines = output_text.strip().split("\n")
    
    # Initialize defaults
    result = {
        "elo_diff": 0.0,
        "error_margin": 0.0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "los_percent": 0.0,
        "draw_ratio_percent": 0.0,
        "raw_output": lines[-15:] if len(lines) >= 15 else lines # Capture a better concluding snapshot
    }

    # Find the *last* score line to ensure we have the final match result
    score_matches = re.findall(r"Score of .*? vs .*?:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", output_text)
    if score_matches:
        final_score = score_matches[-1] # Grabs the final entry
        result["wins"] = int(final_score[0])
        result["losses"] = int(final_score[1])
        result["draws"] = int(final_score[2])

    # Extract final Elo details along with LOS and DrawRatio
    # Pattern accounts for: Elo difference: 13.9 +/- 48.4, LOS: 71.4 %, DrawRatio: 50.0 %
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

def run_tournament(engine_a, engine_b, name_a, name_b):
    """Runs a cutechess-cli tournament matching the requested format."""
    print(f"--- Running Match: {name_a} vs {name_b} ---")
    
    # Strip common prefixes like 'sf_' if you just want '20' or '30' as the engine name,
    # or it will fall back to using whatever name string is passed to it.
    display_name_a = name_a.replace("sf_", "").replace("epoch_", "")
    display_name_b = name_b.replace("sf_", "").replace("epoch_", "")

    cmd = [
        CUTECHESS_CLI_PATH,
        "-engine", f"cmd={engine_a}", f"name={display_name_a}",
        "-engine", f"cmd={engine_b}", f"name={display_name_b}",
        "-each", "proto=uci", f"tc={TIME_CONTROL}", "option.Hash=16", "option.Threads=1",
        "-tournament", "gauntlet",
        "-games", str(GAME_COUNT),
        "-repeat",
        "-openings", f"file={OPENING_BOOK_PATH}", "format=epd", "order=random",
        "-concurrency", str(CONCURRENCY)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    return parse_cutechess_output(process.stdout)

def main():
    results_report = {}

    # Grab and naturally sort all .nnue files 
    new_net_files = list(Path(NEW_NETS_DIR).glob("*.nnue"))
    new_net_files.sort(key=lambda f: natural_sort_key(f.name))
    
    print(f"Found {len(new_net_files)} networks in {NEW_NETS_DIR}.")

    if COMPARE_MODE == "adjacent":
        print("Running ADJACENT chain comparison mode...")
        compiled_nets = compile_all_nets(new_net_files)
        results_report = run_adjacent_comparison(compiled_nets, new_net_files)

    elif COMPARE_MODE == "round_robin":
        print("Running ROUND-ROBIN mode...")
        compiled_nets = compile_all_nets(new_net_files)
        results_report = run_round_robin(compiled_nets)

    else:
        print("Running BASELINE comparison mode...")

        # Pre-compile baselines
        compiled_baselines = []
        for idx, base in enumerate(BASELINE_NETS):
            bin_name = f"sf_base_{idx}_{base['name']}"
            bin_path = compile_stockfish(base["path"], bin_name)
            if bin_path:
                compiled_baselines.append({
                    "name": base["name"],
                    "anchor_elo": base["elo"],
                    "binary_path": bin_path
                })

        for net_file in new_net_files:
            net_name = net_file.stem
            results_report[net_name] = []

            candidate_binary = compile_stockfish(str(net_file), f"sf_{net_name}")
            if not candidate_binary:
                continue

            for base in compiled_baselines:
                match_metrics = run_tournament(
                    candidate_binary,
                    base["binary_path"],
                    name_a=net_name,
                    name_b=base["name"]
                )

                calculated_elo = base["anchor_elo"] + match_metrics["elo_diff"]

                results_report[net_name].append({
                    "compared_against": base["name"],
                    "baseline_anchor_elo": base["anchor_elo"],
                    "calculated_absolute_elo": round(calculated_elo, 2),
                    "match_metrics": match_metrics
                })

    # save results
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results_report, f, indent=4)

    print(f"Done. Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()