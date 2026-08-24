import sys
import os
import glob
import re
import csv
import chess
import chess.pgn
import argparse
from uci_engine import load_nnue, evaluate

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-dir", required=True, help="Directory containing .nnue files")
    parser.add_argument("--pgn", required=True, help="Path to test PGN file")
    parser.add_argument("--max-games", type=int, default=100, help="Max games to test")
    args = parser.parse_args()
    
    # 1. Parse PGN and store positions to evaluate
    print(f"Parsing up to {args.max_games} games from {args.pgn}...")
    positions = []
    
    with open(args.pgn, "r") as pgn:
        games_processed = 0
        while games_processed < args.max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            board = game.board()
            for move in game.mainline_moves():
                # Store the fen and the move to guess
                positions.append((board.fen(), move))
                board.push(move)
                
            games_processed += 1
            if games_processed % 50 == 0:
                print(f"Parsed {games_processed} games...")

    print(f"Total positions to evaluate per network: {len(positions)}")
    
    # 2. Find all networks
    net_files = glob.glob(os.path.join(args.net_dir, "*.nnue"))
    net_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
    
    if not net_files:
        print(f"No .nnue files found in {args.net_dir}")
        return
        
    print(f"Found {len(net_files)} networks in {args.net_dir}")
    
    results_csv = os.path.join(args.net_dir, "human_like_results.csv")
    
    # 3. Evaluate each network
    with open(results_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Network", "Matched_Moves", "Total_Moves", "Match_Percentage"])
        
        for net_file in net_files:
            net_name = os.path.basename(net_file)
            print(f"\n--- Testing {net_name} ---")
            weights = load_nnue(net_file)
            
            matched_moves = 0
            total_moves = len(positions)
            
            # We recreate the board object once and just use set_fen to be faster
            board = chess.Board()
            
            for i, (fen, actual_move) in enumerate(positions):
                board.set_fen(fen)
                best_score = -999999
                best_move = None
                
                for legal in board.legal_moves:
                    board.push(legal)
                    score = -evaluate(weights, board)
                    board.pop()
                    
                    if score > best_score:
                        best_score = score
                        best_move = legal
                        
                if best_move == actual_move:
                    matched_moves += 1
                    
                if (i + 1) % 2000 == 0:
                    print(f"  Processed {i + 1}/{total_moves} positions...")
                    
            match_pct = (matched_moves / total_moves) * 100 if total_moves > 0 else 0
            print(f"Result for {net_name}: {matched_moves}/{total_moves} ({match_pct:.2f}%)")
            
            writer.writerow([net_name, matched_moves, total_moves, f"{match_pct:.2f}"])
            csvfile.flush()
            
    print(f"\nAll tests finished. Results saved to {results_csv}")

if __name__ == "__main__":
    main()
