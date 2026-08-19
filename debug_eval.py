#!/usr/bin/env python3
"""Direct evaluation diagnostic."""
import sys, struct, chess
import numpy as np

sys.path.insert(0, '/home/adamz/Documents/praca-magisterska')

import importlib.util
spec = importlib.util.spec_from_file_location(
    "uci_engine", "/home/adamz/Documents/praca-magisterska/uci_engine.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
load_nnue = mod.load_nnue
evaluate  = mod.evaluate

w = load_nnue('/home/adamz/Documents/praca-magisterska/nnue/control/epoch_300.nnue')

board = chess.Board()
board.push_uci('e2e4'); board.push_uci('e7e5')

print("=== Position: after 1.e4 e5, WHITE to move ===\n")

scores = {}
for move in board.legal_moves:
    board.push(move)
    sc = evaluate(w, board)
    scores[move.uci()] = sc
    board.pop()

# Show top 5 and worst 5 moves (from white's perspective = -sc)
ranked = sorted(scores.items(), key=lambda x: -x[1])  # higher = better for black

print("Score after move (from BLACK=STM perspective, should be positive after e1e2 = black winning):")
for mv, sc in ranked[:5]:
    print(f"  {mv}: {sc:+d} cp (white move value = {-sc:+d} cp)")
print("  ...")
for mv, sc in ranked[-5:]:
    print(f"  {mv}: {sc:+d} cp (white move value = {-sc:+d} cp)")

print(f"\ne1e2 specifically: evaluate={scores.get('e1e2','?')}")
print(f"g1f3 specifically: evaluate={scores.get('g1f3','?')}")
print("\nFor e1e2 to be CORRECT: its evaluate() should be LARGE POSITIVE (black winning).")
print("If e1e2 evaluate() ≈ -39, the sign is WRONG.")
