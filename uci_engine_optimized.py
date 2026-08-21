#!/usr/bin/env python3
"""
Standalone Python UCI chess engine using a custom .nnue net trained with nnue-pytorch.
Zero nnue-pytorch dependencies — reads the .nnue binary format directly.

Quantization constants (from nnue-pytorch model/quantize.py defaults):
  nnue2score = 600.0 | ft_quantized_one = 255.0 | hidden_quantized_one = 127.0
  weight_scale_hidden = 64.0 | weight_scale_out = 16.0

Usage:
    python3 uci_engine.py                          # uses NNUE_PATH below
    python3 uci_engine.py --net /path/to/net.nnue
"""

import sys
import time
import struct
import argparse
import chess
import os

# Prevent numpy from spawning threads. When Cutechess runs 10 games concurrently (20 engine instances),
# numpy's default multithreading (often equal to physical cores) causes massive CPU contention,
# slowing execution from seconds to several minutes per move.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# ======================================================================
NNUE_PATH   = '/home/adamz/Documents/praca-magisterska/nnue/control/epoch_100.nnue'
ENGINE_NAME = 'PyNNUE'
ENGINE_VER  = '1.0'
# ── Architecture (must match trainer defaults) ─────────────────────────
L1            = 1024   # feature transformer output per side
L2            = 31     # hidden layer 1 outputs  (L2+1 stored in file)
L3            = 32     # hidden layer 2 outputs
NUM_LS        = 8      # layer stacks / buckets
NUM_PSQT      = 8      # PSQT buckets
# ── Quantization (from QuantizationConfig defaults) ───────────────────
FT_SCALE      = 255.0          # ft_quantized_one
FC_SCALE_H    = 64.0           # weight_scale_hidden
FC_SCALE_OUT  = 16.0           # weight_scale_out
HIDDEN_ONE    = 127.0          # hidden_quantized_one
NNUE2SCORE    = 600.0
# Derived bias scales
FC_BIAS_SCALE_H   = FC_SCALE_H * HIDDEN_ONE          # 8128
FC_BIAS_SCALE_OUT = FC_SCALE_OUT * NNUE2SCORE         # 9600
FC_W_SCALE_OUT    = NNUE2SCORE * FC_SCALE_OUT / HIDDEN_ONE  # ≈75.6
# ── Feature set (HalfKAv2_hm) ─────────────────────────────────────────
NUM_REAL_FEATURES = 704 * 32   # 22528  (11 piece types × 64 sq × 32 king buckets)
MATE_SCORE        = 32000
INF               = 99999
# ======================================================================

# fmt: off
_KING_BUCKETS = [
  -1,-1,-1,-1, 31,30,29,28,
  -1,-1,-1,-1, 27,26,25,24,
  -1,-1,-1,-1, 23,22,21,20,
  -1,-1,-1,-1, 19,18,17,16,
  -1,-1,-1,-1, 15,14,13,12,
  -1,-1,-1,-1, 11,10, 9, 8,
  -1,-1,-1,-1,  7, 6, 5, 4,
  -1,-1,-1,-1,  3, 2, 1, 0,
]
# fmt: on

def _orient(is_white: bool, sq: int, ksq: int) -> int:
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white)) ^ sq

def halfka_idx(is_white: bool, ksq: int, sq: int, piece_type: int, piece_color: bool) -> int:
    """Feature index in EXPORT format (704 per king bucket, 11 piece types × 64)."""
    p_idx = (piece_type - 1) * 2 + (piece_color != is_white)
    o_ksq = _orient(is_white, ksq, ksq)
    return _orient(is_white, sq, ksq) + p_idx * 64 + _KING_BUCKETS[o_ksq] * 704


# ──────────────────────────────────────────────────────────────────────
# .nnue binary reader  (handles COMPRESSED_LEB128 for FT section)
# ──────────────────────────────────────────────────────────────────────

def _read_u32(f): return struct.unpack('<I', f.read(4))[0]
def _pad32(n): return n if n % 32 == 0 else n + 32 - n % 32

_LEB_MAGIC = b"COMPRESSED_LEB128"

def _decode_sleb128(data: bytes, n: int) -> np.ndarray:
    """Decode n signed LEB128 integers from a bytes object."""
    result = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        r = 0
        shift = 0
        while True:
            byte = data[k]; k += 1
            r |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                if byte & 0x40:
                    r |= -(1 << shift)   # sign-extend
                result[i] = r
                break
    return result

def _read_tensor(f, dtype: np.dtype, count: int) -> np.ndarray:
    """Read `count` values, transparently handling COMPRESSED_LEB128."""
    peek = f.read(17)
    if peek == _LEB_MAGIC:
        length = struct.unpack('<I', f.read(4))[0]
        compressed = f.read(length)
        return _decode_sleb128(compressed, count).astype(dtype)
    else:
        # Not compressed — back up and read raw
        remaining = count * np.dtype(dtype).itemsize - 17
        raw = peek + f.read(remaining)
        return np.frombuffer(raw, dtype=dtype).copy()

class NNUEWeights:
    __slots__ = ['ft_bias', 'ft_weight', 'ft_psqt',
                 'l1_bias', 'l1_weight',
                 'l2_bias', 'l2_weight',
                 'lo_bias', 'lo_weight']

def load_nnue(path: str) -> NNUEWeights:
    w = NNUEWeights()
    with open(path, 'rb') as f:
        # ── Header ───────────────────────────────────────────────────
        _version = _read_u32(f)
        _hash    = _read_u32(f)
        desc_len = _read_u32(f)
        f.read(desc_len)

        # ── Feature transformer ──────────────────────────────────────
        _ft_hash = _read_u32(f)

        # Bias: L1 int16, possibly leb128-compressed
        raw_bias = _read_tensor(f, np.int16, L1)
        w.ft_bias = raw_bias.astype(np.float32) / FT_SCALE

        # Weights: [NUM_REAL_FEATURES, L1] int16, possibly leb128-compressed
        raw_w = _read_tensor(f, np.int16, NUM_REAL_FEATURES * L1)
        w.ft_weight = raw_w.reshape(NUM_REAL_FEATURES, L1).astype(np.float32) / FT_SCALE

        # PSQT: [NUM_REAL_FEATURES, NUM_PSQT] int32, possibly leb128-compressed
        raw_p = _read_tensor(f, np.int32, NUM_REAL_FEATURES * NUM_PSQT)
        w.ft_psqt = raw_p.reshape(NUM_REAL_FEATURES, NUM_PSQT).astype(np.float32) \
                    / (NNUE2SCORE * FC_SCALE_OUT)

        # ── 8 layer stacks ───────────────────────────────────────────
        out_l1 = L2 + 1     # 32 per bucket (includes skip)
        out_l2 = L3         # 32
        out_lo = 1

        in_l1  = L1 * 2     # 2048 (both sides concatenated after crelu)
        # After sqrcrelu + crelu concat: each side 512 elements → 1024 total
        # but in the SqrCReLU variant: L1//2 * 2 pairs → L1//2 values each side
        # actual l1 input after the SqrCReLU: 2*(L1//2) = L1 = 1024 per position
        # see model.py forward: l0_s1 = [l0_s[0]*l0_s[1], l0_s[2]*l0_s[3]] → L1//2*2 = L1
        in_l1_actual = L1   # 1024 (post halfkp-product, both sides)
        in_l2  = L2 * 2     # 62 (sqr+linear concat: L2 each)
        in_lo  = L3         # 32

        w.l1_bias   = np.zeros((NUM_LS, out_l1), np.float32)
        w.l1_weight = np.zeros((NUM_LS, out_l1, in_l1_actual), np.float32)
        w.l2_bias   = np.zeros((NUM_LS, out_l2), np.float32)
        w.l2_weight = np.zeros((NUM_LS, out_l2, in_l2), np.float32)
        w.lo_bias   = np.zeros((NUM_LS, out_lo), np.float32)
        w.lo_weight = np.zeros((NUM_LS, out_lo, in_lo), np.float32)

        for i in range(NUM_LS):
            _fc_hash = _read_u32(f)

            # l1: bias int32, weights int8 (padded input to multiple of 32)
            b1 = np.frombuffer(f.read(out_l1 * 4), dtype=np.int32).astype(np.float32)
            w.l1_bias[i] = b1 / FC_BIAS_SCALE_H
            p1 = _pad32(in_l1_actual)
            r1 = np.frombuffer(f.read(out_l1 * p1), dtype=np.int8).astype(np.float32)
            w.l1_weight[i] = r1.reshape(out_l1, p1)[:, :in_l1_actual] / FC_SCALE_H

            # l2: bias int32, weights int8
            b2 = np.frombuffer(f.read(out_l2 * 4), dtype=np.int32).astype(np.float32)
            w.l2_bias[i] = b2 / FC_BIAS_SCALE_H
            p2 = _pad32(in_l2)
            r2 = np.frombuffer(f.read(out_l2 * p2), dtype=np.int8).astype(np.float32)
            w.l2_weight[i] = r2.reshape(out_l2, p2)[:, :in_l2] / FC_SCALE_H

            # output: bias int32, weights int8
            bo = np.frombuffer(f.read(out_lo * 4), dtype=np.int32).astype(np.float32)
            w.lo_bias[i] = bo / FC_BIAS_SCALE_OUT
            po = _pad32(in_lo)
            ro = np.frombuffer(f.read(out_lo * po), dtype=np.int8).astype(np.float32)
            w.lo_weight[i] = ro.reshape(out_lo, po)[:, :in_lo] / FC_W_SCALE_OUT

    return w


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def _get_features(board: chess.Board):
    """Returns (white_ft_indices, black_ft_indices, ls_idx).
    Both kings are active in HalfKAv2_hm. In export format the king block
    (10*64 offset per bucket) holds: own king at o_ksq, opp king at its
    oriented square — both map to  bucket*704 + 10*64 + oriented_sq.
    """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    w_idx, b_idx = [], []

    # Non-king pieces (p_idx 0..9)
    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        if pt == chess.KING:
            continue
        pc = (piece.color == chess.WHITE)
        w_idx.append(halfka_idx(True,  wk, sq, pt, pc))
        b_idx.append(halfka_idx(False, bk, sq, pt, pc))

    # White's POV king features (own + opponent, both in the king block)
    w_o_ksq = _orient(True, wk, wk)
    w_base  = _KING_BUCKETS[w_o_ksq] * 704 + 10 * 64
    w_idx.append(w_base + w_o_ksq)                 # own (white) king
    w_idx.append(w_base + _orient(True, bk, wk))   # opponent (black) king

    # Black's POV king features
    b_o_ksq = _orient(False, bk, bk)
    b_base  = _KING_BUCKETS[b_o_ksq] * 704 + 10 * 64
    b_idx.append(b_base + b_o_ksq)                 # own (black) king
    b_idx.append(b_base + _orient(False, wk, bk))  # opponent (white) king

    piece_count = len(board.piece_map())
    ls = min((piece_count - 1) // 4, 7)
    return w_idx, b_idx, ls


def evaluate(w: NNUEWeights, board: chess.Board) -> int:
    """Return score in centipawns from side-to-move's perspective."""
    w_idx, b_idx, ls = _get_features(board)

    wft = w.ft_weight[w_idx].sum(axis=0) + w.ft_bias   # [L1]
    bft = w.ft_weight[b_idx].sum(axis=0) + w.ft_bias   # [L1]

    # PSQT: model uses (wpsqt - bpsqt) * (us - 0.5)
    # us=1 for white → factor 0.5; us=0 for black → factor -0.5
    wpsqt = w.ft_psqt[w_idx, ls].sum()
    bpsqt = w.ft_psqt[b_idx, ls].sum()
    psqt_factor = 0.5 if board.turn == chess.WHITE else -0.5
    psqt_score = (wpsqt - bpsqt) * psqt_factor

    # STM perspective: us_ft = side to move, them_ft = opponent
    if board.turn == chess.WHITE:
        us_ft, them_ft = wft, bft
    else:
        us_ft, them_ft = bft, wft

    # SqrCReLU  (model.py):
    # l0_ = clamp(cat([us_ft, them_ft]), 0, 1)           → [2048]
    # split into 4×512, product pairs → [1024], scale 127/128
    h = L1 // 2  # 512
    l0_full = np.clip(np.concatenate([us_ft, them_ft]), 0.0, 1.0)  # [2048]
    l0 = np.concatenate([
        l0_full[:h]      * l0_full[h:L1],      # us  product [512]
        l0_full[L1:L1+h] * l0_full[L1+h:],     # them product [512]
    ]) * (127.0 / 128.0)                        # [1024]

    # l1: 1024 → 32  (SqrCReLU + linear → 62 outputs)
    l1_raw  = w.l1_weight[ls] @ l0 + w.l1_bias[ls]   # [32]
    l1_skip = l1_raw[L2:]                              # skip connection [1]
    l1_main = l1_raw[:L2]                              # [31]
    # model.py: clamp(cat([pow(l1x_,2)*(255/256), l1x_]), 0, 1)
    # square BEFORE clamping, apply 255/256 scale
    l1_sqr = np.clip(l1_main ** 2 * (255.0 / 256.0), 0.0, 1.0)
    l1_lin = np.clip(l1_main, 0.0, 1.0)
    l1_out = np.concatenate([l1_sqr, l1_lin])          # [62]

    # l2: 62 → 32, CReLU
    l2_raw = w.l2_weight[ls] @ l1_out + w.l2_bias[ls]
    l2_out = np.clip(l2_raw, 0.0, 1.0)

    # output: 32 → 1
    net_out = (w.lo_weight[ls] @ l2_out + w.lo_bias[ls])[0]

    total = net_out + l1_skip[0] + psqt_score
    return int(total * NNUE2SCORE)


# ──────────────────────────────────────────────────────────────────────
# Alpha-beta search & Move Ordering
# ──────────────────────────────────────────────────────────────────────

# Global Transposition Table
TT = {}

def clear_tt():
    global TT
    TT.clear()

def move_score(board, move):
    """Assigns a score to a move for ordering (MVV-LVA)."""
    if board.is_capture(move):
        if board.is_en_passant(move):
            return 105
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            return 100 + (victim.piece_type * 10 - attacker.piece_type)
        return 100
    if move.promotion:
        return 90 + move.promotion
    return 0

def quiesce(w, board, alpha, beta, nodes, nodes_limit):
    if nodes_limit and nodes[0] >= nodes_limit:
        return evaluate(w, board)
        
    nodes[0] += 1
    stand_pat = evaluate(w, board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
        
    moves = list(board.generate_pseudo_legal_captures())
    moves = [m for m in moves if board.is_legal(m)]
    moves.sort(key=lambda m: move_score(board, m), reverse=True)
    
    for move in moves:
        board.push(move)
        score = -quiesce(w, board, -beta, -alpha, nodes, nodes_limit)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def alphabeta(w, board, depth, alpha, beta, nodes, nodes_limit):
    if nodes_limit and nodes[0] >= nodes_limit:
        return evaluate(w, board)
        
    nodes[0] += 1
    
    tt_key = board._transposition_key()
    tt_entry = TT.get(tt_key)
    tt_move = None
    
    if tt_entry and tt_entry[0] >= depth:
        tt_depth, tt_score, tt_flag, tt_move = tt_entry
        if tt_flag == 'EXACT':
            return tt_score
        elif tt_flag == 'LOWER':
            alpha = max(alpha, tt_score)
        elif tt_flag == 'UPPER':
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score

    if board.is_checkmate():
        return -MATE_SCORE + board.ply()
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0
    if depth == 0:
        return quiesce(w, board, alpha, beta, nodes, nodes_limit)
        
    moves = list(board.legal_moves)
    # Order by: 1. TT move (if exists), 2. MVV-LVA score
    moves.sort(key=lambda m: (m == tt_move, move_score(board, m)), reverse=True)
    
    best_score = -INF
    best_move = None
    alpha_orig = alpha
    
    for move in moves:
        board.push(move)
        score = -alphabeta(w, board, depth - 1, -beta, -alpha, nodes, nodes_limit)
        board.pop()
        
        if score > best_score:
            best_score = score
            best_move = move
            
        if score >= beta:
            TT[tt_key] = (depth, score, 'LOWER', best_move)
            return score
            
        if score > alpha:
            alpha = score
            
    if best_score <= alpha_orig:
        TT[tt_key] = (depth, best_score, 'UPPER', best_move)
    else:
        TT[tt_key] = (depth, best_score, 'EXACT', best_move)
        
    return best_score

def search(w, board, max_depth=5, movetime_ms=None, nodes_limit=None):
    best_move, best_score = None, -INF
    start  = time.time()
    nodes  = [0]
    
    # If a nodes limit is given, we can search deeper until we hit the node limit
    if nodes_limit:
        max_depth = 99
        
    for depth in range(1, max_depth + 1):
        d_best, d_score = None, -INF
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: (m == best_move, move_score(board, m)), reverse=True)
        
        for move in moves:
            if nodes_limit and nodes[0] >= nodes_limit:
                break
                
            board.push(move)
            score = -alphabeta(w, board, depth - 1, -INF, INF, nodes, nodes_limit)
            board.pop()
            if score > d_score:
                d_score, d_best = score, move
            if score > -INF:
                pass
                
        if nodes_limit and nodes[0] >= nodes_limit and d_best is None:
            # We ran out of nodes before evaluating anything at this depth
            break
            
        if d_best:
            best_move, best_score = d_best, d_score
            
        elapsed = int((time.time() - start) * 1000)
        nps     = int(nodes[0] / max(time.time() - start, 0.001))
        
        # Don't print stats every depth if we're just blazing through nodes
        if not nodes_limit or elapsed > 100:
            print(f'info depth {depth} score cp {best_score} nodes {nodes[0]} '
                  f'nps {nps} time {elapsed} pv {best_move.uci() if best_move else "0000"}', flush=True)
              
        if movetime_ms and elapsed >= movetime_ms * 0.8:
            break
        if nodes_limit and nodes[0] >= nodes_limit:
            break
            
    return best_move, best_score


# ──────────────────────────────────────────────────────────────────────
# UCI loop
# ──────────────────────────────────────────────────────────────────────

def debug_log(msg):
    print(f"[ENGINE_DEBUG] {msg}", file=sys.stderr, flush=True)

def uci_loop(weights):
    board       = chess.Board()
    max_depth   = 4
    
    debug_log("Engine started, entering UCI loop")
    
    print(f'id name {ENGINE_NAME} {ENGINE_VER}', flush=True)
    print('id author nnue-pytorch', flush=True)
    print('uciok', flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        debug_log(f"Received: {line}")
        
        if line == 'uci':
            print(f'id name {ENGINE_NAME} {ENGINE_VER}', flush=True)
            print('id author nnue-pytorch', flush=True)
            print('uciok', flush=True)
        elif line == 'isready':
            print('readyok', flush=True)
        elif line == 'ucinewgame':
            board = chess.Board()
            clear_tt()
        elif line.startswith('position'):
            board = chess.Board()
            parts = line.split()
            i = 1
            if i < len(parts) and parts[i] == 'fen':
                i += 1
                fen_parts = []
                while i < len(parts) and parts[i] != 'moves':
                    fen_parts.append(parts[i]); i += 1
                board = chess.Board(' '.join(fen_parts))
            elif i < len(parts) and parts[i] == 'startpos':
                i += 1
            if i < len(parts) and parts[i] == 'moves':
                i += 1
                for m in parts[i:]:
                    board.push_uci(m)
        elif line.startswith('go'):
            parts = line.split()
            movetime_ms = None
            wtime = btime = winc = binc = None
            depth_override = None
            nodes_limit = None
            i = 1
            while i < len(parts):
                t = parts[i]
                if t == 'movetime' and i+1 < len(parts): movetime_ms = int(parts[i+1]); i += 2
                elif t == 'depth'    and i+1 < len(parts): depth_override = int(parts[i+1]); i += 2
                elif t == 'nodes'    and i+1 < len(parts): nodes_limit = int(parts[i+1]); i += 2
                elif t == 'wtime'    and i+1 < len(parts): wtime = int(parts[i+1]); i += 2
                elif t == 'btime'    and i+1 < len(parts): btime = int(parts[i+1]); i += 2
                elif t == 'winc'     and i+1 < len(parts): winc = int(parts[i+1]); i += 2
                elif t == 'binc'     and i+1 < len(parts): binc = int(parts[i+1]); i += 2
                else: i += 1
            if movetime_ms is None and (wtime or btime):
                t = (wtime if board.turn == chess.WHITE else btime) or 10000
                inc = (winc if board.turn == chess.WHITE else binc) or 0
                movetime_ms = max(t // 20 + inc // 2, 50)
            d = depth_override if depth_override else max_depth
            best, _ = search(weights, board, max_depth=d, movetime_ms=movetime_ms, nodes_limit=nodes_limit)
            best_uci = best.uci() if best else "0000"
            debug_log(f"Sending bestmove: {best_uci}")
            print(f'bestmove {best_uci}', flush=True)
        elif line == 'quit':
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default=NNUE_PATH)
    args = parser.parse_args()
    print(f'Loading {args.net} ...', file=sys.stderr, flush=True)
    weights = load_nnue(args.net)
    print('Ready.', file=sys.stderr, flush=True)
    uci_loop(weights)

if __name__ == '__main__':
    main()
