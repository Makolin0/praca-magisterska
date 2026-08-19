#!/usr/bin/env python3
"""Precise .nnue layout probe — works backwards from layer stacks."""
import struct
import numpy as np

NNUE = '/home/adamz/Documents/praca-magisterska/nnue/control/epoch_100.nnue'

def pad32(n): return n if n % 32 == 0 else n + 32 - n % 32
def ru32(f): return struct.unpack('<I', f.read(4))[0]

# ── Compute expected fc_hash ─────────────────────────────────────────
def compute_fc_hash(L1=1024, L2=31, L3=32):
    prev = (0xEC42E90D ^ (L1 * 2)) & 0xFFFFFFFF
    for out_per_bucket, has_relu in [(L2+1, True), (L3, True), (1, False)]:
        h = (0xCC03DAE4 + out_per_bucket) & 0xFFFFFFFF
        h ^= prev >> 1
        h ^= (prev << 31) & 0xFFFFFFFF
        if has_relu:
            h = (h + 0x538D24C7) & 0xFFFFFFFF
        prev = h
    return h

FC_HASH = compute_fc_hash()
print(f"Expected fc_hash: 0x{FC_HASH:08X}")

with open(NNUE, 'rb') as f:
    total = f.seek(0, 2); f.seek(0)

    # ── Header ───────────────────────────────────────────────────────
    _v = ru32(f); net_hash = ru32(f); dlen = ru32(f); f.read(dlen)
    print(f"Header ends at: {f.tell()}")
    _ft_hash = ru32(f)

    # ── Scan for fc_hash from end of file backwards ───────────────────
    # Layer stacks are at the very end. Scan for fc_hash occurrence.
    L1,L2,L3 = 1024,31,32
    per_bucket = (4 + (L2+1)*4 + (L2+1)*pad32(L1)
                    + L3*4 + L3*pad32(L2*2)
                    + 1*4  + 1*pad32(L3))
    ls_total = per_bucket * 8

    # Check if fc_hash is at (total - ls_total)
    candidate_pos = total - ls_total
    f.seek(candidate_pos)
    found_hash = ru32(f)
    print(f"\nAt pos {candidate_pos} (total-ls_total): 0x{found_hash:08X}  match={found_hash==FC_HASH}")

    # Try nearby positions
    for delta in range(-2000, 2001, 4):
        pos = candidate_pos + delta
        if pos < 0 or pos + 4 > total: continue
        f.seek(pos)
        h = ru32(f)
        if h == FC_HASH:
            print(f"fc_hash FOUND at pos {pos} (delta={delta:+d})")

    # ── FT section analysis ───────────────────────────────────────────
    # If fc_hash found at ls_start, FT section = [ft_hash_end .. ls_start]
    ft_end_pos = total - ls_total
    ft_start_pos = f.seek(0) or 96 + 4  # after header + ft_hash

    ft_section_bytes = ft_end_pos - ft_start_pos
    print(f"\nFT section: {ft_start_pos} .. {ft_end_pos} = {ft_section_bytes} bytes")

    # FT bias is int16 × L1 = 2048 bytes
    # Remaining = weights + PSQT
    weights_psqt = ft_section_bytes - L1 * 2
    print(f"Weights + PSQT: {weights_psqt} bytes")

    # Solve for T (total features, int8 weights):
    # T * L1  +  T * 8 * 4  =  weights_psqt
    # T * (L1 + 32) = weights_psqt
    T_int8 = weights_psqt / (L1 + 32)
    print(f"Solved T (int8, NUM_PSQT=8): {T_int8:.4f}")

    # Try NUM_PSQT = 1..16
    for np_ in range(1, 17):
        T = weights_psqt / (L1 + np_ * 4)
        if abs(T - round(T)) < 0.001:
            print(f"  EXACT match with NUM_PSQT={np_}: T={round(T)} features")

    # ── Read FT bias and first few weights ────────────────────────────
    f.seek(ft_start_pos)
    bias_int16 = np.frombuffer(f.read(L1 * 2), dtype=np.int16)
    print(f"\nFT bias (first 8, int16): {bias_int16[:8]}")
    print(f"FT bias range: [{bias_int16.min()}, {bias_int16.max()}]")

    # Peek at first 32 bytes of weight section
    raw = f.read(32)
    as_int8  = np.frombuffer(raw, dtype=np.int8)
    as_int16 = np.frombuffer(raw, dtype=np.int16)
    print(f"\nFirst 32 bytes after bias as int8:  {as_int8[:16]}")
    print(f"First 32 bytes after bias as int16: {as_int16[:8]}")
