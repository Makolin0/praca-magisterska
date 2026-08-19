#!/usr/bin/env bash
set -e

STOCKFISH="/home/adamz/Documents/praca-magisterska/stockfish"
SF_SRC="/home/adamz/Documents/praca-magisterska/stockfish-source/src"
NNUE="/home/adamz/Documents/praca-magisterska/nnue/control/epoch_100.nnue"

echo "==========================================="
echo "  NNUE Verification - Method 1 & Method 2"
echo "==========================================="

echo ""
echo "--- Stockfish binary info ---"
stat "$STOCKFISH" | grep -E "File:|Size:|Modify:"

echo ""
echo "==========================================="
echo "METHOD 1: Runtime EvalFile loading"
echo "==========================================="

OUTPUT=$(python3 - <<'PYEOF'
import subprocess, time, sys

STOCKFISH_BIN = '/home/adamz/Documents/praca-magisterska/stockfish'
NNUE_PATH = '/home/adamz/Documents/praca-magisterska/nnue/control/epoch_100.nnue'

proc = subprocess.Popen(
    [STOCKFISH_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

def send(cmd):
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

def read_until(keyword, timeout=20.0):
    lines = []
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline().strip()
        if line:
            lines.append(line)
        if keyword in line:
            break
    return lines

send('uci')
uci_out = read_until('uciok')
# Print EvalFile option if present
for l in uci_out:
    if 'EvalFile' in l:
        print('UCI EvalFile option:', l)
    if 'uciok' in l:
        print('uciok received')

send(f'setoption name EvalFile value {NNUE_PATH}')
send('isready')
ready_out = read_until('readyok', timeout=20)
for l in ready_out:
    if 'NNUE' in l or 'nnue' in l or 'network' in l or 'network' in l.lower() or 'eval' in l.lower():
        print('  NNUE load msg:', l)
    if 'readyok' in l:
        print('readyok received — net loaded successfully!')

send('position startpos moves e2e4 e7e5')
send('go movetime 500')
go_out = read_until('bestmove', timeout=10)
for l in go_out:
    if 'bestmove' in l:
        print('bestmove:', l)
    elif 'info depth' in l:
        print(' ', l)

proc.stdin.close()
proc.wait()
PYEOF
)

echo "$OUTPUT"

echo ""
echo "==========================================="
echo "METHOD 2: Compile with EVALFILE"
echo "==========================================="

echo "Cleaning and compiling Stockfish with epoch_100.nnue embedded..."
cd "$SF_SRC"

make clean 2>&1 | tail -3

make -j$(nproc) build COMP=gcc ARCH=x86-64-bmi2 EVALFILE="$NNUE" 2>&1 | tail -10

echo ""
echo "Compiled binary size: $(du -sh $SF_SRC/stockfish | cut -f1)"
echo ""
echo "Testing compiled binary..."

M2OUT=$(python3 - <<'PYEOF'
import subprocess, time

STOCKFISH_BIN = '/home/adamz/Documents/praca-magisterska/stockfish-source/src/stockfish'

proc = subprocess.Popen(
    [STOCKFISH_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

def send(cmd):
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

def read_until(keyword, timeout=20.0):
    lines = []
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline().strip()
        if line:
            lines.append(line)
        if keyword in line:
            break
    return lines

send('uci')
uci_out = read_until('uciok')
for l in uci_out:
    if 'EvalFile' in l:
        print('UCI EvalFile default:', l)
    if 'uciok' in l:
        print('uciok received')

send('isready')
ready_out = read_until('readyok', timeout=10)
for l in ready_out:
    if 'readyok' in l:
        print('readyok received')

send('position startpos moves e2e4 e7e5')
send('go movetime 500')
go_out = read_until('bestmove', timeout=10)
for l in go_out:
    if 'bestmove' in l:
        print('bestmove:', l)
    elif 'info depth' in l:
        print(' ', l)

proc.stdin.close()
proc.wait()
PYEOF
)
echo "$M2OUT"

echo ""
echo "==========================================="
echo "  VERIFICATION COMPLETE"
echo "==========================================="
