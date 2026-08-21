import subprocess
import time
import threading

ENGINE_CMD = [
    '/home/adamz/Documents/praca-magisterska/.venv/bin/python3',
    '/home/adamz/Documents/praca-magisterska/uci_engine_optimized.py',
    '--net', '/home/adamz/Documents/praca-magisterska/nnue/control/epoch_300.nnue',
]

proc = subprocess.Popen(
    ENGINE_CMD,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Print stderr in background so crash messages are visible
def drain_stderr():
    for line in proc.stderr:
        print('  [STDERR]', line.rstrip(), flush=True)
threading.Thread(target=drain_stderr, daemon=True).start()

# Give the engine time to load the model before sending commands
print('Waiting for engine to load model (up to 30s)...', flush=True)
time.sleep(2)

def send(cmd):
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

def read_until(keyword, timeout=60.0):
    lines = []
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline().strip()
        if line:
            print('  <', line)
            lines.append(line)
        if keyword in line:
            break
    return lines

print('=== uci ===')
send('uci')
read_until('uciok', timeout=30)   # allow time to load model

print('\n=== isready ===')
send('isready')
read_until('readyok', timeout=10)

print('\n=== go movetime 2000 ===')
send('position startpos moves e2e4 e7e5')
send('go movetime 2000')
out = read_until('bestmove', timeout=60)

proc.stdin.close()
proc.terminate()
proc.wait()

ok = any('bestmove' in l for l in out)
print(f"\n>>> RESULT: {'✅ SUCCESS' if ok else '❌ FAILED'}")
