# Opis pracy magisterskiej

Curriculum Learning

## Główne informacje

- nazwa: Uczenie silnika szachowego na partiach o rosnącym Elo
- autor: Adam Zieliński
- promotor: prof Łukasz Mikulski

## Repozytoria

- [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) - program do pracy na plikach pgn, zapisu gier szachowych
- [stockfish nnue](https://github.com/official-stockfish/nnue-pytorch) - projekt pozwalający na uczenie własnych silników szachowych na architekturze NNUE
- [lichess database](https://database.lichess.org) - baza danych gier szachowych z której pobrałem dane uczące
- [stockfish](https://github.com/official-stockfish/Stockfish) - Główne repozytorium Stockfisha, głównie użyte do narzędzi oceniających ruchy dla danych uczących
- [cutechess](https://github.com/cutechess/cutechess) - ocena elo stworzonych silników
- [Książka otwarć](https://github.com/official-stockfish/books/blob/master/UHO_4060_v4.epd.zip)

## Droga do obecnego eksperymentu

Podstawowym pomysłem było rozwinięcie mojej pracy inżynierskiej poprzez wymianę podstawowego modelu Stockfish na model który uczyłby się razem z graczem, na podstawie wspólnie rozegranych partii. Przez co szybciej uczyłby się odpowiadać na powtarzane ataki gracza i zmuszać go do częstszej zmiany taktyki. Pomysł został odrzucony przez zbyt małą pulę gier potrzebną na naukę modelu.

Następna iteracha polegała na douczaniu modelu grami pobranymi z publicznych baz danych w momencie gdy gracz przekroczy pewien procent ostatnio wygranych partii.

Aby sprawdzić czy taka implementacja byłaby poprawnie działająca, przeprowadzam aktualny eksperyment sprawdzający zachowanie modelu podczas uczenia na zwiększających się jakościowo danych.

## Próby

Początkowo próbowałem użyć LeelaChessZero, natomiast przez spore problemy z uruchomieniem związane ze słabą dokumentacją poddałem się, i zmieniłem użyty silnik na Stockfish NNUE.

## Dane wejściowe

### rozdzielenie pliku na przedziały elo

```bash
cargo run -- ../data/input/raw/lichess_db_standard_rated_2026-03.pgn
```

### oczyszczenie plików

```bash
./pgn-extract -o ./data/input/clean/clean_1200_under_2mil.pgn --minply 40 -C -V -N --nobadresults --stopafter 2000000 --notags --nomovenumbers ./data/input/split/comp_1200_under.pgn
```

- `minply` - minimalna liczba ruchów
- `-C -v -N -7` - ignorowanie niepotrzebnych danych
- `--nobadresults` - filtruje popsute wyniki

### zamiana na plain

```bash
python ./sf_source_tools/script/pgn_to_plain.py --pgn ./data/input/clean/clean_1200_under_2mil.pgn
mv plain.txt ./data/input/plain/clean_1200_under_2mil.plain
```

### do binpack

```bash
./stockfish-tools convert ./data/input/plain/clean_1200_under_2mil.plain ./data/input/binpack/clean_1200_under_2mil.binpack
```

- 2 mln gier w każdym przedziale
- Dane zostały pobrane z <https://database.lichess.org>
  - 2026 - March: 90,074,196 rozgrywek

## Nauka

### uruchomienie środowiska do nauki

Znajdując się w folderze nnue-pytorch

modyfikujemy plik run_docker.sh zmieniając na końcu część:

```bash
docker run -it \
  $GPU_FLAGS \
  $USER_FLAG \
  --group-add render \
  --group-add video \
  --group-add kvm \
  -v "$(pwd)":/workspace/nnue-pytorch \
  -v "$DATA_PATH":/data \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $IMAGE_TAG
```

i uruchamiamy

```bash
./run_docker.sh
```

podajemy folder z danymi: `../data`

następnym razem wchodzimy do już istniejącego kontenera

### Rozpoczęcie nauki silnika (w kontenerze)

```bash
python train.py \
    /data/input/binpack/clean_1200_under_2mil.binpack \
    --validation-datasets /data/input/binpack/clean_1200_under_200k.binpack \
    --default-root-dir /data/trained_nets/ \
    --gpus 0 \
    --threads 4 \
    --batch-size 16384 \
    --lambda 0.3 \
    --max-epochs 100 \
    --features "HalfKAv2_hm^" \
    --network-save-period 10
```

fresh
```bash
python train.py \
        /data/input/binpack/clean_1200_under_2mil.binpack \
        --validation-datasets /data/input/binpack/clean_1200_under_200k.binpack \
        --features "HalfKAv2_hm^" \
        --default-root-dir /data/trained_nets/test/ \
        --max_epochs 1 \
        --gpus 1
```

Kontynuacja nauki:
`--resume-from-checkpoint /path/to/your/checkpoint.ckpt`

### Zamiana wyników uczenia na modele (nadal w kontenerze)

```bash
python serialize.py /data/trained_nets/lightning_logs/version_1/checkpoints/epoch=9-step=61040.ckpt /data/nnue/1200_under/epoch_10.nnue --features="HalfKAv2_hm^"
```

nowa wersja do przetestowania

```bash
python serialize.py \
        --features "HalfKAv2_hm" \
        /data/trained_nets/test/lightning_logs/version_0/checkpoints/last.ckpt \
        custom_net.nnue
```

### Stworzenie pełnego silnika na podstawie modelu

w katalogu stockfish-source/src

```bash
make clean
make -j$(nproc) profile-build COMP=gcc ARCH=x86-64-bmi2 EVALFILE=/home/adamz/Documents/praca-magisterska/data/nnue/1200_under/epoch_10.nnue
```

### Porównanie modeli ze sobą

w środku pliku `evaluate_nets.py` podmienić odpowiednie ścieżki i wykonać

```bash
python evaluate_nets.py
```

Skrypt porównuje sąsiadujące modele, epoch_10 z epoch_20, 20 z 30, 30 z 40 itd.

### przewidziany czas na obliczenia

ai mówi że powinno być minimum 400 epok, idealnie 800

kalkulacja czasu trwania nauki

- 1 epoka uczyła się 6 minut, 2 epoki dały podobny wynik
- 400 epok będzie uczyć się 40h
- Czy tyle powinno być na każdą grupę elo?
- Myślę że wystarczy 100 epok na grupę, wtedy sumarycznie będzie ich 600
- 600 epok to 60h nauki
- Zakładam że zajmie między 50-80h
- komputer zostawiony na 10 nocy max?

- 60h nauki x2 bo jeszcze do porównania 120h nauki
- Jeśli wezmę tylko 3 przedziały wstępnie, zajmie 60h nauczenie obydwu
- 2h zajmie wyliczenie zmian elo między silnikami dla 1 kubełka
- w sumie 12h zajmie liczenie zmian elo dla obydwu eksperymentów

## Ocena modeli

### Porównanie funkcji straty podczas uczenia

#### Uczenie postępowe (research) vs losowe (control)

Kontrolna:
=======================================================
        ANALIZA STABILNOŚCI I ZMIENNOŚCI LOSS        
=======================================================
Liczba przeanalizowanych kroków (steps) : 1499
-------------------------------------------------------
1. Odchylenie std. różnic (Std Dev of ΔLoss) : 0.001554
2. Średnia zmiana bezwzględna (Mean |ΔLoss|)  : 0.001298
3. Odchylenie szumu od EMA (Noise Volatility): 0.001137
4. Ogólne odchylenie std. (Total Loss Std)   : 0.002484
-------------------------------------------------------
Liczba wykrytych nagłych skoków (>3σ ΔLoss)  : 1
=======================================================

Badawcza:
=======================================================
        ANALIZA STABILNOŚCI I ZMIENNOŚCI LOSS        
=======================================================
Liczba przeanalizowanych kroków (steps) : 1499
-------------------------------------------------------
1. Odchylenie std. różnic (Std Dev of ΔLoss) : 0.001352
2. Średnia zmiana bezwzględna (Mean |ΔLoss|)  : 0.000644
3. Odchylenie szumu od EMA (Noise Volatility): 0.002060
4. Ogólne odchylenie std. (Total Loss Std)   : 0.006142
-------------------------------------------------------
Liczba wykrytych nagłych skoków (>3σ ΔLoss)  : 6
=======================================================

Wskaźnik	Grupa Kontrolna (Control)	Grupa Badawcza (Research)	Interpretacja
Noise Volatility (odchylenie od trendu)	0.001137	0.002060	+81% w próbie badawczej. Wskazuje na występowanie silnych wstrząsów i niestabilności powiązanych ze zmianą rozkładu danych.
Total Loss Std (całkowita zmienność)	0.002484	0.006142	+147% w próbie badawczej. Świadczy o braku monotonnej zbieżności i dużych rozstępach wartości funkcji błędu.
Mean |ΔLoss| (średnia zmiana krokowa)	0.001298	0.000644	W próbie badawczej błąd między sąsiednimi krokami wewnątrz danego kubełka jest bardzo stabilny (szum lokalny jest niższy).
Liczba wykrytych skoków	1	6	Odzwierciedla punktowe destabilizacje modelu w momencie wprowadzenia nowej grupy Elo.

"Analiza numeryczna funkcji straty potwierdziła istotne różnice w dynamice uczenia obu modeli. Choć linia błędu grupy kontrolnej charakteryzowała się wyższym lokalnym mikroszumem krok do kroku (Mean ∣ΔLoss∣=0.001298), to model badawczy wykazuje niemal dwukrotnie wyższą zmienność szumu względem trendu (Noise Volatility=0.002060) oraz 2.5-krotnie wyższe ogólne odchylenie standardowe błędu. Wynika to bezpośrednio z gwałtownych skoków funkcji straty w momentach przełączania zbiorów uczących (Co-Domain Shift), które lokalnie destabilizowały proces optymalizacji wag."

Na pierwszy rzut oka widać róznice w postaci skoków błędu przy uczeniu postępowym,
pojawiały się one dokładnie w momencie zmiany danych uczących, kontrastując z gładko
malejącym wykresem próbki kontrolnej. Wartość błędu między pojedynczymi krokami równiez
jest zauwazalnie nizsza.

Jak obliczyć jakąś liczbę która będzie wskazywać takie nizsze wahania funkcji błędu?

#### Uczenie postępowe (research) vs postępowe odwrócone

odwrócone:
=======================================================
        ANALIZA STABILNOŚCI I ZMIENNOŚCI LOSS        
=======================================================
Liczba przeanalizowanych kroków (steps) : 1501
-------------------------------------------------------
1. Odchylenie std. różnic (Std Dev of ΔLoss) : 0.001505
2. Średnia zmiana bezwzględna (Mean |ΔLoss|)  : 0.000784
3. Odchylenie szumu od EMA (Noise Volatility): 0.001988
4. Ogólne odchylenie std. (Total Loss Std)   : 0.005905
-------------------------------------------------------
Liczba wykrytych nagłych skoków (>3σ ΔLoss)  : 5
=======================================================

Wskaźnik,Próba Badawcza (Progresywna),Próba Odwrócona,Różnica (%)
Liczba kroków,1499,1501,—
Std Dev of ΔLoss (Odch. std. różnic),0.001352,0.001505,+11.3%
Mean ∥ΔLoss∥ (Średnia zmiana bezwzględna),0.000644,0.000784,+21.7%
Noise Volatility (Szum wokół trendu EMA),0.002060,0.001988,-3.5%
Total Loss Std (Całkowite odchylenie std.),0.006142,0.005905,-3.9%
Liczba wykrytych skoków (>3σ),6,5,-1 skok

Szegółowa interpretacja i wnioski do pracy
1. Wskaźniki makro (Total Loss Std oraz Noise Volatility) są niemal identyczne
Różnica w całkowitej zmienności (Total Loss Std≈0.0060) oraz odchyleniu od trendu (Noise Volatility≈0.0020) wynosi poniżej 4%. Ponadto obie próby wykazują zbliżoną liczbę gwałtownych skoków błędu (6 vs 5).
Wniosek naukoy: To bezpośredni dowód na to, że wstrząsy sieci nie wynikają z "poziomu trudności" danych (Elo), lecz z samego faktu podmiany rozkładu danych (Distribution Shift / Domain Shift). Model reaguje tak samo gwałtownie na przejście z gier słabych graczy do silnych, jak i na sytuację odwrotną.
2. Drobna różnica w mikroszumie (Mean ∥ΔLoss∥ oraz Std Dev of ΔLoss)
Próba odwrócona wykazuje lekko wyższy lokalny szum krok po kroku (+21.7% dla średniej bezwzględnej zmiany delty).
Wniosek naukowy: Przyczyną jest fakt, że w próbie odwróconej model zaczyna uczenie od gier wysokie Elo (>2000). Gry na wyższym poziomie są bardziej jednorodne pod względem taktycznym (mniej ewidentnych "podstawek", powtarzalne struktury debiutowe), a następnie przechodzi do gier niższych klas (<1200), które charakteryzują się bardzo wysoką wariancją i nieprzewidywalnością ruchów graczy. Wprowadzenie danych o wyższym stopniu losowości w późniejszej fazie uczenia zwiększa lokalny mikroszum optymalizatora.
Gotowy akapit do podrozdziału "Analiza Wyników"
Porównanie uczenia progresywnego i odwróconego:
Zestawienie wskaźników statystycznych funkcji straty dla próby badawczej (uczenie od najniższego do najwyższego Elo) oraz próby odwróconej (uczenie od najwyższego do najniższego Elo) wykazało niemal całkowitą symetrię obu procesów. Całkowite odchylenie standardowe błędu (Total Loss Std) wyniosło odpowiednio 0.006142 oraz 0.005905 (różnica 3.9%), natomiast wskaźnik zmienności szumu względem trendu (Noise Volatility) osiągnął wartości 0.002060 i 0.001988 (różnica 3.5%). W obu przypadkach odnotowano również zbliżoną liczbę drastycznych skoków błędu (6 vs 5).
Wyniki te jednoznacznie obalają hipotezę, jakoby gry graczy o wyższym rankingu Elo stanowiły dla sieci neuronowej problem o "wyższym stopniu trudności optymalizacyjnej". Skoki funkcji straty oraz chwilowa destabilizacja procesu uczenia są wyłącznie konsekwencją dyskontynuacji rozkładu danych uczących (Co-Domain Shift) w momencie przełączania zasobników PGN, niezależnie od kierunku tej zmiany. Podkreśla to, że do zachowania stabilności sieci podczas uczenia na danych z baz partyjnych konieczne jest stosowanie ciągłego mieszania danych (Data Replay) zamiast sekwencyjnej podmiany zbiorów.

Oczekiwałem ze przy zmianie z wyzszego modelu na nizszy, błąd uczenia spadnie, bądź urośnie
zauwazalnie słabiej.
Natomiast tutaj wykresy błędu są bardzo do siebie podobne, co wydaje mi się obala wcześniejszą tezę
przy porównywaniu do modelu kontrolnego.
Róznice w błędzie uczenia prawdopodobnie wywodzą się z samej próby dopasowania modelu
do danych uczących, a nie faktycznie trudniejszych do zrozumienia zachowań gracza im ma większe elo.

### Porównanie róznic elo sąsiadujących modeli

Wydaje mi się ze uzylem za słabych parametrów i porównania mają duzy przedział błędu.

## Wnioski

## Podsumowanie

# Test czy nnue działa w stockfishu

uci
setoption name EvalFile value /home/adamz/Documents/praca-magisterska/data/nnue_potentially_bad/research/epoch_10.nnue
isready
go nodes 1

setoption name EvalFile value /home/adamz/Documents/praca-magisterska/nn-9a0cc2a62c52.nnue

setoption name EvalFile value /home/adamz/Documents/praca-magisterska/data/test.nnue



ubuntu@c07398a4b902:/workspace/nnue-pytorch$ python - <<'PY'                                                      import torch                                             ckpt = torch.load('/data/trained_nets/1200_under/epoch=9-step=61040.ckpt', map_location='cpu')

print(ckpt.keys())

print("state_dict:")
for k,v in ckpt["state_dict"].items():
    print(k, v.shape)
PY
dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers'])
state_dict:
model.input.bias torch.Size([1032])
model.input.features.0.weight torch.Size([24576, 1032])
model.input.features.0.virtual_weight torch.Size([768, 1032])
model.layer_stacks.l1.linear.weight torch.Size([256, 1024])
model.layer_stacks.l1.linear.bias torch.Size([256])
model.layer_stacks.l1.factorized_linear.weight torch.Size([32, 1024])
model.layer_stacks.l1.factorized_linear.bias torch.Size([32])
model.layer_stacks.l2.linear.weight torch.Size([256, 62])
model.layer_stacks.l2.linear.bias torch.Size([256])
model.layer_stacks.output.linear.weight torch.Size([8, 32])
model.layer_stacks.output.linear.bias torch.Size([8])



# Fix docker numpy

## 1. Take ownership of /opt/venv inside the running container
docker exec -u 0 -it 053b65690497 chown -R nnue_user /opt/venv

## 2. Or directly install numpy < 2.5 as root inside the container
docker exec -u 0 -it 053b65690497 /opt/venv/bin/pip install "numpy<2.5"

---

# PyNNUE — Custom Python UCI Engine (AI-assisted development)

This section documents all scripts created to run a custom-trained `.nnue` network as a fully functional UCI chess engine, without relying on Stockfish. All files are located in the project root (`/home/adamz/Documents/praca-magisterska/`).

---

## `uci_engine.py` — Main UCI Chess Engine

### What it does

A standalone Python UCI chess engine that loads a custom `.nnue` file produced by `nnue-pytorch` and uses it to evaluate chess positions. It implements a full alpha-beta search with quiescence search and communicates over stdin/stdout using the standard UCI protocol, making it compatible with any UCI-compliant GUI (CuteChess, Arena, etc.).

The engine has **zero dependencies on nnue-pytorch** — it reads the binary `.nnue` format directly, decoding LEB128-compressed tensors manually.

### Architecture implemented

Matches the trained model (`HalfKAv2_hm` feature set, 8 layer stacks):

| Parameter | Value |
|---|---|
| `L1` | 1024 — feature transformer outputs per side |
| `L2` | 31 — first hidden layer outputs (32 stored, last = skip) |
| `L3` | 32 — second hidden layer outputs |
| `NUM_LS` | 8 — layer stack buckets (by piece count) |
| `NUM_PSQT` | 8 — PSQT output buckets |
| `FT_SCALE` | 255.0 — `ft_quantized_one` |
| `FC_SCALE_H` | 64.0 — hidden weight scale |
| `FC_SCALE_OUT` | 16.0 — output weight scale |
| `NNUE2SCORE` | 600.0 — converts raw output to centipawns |

### CLI parameters

```bash
python3 uci_engine.py                          # uses hardcoded NNUE_PATH
python3 uci_engine.py --net /path/to/net.nnue  # override network file
```

### UCI commands supported

`uci`, `isready`, `ucinewgame`, `position [startpos|fen] [moves ...]`, `go [movetime N] [depth N] [wtime/btime/winc/binc]`, `quit`

### Forward pass pipeline

1. **Feature extraction** (`HalfKAv2_hm` export format, stride 704 per king bucket):
   - All non-king pieces (p_idx 0–9): `bucket * 704 + p_idx * 64 + orient(sq)`
   - **Own king**: `bucket * 704 + 10*64 + o_ksq`
   - **Opponent king**: `bucket * 704 + 10*64 + orient(opp_king_sq)`
2. **Feature transformer**: sum of active weight rows + bias, for both sides
3. **SqrCReLU activation**: `clamp(cat([us_ft, them_ft]), 0, 1)`, split into 4×512 chunks, pairwise products, scale `×(127/128)`
4. **Layer stack** (bucket selected by piece count):
   - L1: raw → `clamp(x²×255/256, 0,1)` (sqr) and `clamp(x, 0,1)` (lin), concat → 62 outputs + 1 skip
   - L2: linear → `clamp(x, 0,1)` → 32 outputs
   - Output: linear → 1 scalar
5. **Final score**: `(net_out + skip + psqt_contrib) × 600` centipawns

---

## `test_nnue_load.py` — Engine Integration Test

### What it does

Launches `uci_engine.py` as a subprocess, communicates over UCI protocol, and verifies that the engine:
1. Starts and responds to `uci` with `uciok`
2. Responds to `isready` with `readyok`
3. Produces a `bestmove` response for the position after `1.e4 e5`

Prints all engine output (stdout + stderr) and reports `✅ SUCCESS` or `❌ FAILED`.

### Configuration

Edit lines 5–9 to change the Python binary, engine path, or network file:

```python
ENGINE_CMD = [
    '/path/to/python3',
    '/path/to/uci_engine.py',
    '--net', '/path/to/epoch_N.nnue',
]
```

### Usage

```bash
python3 test_nnue_load.py
```

---

## `probe_nnue.py` — Binary Format Analyser

### What it does

Reads a `.nnue` file and prints its internal layout: version, hash, description, feature transformer hash, and section sizes. Used to reverse-engineer the exact binary format before writing the loader.

Key diagnostics performed:
- Computes expected `fc_hash` from the model architecture and confirms it matches the bytes at the start of the layer stacks section
- Solves for the number of feature rows (`T`) given the file size
- Checks for LEB128 magic bytes at expected positions
- Reports first 32 bytes after the FT bias in both `int8` and `int16` interpretation

### Usage

```bash
python3 probe_nnue.py   # path hardcoded at top of file
```

---

## `debug_eval.py` — Evaluation Correctness Checker

### What it does

Loads the NNUE weights, evaluates all legal white moves from the position after `1.e4 e5`, and prints each move's score from the side-to-move (black's) perspective. Used to verify that bad moves (e.g., `e1e2` — king to e2) score strongly positive (black winning) and good moves (e.g., `g1f3`, `d2d4`) score negative (white better).

### Usage

```bash
python3 debug_eval.py
```

Expected output (post-fixes):
```
e1e2: +43 cp  (white move value = -43 cp)  ← black winning = correct
g1f3: -18 cp  (white move value = +18 cp)  ← white better = correct
d2d4: -25 cp  (white move value = +25 cp)  ← white better = correct
```

---

## Development Challenges

### 1. Stockfish incompatibility
The `nnue-pytorch` trainer produces nets with `L2=31, FC_0_OUTPUTS=32`, while Stockfish 15.1 expects `FC_0_OUTPUTS=15`. This caused silent evaluation hash mismatches. Compiling Stockfish with `EVALFILE=epoch_100.nnue` appeared to succeed but always loaded the default net due to this mismatch. The fix was to abandon Stockfish entirely.

### 2. nnue-pytorch import chain — missing `cupy`, `lightning`
Attempting to import the Python model directly required `lightning`, `tyro`, and `cupy` (a CUDA-only GPU kernel library). Since the user's AMD GPU (RX 6800 XT) has no cupy support and inference doesn't need training-specific code, the solution was to write a completely standalone reader that parses the binary file directly without any nnue-pytorch dependency.

### 3. LEB128 compression
The `.nnue` file used LEB128 compression for the feature transformer section (bias, weights, PSQT). This was discovered by noticing that the "FT bias values" read as `int16` decoded to the ASCII string `"COMPRESSED_LEB128"`:

```
0x4F43 = 20291  →  'C', 'O'
0x504D = 20557  →  'M', 'P'  ... (spells out COMPRESSED_LEB128)
```

A pure-Python signed LEB128 decoder was implemented (no `numba` dependency).

### 4. Feature index format mismatch (training vs export)
`HalfKAv2_hm` stores weights in **export format** (stride 704 per king bucket, 11 piece types × 64 = 704) but the feature index function initially used **training format** (stride 768, 12 piece types). This caused `IndexError: index 24319 is out of bounds for size 22528`. Fix: change stride from 768 → 704.

### 5. Missing king features
The most subtle bug: both the own king and opponent king are **active features** in HalfKAv2_hm (`MAX_ACTIVE_FEATURES = 32` = 30 non-king pieces + 2 kings). Skipping them meant the network couldn't distinguish safe from exposed king positions — `e1e2` (king to centre) scored identically to `g1f3` (Nf3). In the export format, both kings share the `p_idx=10` block (offset `10*64`), rather than opponent king being at `p_idx=11` (which would overflow the 704-per-bucket layout).

### 6. SqrCReLU activation order
The l1 squared activation must be computed as `clamp(x² × 255/256, 0, 1)` — squaring the **raw** pre-activation value, then clamping. Clamping first and then squaring (`clamp(x)²`) is incorrect for negative inputs and drops the `255/256` scale factor.

### 7. PSQT factor
The model computes PSQT as `(wpsqt − bpsqt) × (us − 0.5)`, where `us = 1` for white and `0` for black. This gives a factor of `±0.5`, not `±1`. Using `±1` doubled the PSQT contribution.