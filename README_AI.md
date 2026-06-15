# Opis pracy magisterskiej

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
    --validation-datasets /data/input/binpack/clean_1200_under_200k_test.binpack \
    --default-root-dir /data/trained_nets/ \
    --gpus 0 \
    --threads 4 \
    --batch-size 16384 \
    --lambda 0.3 \
    --max-epochs 100 \
    --features "HalfKAv2_hm^" \
    --network-save-period 10
```

Kontynuacja nauki:
`--resume-from-checkpoint /path/to/your/checkpoint.ckpt`

### Zamiana wyników uczenia na modele (nadal w kontenerze)

```bash
python serialize.py /data/trained_nets/lightning_logs/version_1/checkpoints/epoch=9-step=61040.ckpt /data/nnue/1200_under/epoch_10.nnue --features="HalfKAv2_hm^"
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

## Silnik

## Ocena modeli

## Wnioski

## Podsumowanie
