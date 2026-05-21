
# Uczenie silnika szachowego na partiach o rosnącym Elo

## Środowisko uczenia

- Ubuntu 24
- Python 3.12
- GPU RX 6800xt

### Uzyte programy

- [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
- [stockfish nnue](https://github.com/official-stockfish/nnue-pytorch)
- [lichess database](https://database.lichess.org)

### Przygotowanie środowiska

```bash
setup_env.ipynb
```

## Przygotowanie danych uczących

### Pobranie danych

```bash
wget https://database.lichess.org/standard/lichess_db_standard_rated_[year]_[month].pgn.zst
```

### Filtracja

Dla kazdego wygenerowanego pliku uzywamy pgn-extract

### Transformacja

```bash
python pgn_to_plain.py ...
```

```bash
python serialize
```
