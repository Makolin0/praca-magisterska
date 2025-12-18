
# Eksperyment uczenia sieci stopniowo danymi lepszej jakości

## Przygotowanie środowiska

### baza danych gier

https://database.lichess.org

### pgn-extract

Program który filtruje bazę danych z grami szachowymi

Kompilacja programu

```bash
git clone https://github.com/kentdjb/pgn-extract
cd pgn-extract
make
```

macOS 

```bash
brew install cmake boost
```

```bash
git clone https://github.com/DanielUranga/trainingdata-tool.git
cd trainingdata-tool
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DCMAKE_C_FLAGS="-Wno-error=implicit-function-declaration -Wno-deprecated-non-prototype"
cmake --build .
```