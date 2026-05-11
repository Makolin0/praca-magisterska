
# Eksperyment uczenia sieci stopniowo danymi lepszej jakości

modyfikacja run docker w nnue-pytorch

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

## Przygotowanie środowiska

Osobiście pracowałem z:

- Ubuntu 24
- GPU rx 6800xt
- Python 3.12.3

Zależności

```bash
sudo apt-get update && sudo apt-get install -y zstd make g++
```

Środowisko python

- `setup_env.ipynb` - pobranie i kompilacja narzędzi


### baza danych gier

<https://database.lichess.org>

```bash
wget https://database.lichess.org/standard/lichess_db_standard_rated_2014-05.pgn.zst
unzstd lichess_db_standard_rated_2014-05.pgn.zst -o raw_db.pgn
```

### pgn-extract

Program filtrujący bazę danych z grami szachowymi

```bash
git clone https://github.com/kentdjb/pgn-extract
cd pgn-extract
make
cd ..
mv ./pgn-extract ./pgn-extract-repo
mv ./pgn-extract-repo/pgn-extract ./pgn-extract
```

### trainingdata-tool

Instalacja trainingdata-tool

```bash
git clone https://github.com/DanielUranga/trainingdata-tool.git
cd trainingdata-tool
git submodule update --init --recursive
mkdir -p build && cd build

cmake -DCMAKE_CXX_FLAGS="-include cstdint -fpermissive" -DCMAKE_CXX_STANDARD=14 ..
# cmake -DCMAKE_CXX_STANDARD=14 ..
cmake --build .
cd ../..
mv ./trainingdata-tool ./trainingdata-tool-repo
mv ./trainingdata-tool-repo/build/trainingdata-tool ./trainingdata-tool
```

### lczero-training

Specyficzna wersja protocol buffers

```bash
PROTOC_ZIP=protoc-3.12.4-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.12.4/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
sudo chmod +rx /usr/local/bin/protoc
rm -f $PROTOC_ZIP
```

instalacja

```bash
git clone https://github.com/LeelaChessZero/lczero-training
cd lczero-training
git submodule update --init --recursive

./init.sh
```

## preprocessing danych

```bash
./pgn-extract -7 -C -s --stopafter 100000 -o output.pgn input.pgn
./trainingdata-tool output.pgn
```

## Trenowanie

```bash
python ./lczero-training/tf/train.py --cfg config.yaml --output ./model.txt
```
