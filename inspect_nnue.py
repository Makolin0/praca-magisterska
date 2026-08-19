import struct

def inspect_nnue(file_path):
    with open(file_path, "rb") as f:
        # Read first 4 bytes (Magic Number)
        magic = struct.unpack("<I", f.read(4))[0]
        # Read Hash / Version
        arch_hash = struct.unpack("<I", f.read(4))[0]
        
        print(f"Magic Number: 0x{magic:08x}")
        print(f"Arch Hash:    0x{arch_hash:08x}")

inspect_nnue("/home/adamz/Documents/praca-magisterska/data/nnue_potentially_bad/research/epoch_10.nnue")