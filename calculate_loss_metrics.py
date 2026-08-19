import pandas as pd
import numpy as np


def analyze_loss_stability(csv_file_path):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(csv_file_path)

    # Odrzucamy wiersze, które nie mają wartości 'train_loss' (np. podsumowania epok)
    df_clean = df.dropna(subset=['train_loss']).copy()
    
    # Upewniamy się, że dane są posortowane chronologicznie po kroku (step)
    df_clean = df_clean.sort_values('step').reset_index(drop=True)

    loss = df_clean['train_loss']

    # 1. Różnice krok po kroku: delta_t = L_t - L_{t-1}
    df_clean['loss_delta'] = loss.diff()

    # 2. Wykładniczo ważona średnia krocząca (EMA) dla wygładzenia trendu (span=20 kroków)
    df_clean['loss_ema'] = loss.ewm(span=20, adjust=False).mean()

    # 3. Szum/Odchylenie od trendu: Noise_t = L_t - EMA_t
    df_clean['loss_noise'] = loss - df_clean['loss_ema']

    # --- OBLICZANIE WSKAŹNIKÓW AGREGOWANYCH ---

    # A. Odchylenie standardowe różnic kroków (Std Dev of Step Differences)
    # Mierzy lokalną skokowość/szorstkość wykresu. Im niższa, tym gładszy wykres.
    std_delta = df_clean['loss_delta'].std()

    # B. Średni błąd bezwzględny różnic (Mean Absolute Step Change)
    mean_abs_delta = df_clean['loss_delta'].abs().mean()

    # C. Odchylenie standardowe szumu względem trendu (EMA Noise Volatility)
    # Pokazuje, jak bardzo surowy loss "drży" wokół swojego głównego trendu.
    std_noise = df_clean['loss_noise'].std()

    # D. Całkowite odchylenie standardowe surowej funkcji straty (Total Loss Std Dev)
    std_total = loss.std()

    # Wyszukanie skoków (outlierów) – np. zmian większych niż 3-krotność odchylenia standardowego delty
    delta_threshold = 3 * std_delta
    spikes = df_clean[df_clean['loss_delta'].abs() > delta_threshold]

    # --- WYŚWIETLENIE WYNIKÓW ---
    print("=" * 55)
    print("        ANALIZA STABILNOŚCI I ZMIENNOŚCI LOSS        ")
    print("=" * 55)
    print(f"Liczba przeanalizowanych kroków (steps) : {len(df_clean)}")
    print("-" * 55)
    print(f"1. Odchylenie std. różnic (Std Dev of ΔLoss) : {std_delta:.6f}")
    print(f"2. Średnia zmiana bezwzględna (Mean |ΔLoss|)  : {mean_abs_delta:.6f}")
    print(f"3. Odchylenie szumu od EMA (Noise Volatility): {std_noise:.6f}")
    print(f"4. Ogólne odchylenie std. (Total Loss Std)   : {std_total:.6f}")
    print("-" * 55)
    print(f"Liczba wykrytych nagłych skoków (>3σ ΔLoss)  : {len(spikes)}")
    print("=" * 55)

    return {
        "std_delta": std_delta,
        "mean_abs_delta": mean_abs_delta,
        "std_noise": std_noise,
        "std_total": std_total,
        "spikes_count": len(spikes)
    }


if __name__ == "__main__":
    # Zastąp 'twoj_plik.csv' ścieżką do swojego pliku
    metrics = analyze_loss_stability('graphs/train/reverse/metrics_joined.csv')