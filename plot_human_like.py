import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)\.nnue', filename)
    if match:
        return int(match.group(1))
    return 0

def plot_human_like():
    control_csv = "graphs/elo/human_like_results_control.csv"
    research_csv = "graphs/elo/human_like_results_research.csv"
    
    if not os.path.exists(control_csv) or not os.path.exists(research_csv):
        print("Błąd: Pliki CSV nie istnieją w folderze graphs/elo/")
        return
        
    df_control = pd.read_csv(control_csv)
    df_research = pd.read_csv(research_csv)
    
    df_control['Epoch'] = df_control['Network'].apply(extract_epoch)
    df_research['Epoch'] = df_research['Network'].apply(extract_epoch)
    
    df_control = df_control.sort_values('Epoch')
    df_research = df_research.sort_values('Epoch')
    
    plt.figure(figsize=(10, 6))
    
    # Plot control
    plt.plot(df_control['Epoch'], df_control['Match_Percentage'], label='Grupa kontrolna (0-1800 Elo)', color='blue', marker='o', markersize=4)
    
    # Plot research segments based on the bucket changes
    df_r1 = df_research[df_research['Epoch'] <= 100]
    df_r2 = df_research[(df_research['Epoch'] > 100) & (df_research['Epoch'] <= 200)]
    df_r3 = df_research[df_research['Epoch'] > 200]
    
    plt.plot(df_r1['Epoch'], df_r1['Match_Percentage'], label='Grupa badawcza (test: 1200 Elo)', color='red', marker='s', markersize=4)
    plt.plot(df_r2['Epoch'], df_r2['Match_Percentage'], label='Grupa badawcza (test: 1200-1500 Elo)', color='darkred', marker='s', markersize=4)
    plt.plot(df_r3['Epoch'], df_r3['Match_Percentage'], label='Grupa badawcza (test: 1500-1800 Elo)', color='indianred', marker='s', markersize=4)
    
    # Draw vertical lines to show bucket changes
    plt.axvline(x=105, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=205, color='gray', linestyle='--', alpha=0.7)
    
    plt.text(105, plt.ylim()[1]*0.98, ' Zmiana koszyka', rotation=90, va='top', color='gray')
    plt.text(205, plt.ylim()[1]*0.98, ' Zmiana koszyka', rotation=90, va='top', color='gray')
    
    plt.title('Zgodność z ruchami ludzkimi (Human-like Index)')
    plt.xlabel('Epoka treningu')
    plt.ylabel('Zgodność przewidywań (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    out_path = "graphs/elo/human_like_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Wykres zapisano do {out_path}")

if __name__ == "__main__":
    plot_human_like()
