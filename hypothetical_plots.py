import matplotlib.pyplot as plt
import numpy as np

# Konfiguracja stylu
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Oś X: Epoki (0 - 100)
epochs = np.linspace(0, 100, 500)

# --- MODEL KONTROLNY (Logistyczny wzrost z szumem) ---
# Uczy się powoli, na początku ma trudności z chaosem danych
control_elo = 400 + 1800 * (1 / (1 + np.exp(-0.05 * (epochs - 50))))
noise = np.random.normal(0, 30, size=epochs.shape)
control_elo_noisy = control_elo + noise

# --- MODEL BADACZY (Schodkowy / Curriculum Learning) ---
# Szybszy start na prostych danych, wyraźne skoki przy zmianie koszyka
research_elo = np.zeros_like(epochs)
for i, e in enumerate(epochs):
    if e < 25: # Koszyk 1: 800-1200 Elo
        research_elo[i] = 400 + 600 * (1 - np.exp(-0.15 * e))
    elif e < 50: # Koszyk 2: 1200-1600 Elo
        research_elo[i] = 1000 + 500 * (1 - np.exp(-0.12 * (e - 25)))
    elif e < 75: # Koszyk 3: 1600-2000 Elo
        research_elo[i] = 1500 + 600 * (1 - np.exp(-0.10 * (e - 50)))
    else: # Koszyk 4: 2000+ Elo
        research_elo[i] = 2100 + 400 * (1 - np.exp(-0.08 * (e - 75)))

# Rysowanie wykresów
plt.plot(epochs, control_elo_noisy, label='Model Kontrolny (Dane losowe)', 
         color='gray', alpha=0.5, linestyle='--')
plt.plot(epochs, research_elo, label='Model Badawczy (Curriculum Learning)', 
         color='#2ecc71', linewidth=3)

# Dodanie pionowych linii oznaczających zmianę koszyka danych
for step in [25, 50, 75]:
    plt.axvline(x=step, color='red', linestyle=':', alpha=0.3)
    plt.text(step-1, 2400, f'Zmiana\nkoszyka', color='red', 
             fontsize=9, ha='right', va='top', rotation=90)

# Opis osi i tytuł
plt.title('Hipotetyczna progresja rankingu Elo w czasie uczenia', fontsize=14, pad=20)
plt.xlabel('Epoki / Czas uczenia', fontsize=12)
plt.ylabel('Ranking Elo (Siła gry)', fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True)

# Zakresy osi
plt.ylim(300, 2600)
plt.xlim(0, 100)

plt.tight_layout()
plt.show()