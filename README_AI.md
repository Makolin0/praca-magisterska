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

## Droga do obecnego eksperymentu

Podstawowym pomysłem było rozwinięcie mojej pracy inżynierskiej poprzez wymianę podstawowego modelu Stockfish na model który uczyłby się razem z graczem, na podstawie wspólnie rozegranych partii. Przez co szybciej uczyłby się odpowiadać na powtarzane ataki gracza i zmuszać go do częstszej zmiany taktyki. Pomysł został odrzucony przez zbyt małą pulę gier potrzebną na naukę modelu.

Następna iteracha polegała na douczaniu modelu grami pobranymi z publicznych baz danych w momencie gdy gracz przekroczy pewien procent ostatnio wygranych partii.

Aby sprawdzić czy taka implementacja byłaby poprawnie działająca, przeprowadzam aktualny eksperyment sprawdzający jakość modelu na każdym etapie uczenia.

## Dane wejściowe

## Silnik

## Ocena modeli

## Wnioski

## Podsumowanie
