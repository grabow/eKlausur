# Results

## Scope

Vergleich der Inferenz auf `data/dataset/1..61` (insgesamt 742 Seiten, 3141 Tokens) mit:
- YOLOv5 Medium (Seed-Modell)
- YOLO26 Medium (Modell `yolo26m_20260506_best.pt`)

## YOLOv5 Medium

- `letters_total`: 3141
- `letters_correct`: 2856
- `letters_wrong`: 285
- `accuracy_percent`: 90.93
- `extra_pred_tokens`: 24

Quelle:
- `/Users/wiggel/Python/eKlausur2/Results/YOLOv5 Medium/_summary.txt`

## YOLO26 Medium (nach Bugfix)

- `letters_total`: 3141
- `letters_correct`: 2801
- `letters_wrong`: 340
- `accuracy_percent`: 89.18
- `extra_pred_tokens`: 6

Quelle:
- `/Users/wiggel/Python/eKlausur2/Results/YOLO26 Medium/_summary.txt`

## Root Cause Analysis (YOLO26)

Verbleibender Gap zu YOLOv5:
- YOLO26 liegt bei 89.18% vs. YOLOv5 bei 90.93%.
- Fehlerbild YOLO26 ist stark von `?`-Ausgaben geprägt.
- Haupttreiber: `x` (crossed_out) wird häufig als bester Kandidat pro Digit gewählt und anschließend zu `?` gemappt.

Messwerte aus Ablation:
- Baseline (aktuell): `89.18%` (`2801/3141`)
- `prefer_non_x` (Analysemodus): `90.80%` (`2852/3141`)
- Zugewinn: `+51` korrekte Tokens

Interpretation:
- Das Problem ist nicht primär „keine Klasse erkannt“, sondern die Auswahl von `x` in ambigen Fällen.

## Dataset Balance Hinweis

`x` (crossed_out) und `f` sind im Trainings-CSV klar unterrepräsentiert:
- pro A-Z Klasse: ca. 2400 Samples
- `f`: 240 Samples
- `x`: 250 Samples

Das kann zu Instabilität bei diesen Randklassen beitragen.

## Decision

Stand jetzt:
- Keine künstliche Aufblähung der `x`-Klasse.
- Ergebnisse und Analyse werden dokumentiert.
- Optionaler nächster Hebel bleibt die Inferenzregel (`prefer_non_x`), falls für Publikationslauf gewünscht.
