# Results-Konvention

Diese Datei definiert verbindlich, wie Inferenz-Ergebnisse gespeichert, benannt und zusammengefasst werden.

## 1) Ordnerstruktur

Ergebnisse werden unter `Results/<Model-Name>/` abgelegt.

Beispiel:
- `Results/YOLOv5 Medium/`

Innerhalb des Modellordners:
- pro Dataset ein Unterordner mit numerischer ID (`1`, `2`, ..., `61`)
- pro Dataset die Ergebnisdatei, z. B. `recognition_seed_det.txt`
- eine globale Zusammenfassung als `_summary.txt`

Beispiel:
- `Results/YOLOv5 Medium/1/recognition_seed_det.txt`
- `Results/YOLOv5 Medium/61/recognition_seed_det.txt`
- `Results/YOLOv5 Medium/_summary.txt`

## 2) Bewertungsregel (Token-basiert)

Die Auswertung erfolgt gegen `studSolution.txt` tokenweise (Buchstabenpunkte).

Verbindliche Regel:
- Wenn die Vorhersagezeile laenger ist als die Sollzeile, werden nur die ersten `x` Tokens bewertet, wobei `x = Anzahl Soll-Tokens`.
- Zusatztokens hinter `x` werden **nicht** als direkte Fehler in `letters_wrong` gezaehlt, sondern separat als `extra_pred_tokens` berichtet.

## 3) Pflichtfelder in `_summary.txt`

Jede Zusammenfassung muss mindestens folgende Felder enthalten:

- `model`
- `source_file`
- `datasets_expected`
- `datasets_copied`
- `pages_processed`
- `letters_total`
- `letters_correct`
- `letters_wrong`
- `accuracy`
- `accuracy_percent`
- `extra_pred_tokens`
- `created_at`

## 4) Kennzahlen-Definitionen

- `letters_total`: Gesamtzahl Soll-Tokens aus allen bewerteten `studSolution.txt`-Zeilen.
- `letters_correct`: Anzahl Positionen, an denen Soll-Token == Vorhersage-Token.
- `letters_wrong`: `letters_total - letters_correct`.
- `accuracy`: `letters_correct / letters_total`.
- `accuracy_percent`: `accuracy * 100`.
- `pages_processed`: Anzahl bewerteter Seiten (Zeilenpaare aus Soll/Prediction).
- `extra_pred_tokens`: Anzahl zusaetzlicher Vorhersage-Tokens hinter Soll-Laenge.

## 5) Modellnamen fuer Ergebnisordner

Der Modellordner unter `Results/` muss eindeutig sein, z. B.:

- `YOLOv5 Medium`
- `YOLOv5 Large`
- `YOLO26 Medium`
- `YOLO26 Large`

Damit bleibt klar nachvollziehbar, welches Modell welche Ergebnisse erzeugt hat.

## 6) Reproduzierbarkeit

Empfohlen:
- Inferenzskript, Modellpfad und Ausgabe-Dateiname in der jeweiligen Laufdoku festhalten (z. B. `yolov5_m.md`).
- Bei Publikationslaeufen die gleiche Bewertungsregel wie oben unveraendert anwenden.
