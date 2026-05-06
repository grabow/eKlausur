## `data/dataset/` – Ergebnis der Datensammlung (anonymisierte Datasets)

## Aktueller Stand (Dataset)

Unter `data/dataset/` liegen **61** anonymisierte Datasets in nummerierten Unterordnern `1/` bis `61/`.

Jeder Dataset-Ordner enthält typischerweise:

- `page_0.jpg ... page_{n-1}.jpg` (n ist **12** oder **13**)
- `studSolution.txt` – Ground Truth: **Zeile i** entspricht **Bild `page_{i-1}.jpg`**; Tokens sind `A–Z` oder `?` (`?` = unbekannt/leer)
- `structure.yaml` – Seiten-/Aufgabenstruktur; Invariante: `seite: x` ↔ `page_{x-1}.jpg`
- `result.txt` – anonymisiertes Korrektur-Resultat (ohne Matrikelnummer/Points-Header)
