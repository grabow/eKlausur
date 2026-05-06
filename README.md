# eKlausur2

Dieses Projekt ist ein eigenständiger, Python-basierter Evaluations-Workflow zur automatischen Klausurkorrektur.
Die fachliche und methodische Referenz ist das bestehende Projekt **eKlausur**.

## Ziel

Wir wollen die Erkennungsleistung verschiedener Ansätze vergleichbar evaluieren, insbesondere:
- YOLO-basierter Ansatz
- Foundation-Model-Ansätze (z. B. Gemini, OpenAI, weitere)
- Optional ein redundanter Zwei-Stufen-Ansatz (zweites Modell zur Validierung)

## Vorgehen

1. **Daten sammeln**
   - Einwilligungslisten (CSV/XLSX) auswerten.
   - Relevante Matrikelordner aus mehreren Quellverzeichnissen in ein gemeinsames Dataset-Verzeichnis kopieren.
   - Sensible Daten nur gemäß Freigabe und anonymisiert bereitstellen.
   - Lokales Arbeitsverzeichnis dafür: `./data/` (wird nicht eingecheckt, siehe `.gitignore`)
   - Anleitung/Beispiel: `data/README.md`

2. **Python-Skripte schreiben**
   - Skript für Datensammlung/Filterung (`collect_exam_data.py`).
   - Skript(e) für Erkennungsläufe (YOLO + Foundation Models) auf denselben Daten.
   - Optional Skript für Zwei-Stufen-Redundanz (Erkennung + Gegenprüfung durch zweites Modell).

3. **Auswerten**
   - Einheitliche Metriken berechnen (z. B. tokenweise Genauigkeit, Fehlerraten pro Klausurtyp).
   - Ergebnisse pro Modell und pro Klausurtyp ausgeben.
   - Vergleichstabellen/CSVs für Paper und Reproduzierbarkeit erzeugen.

## Referenz: eKlausur

Die technische Umsetzung hier ist bewusst unabhängig von der eKlausur-GUI.
Das methodische Vorgehen orientiert sich aber an eKlausur:
- Lokaler Referenzpfad: `/Users/wiggel/IntelliJIDEA/eKlausur`
- gleiche inhaltliche Aufgabenstellung,
- vergleichbare Eingabe-/Ausgabeformate,
- reproduzierbare Evaluationslogik.

## Laufzeit-Hinweis

Für Läufe bitte diesen Interpreter nutzen:
`/Users/wiggel/Python/eKlausur2/.venv/bin/python`

## GitHub Zugriff (SSH + GH CLI)

SSH fuer GitHub ist eingerichtet.
Test:

```powershell
ssh -T git@github.com
```

Erwartete Rueckmeldung:
`Hi grabow! You've successfully authenticated, but GitHub does not provide shell access.`

GitHub CLI ist installiert unter:
`C:\Program Files\GitHub CLI\gh.exe`

Falls `gh` im aktuellen Terminal nicht gefunden wird, ein neues Terminal oeffnen
oder direkt aufrufen:

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" --version
```
