# ToDo Publikation

## Story

Kernaussage:
- Foundation Models sind fuer die Handschrifterkennung in diesem Klausur-Setup praktisch brauchbar und den lokalen YOLO-Modellen klar ueberlegen.
- Der wichtigste praktische Fehler ist nicht nur Token-Accuracy, sondern die Korrekturfairness:
  - FN: korrekte studentische Antwort wird als falsch erkannt; benachteiligt Studierende.
  - FP: falsche studentische Antwort wird als korrekt erkannt; uebervorteilt Studierende.
- `weak` ist als Hauptverfahren plausibel, weil es FN stark reduziert.
- Die verbleibenden FP-Faelle sind selten und wirken zugunsten der Studierenden.
- Ein optionaler `plain`-Konsistenzcheck kann Hint-Bias sichtbar machen, ist aber wegen des geringen FP-Niveaus vermutlich nicht zwingend noetig.
- Praktische Absicherung: Studierende erhalten die korrigierte Auswertung zur Selbstkontrolle. FN-Faelle koennen reklamiert werden; nach Beispiel-Notentabelle waeren beim Gemini-Run nur ca. 3 von 61 Klausuren notenrelevant betroffen.

Kurzform fuer Paper:
- Foundation Models outperform YOLO baselines.
- Weak-hint inference yields very low false-negative rates.
- Remaining false positives are rare and student-favorable.
- Full automation is feasible, with a small expected appeal workload through student self-review.

## Bereits belegter Gemini-Stand

Modell:
- `LLM OpenRouter Gemini-3.1 Flash-Lite`

Vollstaendige Laeufe:
- `plain`: 61/61 Datasets
- `weak`: 61/61 Datasets

Wichtige Zahlen fuer `weak only`:
- TP: 2380
- FN: 14
- FP: 11
- TN: 736
- FNR: 0.58%
- FP-Rate bezogen auf alle Antwortpositionen: 11 / 3141 = 0.35%
- FP-Rate bezogen auf tatsaechlich falsche Antworten: 11 / 747 = 1.47%

Praktische Folgen:
- 14 FN-Faelle betreffen 8 von 61 Klausuren.
- Mit Beispiel-Notentabelle waeren davon 3 Klausuren notenrelevant schlechter bewertet.
- Erwarteter Reklamationsaufwand bei Selbstkontrolle: ca. 3 Faelle pro 61 Klausuren.
- 11 FP-Faelle bedeuten kleine Uebervorteilung zugunsten der Studierenden.

Optionaler `plain`-Konsistenzcheck:
- Konflikte: 52 Antwortpositionen auf 41 Seiten.
- Review-Aufwand: 41 Bilder ansehen, 52 Einzelentscheidungen pruefen.
- Nutzen: FP von 11 auf 1 reduzierbar; FN von 14 auf 12 reduzierbar.
- Bewertung: als Zusatzanalyse gut, fuer Routinebetrieb vermutlich nicht zwingend.

## Noch zu erheben

### 1) Vollstaendige Modellmatrix

Fuer jedes Foundation Model, das in die Haupttabelle soll, beide Modi auf denselben 61 Datasets erzeugen:
- `plain`
- `weak`

Vollstaendig auf 61/61 fuer beide Modi (Haupttabelle):
- Gemini-3.1 Flash-Lite
- OpenAI GPT-5.2
- Qwen3-VL-30B  (plain auf 61 nachgezogen, instruct-Variante; weak war bereits 61)
- xAI Grok-4.3  (plain auf 61 nachgezogen; weak war bereits 61)

Nur Premium-Subset (Datasets 1..10), beide Modi `plain` + `weak` (bewusst nicht auf 61, zu teuer):
- Anthropic Claude Opus 4.7  -> `-plain-subset_1_10`, `-weak-subset_1_10`
- OpenAI GPT-5.5             -> `-plain-subset_1_10`, `-weak-subset_1_10`

Aussage des Subsets: auch teure Frontier-Modelle schlagen das guenstige Gemini-3.1 Flash-Lite nicht;
der `weak`-Hint verschlechtert Opus 4.7 und GPT-5.5 sogar (FP/Hint-Bias steigt), waehrend Flash-Lite-weak fuehrt.

### 2) Hauptmetriken pro Modell und Modus

Fuer jedes Modell/Setting dokumentieren:
- Token-Accuracy mit Wilson-95%-CI
- TP, FN, FP, TN
- Recall/FNR
- Precision
- Specificity
- Extra Tokens
- Vollstaendigkeit: `datasets_copied: 61`

Prioritaet fuer Paper:
- FNR als Benachteiligungsmetrik
- FP-Rate als Uebervorteilungs-/Hint-Bias-Metrik
- Vergleich zu YOLOv5/YOLO26

### 3) Plain-vs-Weak-Hint-Kombinationsanalyse

Pro Foundation Model mit beiden vollstaendigen Laeufen berechnen:
- `weak korrekt` / `plain korrekt`
- `weak korrekt` / `plain falsch`
- `weak falsch` / `plain korrekt`
- `weak falsch` / `plain falsch`

Zusaetzlich ausweisen:
- Konfliktpositionen insgesamt
- betroffene Seiten/Bilder
- betroffene Klausuren
- FP/FN, die durch optionalen `plain`-Review vermeidbar waeren
- Review-Aufwand pro korrigiertem Fehler

### 4) Praxis- und Notenrelevanzanalyse

Fuer `weak only` pro Modell:
- FN-Faelle auf Klausuren aggregieren
- FN-Faelle auf Seiten/Bilder aggregieren
- Punktverlust pro betroffener Klausur bestimmen
- mit exemplarischer Notentabelle pruefen, wie oft sich die Note aendern wuerde

Wichtig fuer Paper:
- Notentabelle ist nur indikativ, weil die Datasets aus unterschiedlichen Klausuren stammen.
- Aussage vorsichtig formulieren: Beispielhafte Abschaetzung des erwartbaren Reklamationsaufwands.

### 5) Publikationsartefakte erzeugen

Nach Abschluss neuer Laeufe neu bauen:

```bash
/Users/wiggel/Python/eKlausur2/.venv312/bin/python /Users/wiggel/Python/eKlausur2/build_classification_metrics_all_results.py
/Users/wiggel/Python/eKlausur2/.venv312/bin/python /Users/wiggel/Python/eKlausur2/build_publication_comparison.py
```

Pruefen/ergaenzen:
- `Results/classification_metrics_all_results.md`
- `Results/publication_comparison/README.md`
- Pairwise McNemar-Vergleiche
- Tabellen fuer Paper-Haupttext und Supplement

### 6) Methodische Formulierungen klaeren

Begriffe einheitlich verwenden:
- FN = korrekte studentische Antwort faelschlich als falsch bewertet.
- FP = falsche studentische Antwort faelschlich als korrekt bewertet.
- `weak` = Korrektheitskontext/Referenz-Hint.
- `plain` = kein Hint.

Zu klaerende Paper-Entscheidung:
- Hauptverfahren als `weak only` darstellen.
- `plain` nicht als Pflicht-Zweitstufe, sondern als optionale Bias-/Konsistenzanalyse.
- Studentische Selbstkontrolle als praktischen Safety-Mechanismus beschreiben.
