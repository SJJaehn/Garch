# Vorschläge zu paper.tex

Stand: 16.09.2026, zweiter Durchgang (Code-Stellen geprüft). Geordnet nach Wirkung auf die Note, innerhalb der Blöcke nach Aufwand.
Alle Zahlenangaben sind gegen die Tabellen in `Abbildungen/` geprüft.

Legende: **[Muss]** vor Abgabe · **[Sollte]** klarer Gewinn · **[Kann]** wenn Zeit bleibt

---

## 1. Blocker vor Abgabe

- **[Muss] Änderungsmarkierungen auflösen.** `\neu{x}` → `x`, `\alt{…}` komplett löschen, dann `xcolor`, `ulem` und die beiden `\newcommand` aus der Präambel entfernen. Regex in VS Code: `\\neu\{([^{}]*)\}` → `$1`, danach `\\alt\{[^{}]*\}` → leer. Verschachtelte Klammern (z. B. `\hyperref` in `\neu`) von Hand prüfen.
- **[Muss] Einleitung, Abstract, Abschnitt 5 einkommentieren** und das `% TODO` in Abschnitt 5 abarbeiten (siehe 3.).
- **[Muss] `tab_transaktionskosten` an Tabelle 5 angleichen.** Spalte „0 Bp." zeigt MVP-Hist. 0,695 und 1/N 0,581, Tabelle 5 zeigt 0,567 und 0,488 — gleicher Basisfall. Ursache: `tabellen.py` (ab Z. 667) rechnet die Kosten-Sharpe selbst aus `read_returns`, Tabelle 5 kommt aus `read_summary` — also andere Formel (vermutlich ohne r_f, evtl. arithmetisches Mittel). Fix: in der Kostenfunktion dieselbe Sharpe-Definition wie in der Summary verwenden (Überschussrendite über Fed Funds, Gleichung 15), dann muss die 0-Bp-Spalte exakt Tabelle 5 entsprechen. Die drei neuen Sätze in 4.2 zitieren die aktuellen Werte (0,671 → 0,438; 1/N 0,577; Hist. 0,688) — nach dem Fix nachziehen; die Richtung der Aussage bleibt, weil die Kostenwirkung (≈ 64 Umschläge × 5 Bp ≈ 3,2 % p. a. beim MVP-DCC) unabhängig von der Sharpe-Definition ist.
- **[Muss] Tabelle 3 und Tabelle 4 rechnen unter QLIKE auf verschiedenen Terminmengen.** Tabelle 4: GARCH/DCC vs. Historisch auf 6.013 Terminen, DCC vs. GARCH auf 6.079. Folge: Differenz der Mittelwerte aus Tabelle 3 (GARCH − Hist. = −155,94 + 152,81 = **−3,13**) ≠ mittlere Differenz in Tabelle 4 (**−2,56**); für DCC − Hist. −3,87 vs. −3,31. Unter MSE stimmt alles exakt (−0,76 / −0,84 / −0,08, 6.079 Termine überall). Also ist der QLIKE der historischen Matrix an 66 Terminen nicht definiert (NaN/inf — vermutlich singuläre oder nahezu singuläre Stichprobenkovarianz, z. B. wenn ein Sektor im Fenster viele Nullrenditen hat), und `analyze.py` lässt diese Termine paarweise fallen (`dropna()` wie in Z. 116). Ein Gutachter, der nachrechnet, findet das. Fix: alle drei QLIKE-Mittel in Tabelle 3 auf der gemeinsamen Terminmenge (6.013) berechnen, die Terminzahl in die Tabellennotiz schreiben und in 3.4 einen Satz ergänzen, warum Termine entfallen. Zusätzlich prüfen, ob dieselben 66 Termine auch in `tab_qlike_grid` und `tab_bestzaehlung` stecken.
- **[Muss] `paper.tex.vorher` löschen** oder aus dem Repo nehmen, bevor irgendwer den Ordner abgibt.

## 2. Inhalt — Daten

- **[Sollte] 260 Handelstage pro Jahr.** 7.087 Beobachtungen / 27,2 Jahre = 260,5. Das ist der Werktagskalender inklusive Feiertagen (Datastream füllt Feiertage mit dem Vortageskurs). Im Code bestätigt: `backtest.py:83` macht `level.reindex(dates).ffill().pct_change()`, d. h. jeder Kalendertag im Datumsraster ohne Kurs wird zum Nullrenditetag. Passt auch zu den OOS-Startdaten (252 Tage ab April 1999 → März 2000) und zu den „rund 4.280 Handelstagen“ ab 2010 in 3.1 (16,4 Jahre × 260 = 4.274; mit 252 wären es 4.143). Folgen: ~8 Nullrenditetage pro Jahr, leicht gedrückte Varianz- und Korrelationsschätzungen, 252er-Annualisierung minimal daneben.
  - Billigste Lösung: ein Satz in 2.1 („Die Reihen enthalten Feiertage als Werktage mit Nullrendite; auf eine Bereinigung wurde verzichtet, da…") plus ein Halbsatz in den Limitationen.
  - Saubere Lösung: in `backtest.py` nach Z. 83 alle Tage entfernen, an denen sämtliche verfügbaren Sektoren exakt 0 % Rendite haben (`(ret == 0).all(axis=1)`), dann Pipeline neu laufen lassen. Ändert Zahlen in der dritten Stelle, nicht die Aussagen. Nebeneffekt: die 66 undefinierten QLIKE-Termine (siehe Block 1) könnten damit teilweise verschwinden, weil Nullrenditetage die Stichprobenkovarianz in Richtung Singularität drücken.
  - Nach dem Fix muss die Annualisierung (252) und die Tageszahl in 2.1, 3.1 und den Tabellennotizen einmal durchgezogen werden.
- **[Kann]** In 2.1 die Sektorbezeichnungen der 27 Business Groups nennen (Anhangstabelle). Ein Gutachter will wissen, was „Sektor“ hier konkret ist.
- **[Kann]** Fed Funds Rate: angeben, wie die annualisierte Rate in eine Tagesrate umgerechnet wird (÷ 252 oder ÷ 360).

## 3. Inhalt — Methodik

- **[Sollte] Implementierungsabsatz aus dem Abstract in 3.4 verschieben** (oder als eigenen Unterabschnitt 3.5 „Implementierung“). Ins Abstract gehört das nicht; dort reicht ein Halbsatz „Code und Daten liegen im begleitenden Repository“. Inhalt des Absatzes: Python, `arch` für GARCH(1,1), DCC und Verlustmaße eigene Implementierung, `riskfolio`/`scipy` für Portfolios, Neuschätzung an jedem Termin, Parallelisierung. Ergänzen, falls bekannt: Optimierer, Umgang mit Nichtkonvergenz (`backtest.py:136–138` hat offenbar einen Fallback), HAC-Bandbreite im DM-Test (im Code nicht gefunden — falls `statsmodels`-Standard, das sagen). Aus dem Code belegt und im Text nennenswert: 90-%-Regel als `train.notna().mean() >= 0.9` (Z. 581), fehlende Testrenditen werden mit 0 gefüllt (Z. 589), HRP mit **single linkage** wie bei de Prado (Z. 372), CCC-Korrelation mit `fillna(0.0)` (Z. 183 — ein Sektor ohne gültige Residuen bekommt Korrelation null zu allen anderen; kurz begründen oder als Randfall erwähnen).
- **[Sollte] QLIKE als primäres Verlustmaß deklarieren.** Mit sechs DM-Vergleichen überlebt die MSE-Signifikanz (p = 0,026 / 0,035) keine Bonferroni-Korrektur (×6 → 0,16 / 0,21). Der markierte Nebensatz in 3.4 sagt das ehrlich, aber 4.1 („Belastbar ist somit, dass beide dynamischen Schätzer die historische Matrix unter beiden Maßen schlagen“) und das Fazit („Abstände zur historischen Matrix im DM-Test signifikant“) lehnen sich implizit auf die unkorrigierten MSE-Tests. Sauberste Lösung: in 3.4 QLIKE als Hauptkriterium und MSE als Robustheitsprüfung festlegen (dafür gibt es mit der asymmetrischen, ökonomisch motivierten Bestrafung schon das Argument im Text); dann ist Mehrfachtestung kein Thema und „Belastbar“ bezieht sich auf QLIKE. In 4.1 dann: „unter QLIKE signifikant, unter dem MSE in der Punktschätzung bestätigt“.
- **[Sollte] Behr et al. (2008) in 3.3 gegen die eigenen Ergebnisse halten.** „MVP überzeugt besonders in volatilen Marktphasen“ — die Teilperioden zeigen das Gegenteil: 2008/09 MVP −0,25 vs. 1/N −0,12; 2020/21 MVP 0,44 vs. 1/N 0,83. Entweder den Satz streichen oder in Anhang A einen Satz ergänzen, dass die Behr-Aussage im vorliegenden Universum nicht bestätigt wird. Unkommentiert ist es eine offene Flanke.
- **[Kann]** ERC-Lösung: kurz sagen, wie das Gleichungssystem gelöst wird (numerisch, Startwert 1/N) — ein Halbsatz.
- **[Kann]** HRP: „single linkage“ (Z. 372, wie bei de Prado 2016) in 3.3 nennen; `scipy.cluster.hierarchy.linkage` hat mehrere Verfahren, und Ward oder average würden andere Bäume liefern.
- **[Kann]** Beim GARCH-Modell wird μ als Konstante geschätzt — sagen, ob μ aus `arch` oder das Stichprobenmittel verwendet wird.

## 4. Inhalt — Ergebnisse

- **[Sollte] Jobson-Korkie-Tabelle:** „Historical“ und „Naive“ in „Historisch“ und „1/N“ ändern (in `tabellen.py`). Alle anderen Tabellen sind deutsch.
- **[Sollte] Anhang A: „17 auf 25 Sektoren im September 2022“** vs. Tabellenspanne 17–27 für 2022–2026. Wahrscheinlich zwei Schritte: die 2019 gestarteten Reihen erfüllen die 90-%-Regel im 1008-Tage-Fenster ab etwa Herbst 2022 (→ 25), zwei später gestartete Reihen erst 2024/25 (→ 27). Wenn das stimmt, einen Halbsatz ergänzen; wenn nicht, die Zahl korrigieren.
- **[Sollte] Ungenutztes Argument in Anhang B:** In der DCC-Simulation schlägt das *falsch* spezifizierte CCC-GARCH das *korrekt* spezifizierte DCC beim MVP-Sharpe in 9 von 10 Fenstern (z. B. 1008 Tage: 1,050 vs. 0,940), obwohl DCC den besseren QLIKE und die niedrigere realisierte Volatilität hat. Das ist die reinste Form der Kernthese — bekannter DGP, richtiges Modell, trotzdem kein Sharpe-Vorteil. Ein Satz in Anhang B, ein Rückverweis im Fazit.
- **[Sollte] „In sämtlichen 40 Kombinationen“ nachprüfbar machen.** Die Aussage (4.1, Fazit, Abstract) lässt sich aus keiner Tabelle ablesen: `tab_bestzaehlung` zählt nur den *besten* Schätzer (DCC 39/40 unter MSE, 1× GARCH), `tab_qlike_grid` zeigt 14 der 40 Zellen. Eine Zeile „DCC vs. Historisch“ in `tab_bestzaehlung` nach dem Muster von `tab_bilanz` (Anzahl der Läufe, in denen DCC den niedrigeren Verlust hat) macht den Satz belegbar — und liefert die genaue Zahl für „unter QLIKE in nahezu allen“ (aktuell nur ableitbar als ≤ 37).
- **[Sollte] Turnover des MVP plausibilisieren.** 0,255 pro Tag heißt: das MVP mit DCC-Kovarianz schichtet jeden Tag ein Viertel des Portfolios um, ≈ 64 volle Umschläge pro Jahr. Das ist kein Tippfehler (CCC-GARCH liegt mit 0,258 gleichauf, die Simulation mit 0,15), aber außergewöhnlich hoch und der Dreh- und Angelpunkt der Kernthese. Ein Gutachter fragt: springen die Gewichte wirklich so, oder ist das ein Artefakt des long-only-QP, das zwischen fast gleich guten Ecklösungen hin- und herspringt? Diagnose (eine Stunde): Gewichtsverlauf des MVP-DCC für ein ruhiges Jahr plotten und die effektive Anzahl gehaltener Anlagen $1/\sum w_i^2$ als Spalte in Tabelle 5 ausweisen. Das würde nebenbei die bisher nur aus Lee (2011) zitierte, nie gemessene Konzentration des MVP belegen und die Aussage „HRP und ERC sind robuster“ mit einer Zahl unterlegen. Zusätzlicher Befund gratis: da CCC und DCC praktisch denselben Turnover haben, stammt der Umschlag fast vollständig aus der Varianz- und nicht aus der Korrelationsdynamik — das steht so noch nirgends explizit.
- **[Kann]** Jobson-Korkie-Tabelle: getestet werden nur die historischen Varianten gegen 1/N. Da 4.2 behauptet, *alle neun* Kombinationen lägen über 1/N, fehlen die sechs dynamischen Vergleiche gegen 1/N (oder ein Satz, dass sie bei noch kleineren Differenzen erst recht insignifikant sind).
- **[Kann]** Anhang A, Tabelle 9 zeigt MVP mit historischer und **GARCH**-Kovarianz; das Hauptmodell der Arbeit ist DCC. Spalte auf DCC umstellen oder ergänzen.
- **[Kann]** Signifikanz über die 40 Läufe: JK-Tests gibt es nur im Basisfall. Ein Satz in 4.2, dass die Bilanz über 40 Läufe keine Signifikanzaussage trägt, oder — teurer — JK-Tests je Lauf und in `tab_bilanz` die Anzahl signifikanter Fälle ausweisen.
- **[Kann]** In 4.2 den Faktor beim HRP-Turnover präzisieren: GARCH ×6, DCC ×8 („sechs bis acht“ statt „acht beziehungsweise sechs“, was sich derzeit auf HRP/ERC bezieht und beim ERC 5,7 ist).
- **[Kann]** Abbildung: eine Zeitreihe der prognostizierten Portfoliovolatilität (MVP hist. vs. DCC) über 2008–2010 oder 2020 würde die „Trägheit“ der historischen Matrix in einem Blick zeigen und das Turnover-Argument anschaulich machen. Halber Tag, hoher Erklärwert.

## 5. Struktur

- **[Sollte] „Weitergehende Forschung“** ist als eigenes Kapitel nach dem Fazit unüblich. Optionen: (a) als Unterabschnitt 7.1 unter Limitationen, (b) als letzter Absatz des Fazits (dann kürzen auf zwei Ansatzpunkte: kostenbewusster Optimierer und weitere Universen).
- **[Sollte] Abschnitt 5 TODO:** zwei bis drei aktuelle Arbeiten zu HERC/NCO (Raffinot 2017 für hierarchisches Clustering, Raffinot 2018 für HERC, López de Prado 2019/2020 für NCO) und zu DCC in Risk-Parity-Kontexten ergänzen. Nicht mehr — Abschnitt 5 soll die Befunde einordnen, nicht die Einleitung wiederholen.
- **[Kann]** Abstract/Einleitung (auskommentiert): „jede der drei Kovarianzschätzungen mit jedem der vier Allokationsverfahren“ — 1/N verwendet keine Kovarianz. Besser „mit jedem der drei risikobasierten Verfahren, jeweils gegen 1/N“.
- **[Kann]** 3.1: „Der lange Lauf schließt damit den Einbruch der Finanzkrise aus“ — der OOS-Start im November 2008 liegt *nach* dem Kurssturz vom Herbst 2008, aber *vor* dem Tief im März 2009. Präziser: „setzt erst nach dem Kurssturz vom Herbst 2008 ein und erfasst im Wesentlichen die Erholung“.
- **[Kann]** Anhang-Reihenfolge: A (Zeitraum), B (Synthetik), C (Kosten). Sinnvoller wäre Kosten direkt nach A, weil beides empirische Robustheit ist, und Synthetik zuletzt.
- **[Kann]** Abschnitt 4.3 heißt „Robustheit über Zeiträume und Anlageuniversen“, behandelt aber nur den Zeitraum; der Universumsteil ist auskommentiert. Titel kürzen auf „Robustheit über Bewertungszeiträume“ oder den auskommentierten Absatz (Zeilen um 508) in einer gekürzten Fassung wieder aufnehmen — er ist eigentlich gut.

## 6. Stil und Form

- **[Sollte] Titel vs. Kopfkommentar:** Kopfzeile sagt „Risikomodellierung und Risikoprognose mit GARCH-Modellen“, `\title` sagt „Risikobasierte Portfoliooptimierung mit GARCH-Modellen“. Falls die Kopfzeile das offizielle Modulthema ist, sollte es als Untertitel auftauchen.
- **[Sollte] `\author{}` und `\date{}` sind leer.** Namen, Matrikelnummern, Modul, Semester, Betreuer — je nach Vorgabe des Lehrstuhls, gegebenenfalls als Titelseite.
- **[Sollte]** Abkürzungen beim ersten Auftreten ausschreiben: „DM-Test“ (Diebold-Mariano) in 4.1 wird ohne Einführung verwendet; in 3.4 heißt es nur „Test von Diebold & Mariano“. Ebenso „OOS“ falls es irgendwo auftaucht.
- **[Kann]** Gleichungsnummern werden konsequent per `\hyperref` referenziert — gut. Aber die `\bar{h}`-Notation ist inkonsistent: Gleichung 5 nutzt `\bar{h}_{t,h}`, Gleichung 6 `\bar{h}_{1},\dots,\bar{h}_{N}`. Einheitlich `\bar{h}_{i,t,h}` oder Index-Konvention einmal erklären.
- **[Kann]** Tabellenüberschriften stehen unter den Tabellen (`position=bottom`). Bei APA und den meisten WiWi-Lehrstühlen stehen sie oben. Vorgabe prüfen.
- **[Kann]** Zahlenformat: im Text „5~Basispunkte“, „0,562“, „40~Konstellationen“ — konsistent. In Tabellen „Bp.“, im Text „Basispunkte“. Vereinheitlichen.
- **[Kann]** „Turnover“, „Rolling Window“, „Out-of-Sample“, „Rebalancing“/„Rebalancierung“ — beide Schreibweisen für Rebalancing kommen vor. Auf „Rebalancierung“ festlegen (steht in den Tabellen).
- **[Kann]** Der Satz „Beide hier verwendeten Maße gehören zu den Verlustfunktionen, die Laurent et al. (2012) … einsetzen“ direkt nach dem Laurent-2013-Satz: zwei Laurent-Zitate hintereinander mit verschiedenen Jahren irritiert. Einen davon in eine Klammer ziehen.
- **[Kann]** Literaturverzeichnis: prüfen, dass `ledoit2004` und `west1996` (in der .bib) entweder zitiert werden oder raus. Biber warnt nicht, aber ein Gutachter, der die .bib sieht, fragt nach Shrinkage — was übrigens ein naheliegender vierter Kovarianzschätzer wäre (siehe 7.).

## 7. Falls noch Zeit ist (Reihenfolge nach Ertrag/Aufwand)

1. **Kalibrierung prüfen (½ Tag).** Limitationen sagen selbst, dass nur die relative Prognosegüte bewertet wird. Ein Kupiec-Test auf die 1-%-VaR-Überschreitungen des MVP je Kovarianzschätzer wäre die natürliche Ergänzung für eine Arbeit mit „Risikoprognose“ im Modultitel. Eine Tabelle, drei Sätze.
2. **Ledoit-Wolf-Shrinkage als vierter Schätzer (1 Tag).** Die .bib hat den Eintrag schon. Shrinkage ist der Standard-Gegenspieler zur Stichprobenkovarianz und würde die Frage „liegt es an der Dynamik oder an der Schätzgenauigkeit?“ direkt beantworten. Passt in die „Leiter“-Logik als zweiter statischer Schätzer.
3. **Kostenbewusster Optimierer (2 Tage).** Turnover-Strafterm in der MVP-Zielfunktion. Wird in „Weitergehende Forschung“ als „der entscheidende Test“ bezeichnet — wenn er entscheidend ist, ist er auch der beste Kandidat für eine Erweiterung, falls die Zeit reicht.
4. **Seeds/Mehrfachsimulation für Anhang B (2 Stunden).** Drei synthetische Datensätze mit je einer Realisation. Fünf Seeds und Mittelwerte würden die „Sharpe ist zu verrauscht“-Aussage in Anhang B mit einer Streuungsangabe belegen statt nur behaupten.

## 8. Wahrscheinliche Gutachterfragen (Kolloquium / Rückfragen)

Kurz vorbereiten, je zwei Sätze reichen:

1. *Warum kein Shrinkage-Schätzer (Ledoit-Wolf) als Vergleich?* — Antwort vorbereiten, warum die Leiter „statisch → Varianzdynamik → Korrelationsdynamik“ gewählt wurde; Shrinkage wäre eine zweite statische Stufe (siehe Block 7).
2. *Warum long-only?* — Praxisnähe und Vermeidung der Michaud-Fehlermaximierung bei Short-Positionen; ehrlich sagen, dass die MVP-Konzentration mit Shorts noch extremer wäre.
3. *Warum 1008 Tage als Basisfall?* — Tabelle 7/Abbildung: Vorsprung stabilisiert sich ab etwa fünf Jahren, 1008 ist der kürzeste Wert im stabilen Bereich mit langem OOS-Zeitraum.
4. *Ist der Turnover von 0,255 pro Tag realistisch?* — siehe Block 4; ohne Diagnose ist das die gefährlichste Frage.
5. *Warum tägliche Neuschätzung, wenn niemand täglich rebalanciert?* — h = 5/10/21 sind genau dafür da; die Antwort steht in Tabelle 8, aber der Satz „wöchentliches Rebalancing ist der praxisrelevante Fall“ fehlt im Text.
6. *Was heißt „QLIKE ist dimensionsabhängig“ für den Vergleich über Trainingsfenster?* — 3.4 und Anhang A haben die Antwort (nur Differenzen innerhalb einer Spezifikation vergleichen); sie sollte auch in 4.1 einmal explizit stehen.
7. *Wie viele der 66 fehlenden QLIKE-Termine liegen in Stressphasen?* — nach dem Fix aus Block 1 beantwortbar.
8. *Normalverteilung im GARCH — wurde eine t-Verteilung probiert?* — Nein; im Forschungsabschnitt genannt. Ein Hinweis, dass die QML-Schätzung unter Normalverteilung für die Varianzprognose konsistent bleibt (Bollerslev-Wooldridge 1992), nimmt der Frage die Spitze; ggf. als Zitat ergänzen.

## 9. Vor dem Abgeben

- `paper.pdf` frisch bauen: `pdflatex → biber → pdflatex → pdflatex`. Prüfen, dass keine „??“-Referenzen und keine `\citeauthor`-Ausgaben ohne Jahr bei Erstnennung mehr im PDF stehen.
- Seitenzahl gegen die Vorgabe prüfen; aktuell ~24 Seiten mit Anhang, Hauptteil ohne Anhang und Literatur zählen.
- `git status` sauber: `.DS_Store` und `paper.tex.vorher` raus.
