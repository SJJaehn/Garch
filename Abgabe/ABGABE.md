# Abgabe: Code und Ergebnisse (TRBC + synthetische Daten)

Enthalten ist alles, was für die Ergebnisse des empirischen Datensatzes TRBC und
der drei synthetischen Datensätze (MonteCarlo, GARCH_sim, DCC_sim) gebraucht
wird: der Code, die Eingangsdaten und die fertigen Ergebnisdateien.

## Inhalt

```
main.py                 Einstellungen (Datensätze, Fenstergrößen) + Pfade,
                        Datensatzregister, Solver-Einstellungen
datagen.py              erzeugt die synthetischen Kursreihen -> DATA/Artifical/
backtest.py             der Kern: Datenlader, GARCH/DCC, Kovarianzschätzer,
                        Portfoliogewichte, Kennzahlen, Rolling-Window-Schleife
analyze.py              Grafiken je Lauf + Aggregattabelle + Übersichtsgrafiken
regen_all.sh            erzeugt in einem Durchlauf alle Ergebnisse neu
requirements.txt        Python-Pakete
README.md               Projektbeschreibung

paper/tabellen.py       erzeugt die LaTeX-Tabellen der Arbeit aus den CSVs
paper/abbildungen.py    erzeugt die Abbildungen der Arbeit
paper/Abbildungen/      die fertigen Tabellen (.tex) und Abbildungen (.pdf)

DATA/Empirical/TRBC_Business_Sectors_clean.csv   Kursdaten TRBC
DATA/Empirical/FED_FUNDS.csv                     Fed-Funds-Index (risikoloser Zins)
DATA/Artifical/{monte_carlo,garch,dcc}.csv       die synthetischen Kursreihen

Ergebnisse/TRBC/<train>_<pred>/         40 Läufe (10 Trainingsfenster x 4 Horizonte)
Ergebnisse/GARCH_sim/<train>_1/         10 Läufe
Ergebnisse/DCC_sim/<train>_1/           10 Läufe
Ergebnisse/MonteCarlo/<train>_1/        2 Läufe (1008, 1260)
Ergebnisse/Zusammenfassung/             aggregate_results.xlsx + Übersichtsgrafiken
Ergebnisse/Sonstige/                    Rendite-Vola-Grafiken je Datensatz
```

Je Lauf liegen dort `summary.csv` (Kennzahlen je Strategie), `returns.csv`
(Tagesrenditen), `backtest_metrics.csv` (Kennzahlen je Fenster), `qlike.csv`,
`cov_mse.csv` bzw. `cov_rmse.csv`, `sharpe_tests.csv`, bei den synthetischen
Läufen zusätzlich `dmw_tests.csv`, sowie die Grafiken des Laufs. Beim Basisfall
`Ergebnisse/TRBC/1008_1/` kommt `weights.csv` hinzu (die Portfoliogewichte je
Formationstermin).

## Reproduzieren

```bash
pip install -r requirements.txt
./regen_all.sh          # synthetische Daten + alle Läufe + LaTeX-Tabellen
```

Einzelne Schritte:

```bash
python datagen.py       # 1. synthetische Kursreihen neu ziehen (SEED = 1)
python main.py          # 2. Backtest über das Gitter aus dem Einstellungsblock
python analyze.py       # 3. Grafiken + Ergebnisse/Zusammenfassung/
python paper/tabellen.py    # LaTeX-Tabellen nach paper/Abbildungen/
python paper/abbildungen.py # Abbildungen nach paper/Abbildungen/
```

Der Backtest selbst enthält keinen Zufall, die Ergebnisse sind also
reproduzierbar; die einzige Zufallsquelle ist `datagen.py`.

## Nicht enthalten

- **`weights.csv`** (die Portfoliogewichte jedes einzelnen Formationstermins) --
  mit einer Ausnahme: `Ergebnisse/TRBC/1008_1/weights.csv` liegt bei, weil
  `paper/tabellen.py` daraus die Spalte "Anlagen > 1 %" in
  `tab_oekonomie_basis` rechnet. Alle übrigen Gewichtsdateien wären zusammen
  rund 1,6 GB und werden von keiner Tabelle gelesen; sie entstehen bei jedem
  Lauf neu (`LOG_WEIGHTS = True` in `main.py`).
- **Die R-Gegenrechnung** (`validate_with_R.R` und die `*_R.csv`-Dateien). Sie
  diente nur der Kontrolle der Python-Zahlen. `paper/tabellen.py` las die
  Gewichte früher aus `weights_R.csv` und rechnet jetzt mit den Gewichten des
  Python-Backtests; `paper/tabellen.py` läuft damit ohne R vollständig durch.
- **Die Datensätze SP500 und Dow.** Sie stehen im Register in `main.py`, gehören
  aber nicht zu den abgegebenen Ergebnissen, deshalb fehlen ihre CSV-Dateien.
