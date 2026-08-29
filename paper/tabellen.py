"""
Erzeugt sämtliche Tabellen der Arbeit aus den Ergebnis-CSVs.

Gelesen wird ausschliesslich aus CSV-Dateien:
  DATA/Empirical/TRBC_Business_Sectors_clean.csv  (Deskription, Universum)
  Ergebnisse/<Datensatz>/<train>_<pred>/summary.csv
  Ergebnisse/<Datensatz>/<train>_<pred>/qlike.csv
  Ergebnisse/<Datensatz>/<train>_<pred>/returns.csv
  Ergebnisse/<Datensatz>/<train>_<pred>/sharpe_tests.csv

Geschrieben wird je Tabelle eine eigene .tex-Datei nach paper/Abbildungen/.
Diese Dateien enthalten eine vollständige table-Umgebung und werden im
Manuskript per \\input{Abbildungen/<name>} eingebunden.

    python paper/tabellen.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# der Loader aus backtest.py wird wiederverwendet, damit die Deskription exakt
# auf derselben Datengrundlage steht wie der Backtest
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
import backtest


# =============================================================================
# Settings
# =============================================================================

OUT_DIR = os.path.join(REPO_DIR, "paper", "Abbildungen")
RESULTS_DIR = os.path.join(REPO_DIR, "Ergebnisse")

DATASET = "TRBC"                 # empirischer Datensatz der Hauptanalyse
BASE_TRAIN = 1008                # Basisfall: 4 Jahre Training
BASE_PRED = 1                    # Basisfall: 1 Tag Prognose

TRAIN_WINDOWS = [252, 504, 756, 1008, 1260, 1512, 1764, 2016, 2268, 2520]
PRED_WINDOWS = [1, 5, 10, 21]

SYNTH_DATASETS = ["MonteCarlo", "GARCH_sim", "DCC_sim"]
SYNTH_LABEL = {"MonteCarlo": "Monte Carlo", "GARCH_sim": "GARCH-Sim.",
               "DCC_sim": "DCC-Sim."}

# Gemeinsamer Bewertungszeitraum für alle Vergleiche über Trainingsfenster und
# Prognosehorizonte hinweg. Das längste Fenster (2520 Tage) startet erst im
# November 2008 und würde den Einbruch der Finanzkrise nicht mehr, die Erholung
# danach aber vollständig enthalten -- das allein hebt seine Sharpe Ratio
# gegenüber kurzen Fenstern an. Ab 2010 sind Einbruch und Erholung für alle
# Fenster gleichermaßen ausgeschlossen.
COMMON_START = "2010-01-01"

# Teilperioden für die Strukturbruchanalyse
SUB_PERIODS = [("2003--2007", "2003-01-01", "2007-12-31"),
               ("2008--2009", "2008-01-01", "2009-12-31"),
               ("2010--2019", "2010-01-01", "2019-12-31"),
               ("2020--2021", "2020-01-01", "2021-12-31"),
               ("2022--2026", "2022-01-01", "2026-12-31")]

COV_ORDER = ["Historical", "GARCH", "DCC"]
COV_LABEL = {"Historical": "Historisch", "GARCH": "GARCH", "DCC": "DCC-GARCH"}
MODEL_ORDER = ["MVP", "HRP", "ERC"]

# Kostensätze der Transaktionskostenrechnung, in Basispunkten je umgeschlagener
# Einheit Portfoliovolumen
COST_LEVELS = [0, 5, 10, 20]

TRADING_DAYS = 252


# =============================================================================
# Hilfsfunktionen: Formatierung und Ausgabe
# =============================================================================

"""
Formatiert eine Zahl mit k Nachkommastellen im deutschen Format (Dezimalkomma).
Fehlende Werte werden als Gedankenstrich gesetzt.
"""
def num(value, k=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{k}f}".replace(".", ",")


"""
Formatiert ein Datum im deutschen Format TT.MM.JJJJ.
"""
def datum(value):
    return pd.Timestamp(value).strftime("%d.%m.%Y")


"""
Formatiert einen p-Wert. Sehr kleine Werte werden abgeschnitten, damit in der
Tabelle keine Nullen stehen.
"""
def pval(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0,001"
    return num(p, 3)


"""
Signifikanzsterne zu einem p-Wert (1 %, 5 %, 10 %).
"""
def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "$^{***}$"
    if p < 0.05:
        return "$^{**}$"
    if p < 0.10:
        return "$^{*}$"
    return ""


"""
Schreibt eine fertige table-Umgebung nach paper/Abbildungen/<name>.tex.
"header" ist entweder eine einzelne Kopfzeile ohne abschliessendes \\\\ oder
eine Liste bereits vollständig gesetzter Kopfzeilen (für mehrzeilige Köpfe
mit \\cmidrule). "rows" ist eine Liste formatierter Datenzeilen, "spec" die
Spaltendefinition des tabular.
"""
def write_table(name, caption, label, spec, header, rows, note=None, small=True,
                wide=False):
    lines = []
    lines.append("% automatisch erzeugt von paper/tabellen.py -- nicht von Hand editieren")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    if small:
        lines.append("  \\footnotesize")
    # breite Tabellen werden auf die Textbreite gestaucht, statt in den Rand zu laufen
    if wide:
        lines.append("  \\resizebox{\\linewidth}{!}{%")
    lines.append("  \\begin{tabular}{" + spec + "}")
    lines.append("    \\toprule")
    if isinstance(header, str):
        lines.append("    " + header + " \\\\")
    else:
        for line in header:
            lines.append("    " + line)
    lines.append("    \\midrule")
    for row in rows:
        if row == "MIDRULE":
            lines.append("    \\midrule")
        else:
            lines.append("    " + row + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    if wide:
        lines.append("  }")
    # Beschriftung steht unter der Tabelle
    lines.append("  \\caption{" + caption + "}")
    lines.append("  \\label{" + label + "}")
    if note:
        lines.append("  \\begin{minipage}{\\linewidth}\\vspace{0.4em}\\centering\\footnotesize")
        lines.append("  " + note)
        lines.append("  \\end{minipage}")
    lines.append("\\end{table}")

    path = os.path.join(OUT_DIR, name + ".tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  geschrieben:", os.path.relpath(path, REPO_DIR))


# =============================================================================
# Hilfsfunktionen: Einlesen der Ergebnis-CSVs
# =============================================================================

"""
Pfad eines Backtest-Laufs.
"""
def run_dir(dataset, train, pred):
    return os.path.join(RESULTS_DIR, dataset, f"{train}_{pred}")


"""
summary.csv eines Laufs, indiziert nach der Strategiebezeichnung
("MVP DCC", "Naive", ...).
"""
def read_summary(dataset, train, pred):
    df = pd.read_csv(os.path.join(run_dir(dataset, train, pred), "summary.csv"))
    name = np.where(df["Model"] == "Naive", "Naive",
                    df["Model"] + " " + df["Covariance Type"])
    df.insert(0, "Strategie", name)
    return df.set_index("Strategie")


"""
qlike.csv eines Laufs (Zeilen = Formationstermine, Spalten = Kovarianzschätzer).
"""
def read_qlike(dataset, train, pred):
    return pd.read_csv(os.path.join(run_dir(dataset, train, pred), "qlike.csv"),
                       index_col=0, parse_dates=True)


"""
returns.csv eines Laufs (Zeilen = Bewertungstage, Spalten = Strategien).
"""
def read_returns(dataset, train, pred):
    return pd.read_csv(os.path.join(run_dir(dataset, train, pred), "returns.csv"),
                       index_col=0, parse_dates=True)


"""
Kovarianz-MSE je Rebalancierungstermin (Zeilen = Termine, Spalten = Schätzer).

Die synthetischen Läufe schreiben den MSE direkt nach cov_mse.csv. Die
empirischen Läufe stammen aus einer älteren Fassung des Backtests, die je
Fenster die Wurzel gezogen und nach cov_rmse.csv geschrieben hat; Quadrieren
stellt den MSE des Fensters exakt wieder her. Der MSE wird erst danach über die
Fenster gemittelt: Ein vorab je Fenster gezogener und dann gemittelter RMSE
gehört nicht mehr zur robusten Verlustklasse nach Patton (2011) und kann die
Schätzer falsch ordnen.
"""
def read_cov_mse(dataset, train, pred):
    folder = run_dir(dataset, train, pred)
    path_mse = os.path.join(folder, "cov_mse.csv")
    if os.path.exists(path_mse):
        return pd.read_csv(path_mse, index_col=0, parse_dates=True)
    rmse = pd.read_csv(os.path.join(folder, "cov_rmse.csv"), index_col=0, parse_dates=True)
    return rmse ** 2


# =============================================================================
# Hilfsfunktionen: Kennzahlen und Tests
# =============================================================================

"""
Annualisierte Sharpe Ratio bei einem risikofreien Zins von null, identisch zur
Spalte "Ann. Sharpe (rf=0)" in summary.csv.
"""
def sharpe0(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return (r.mean() * TRADING_DAYS) / sd if sd > 0 else np.nan


"""
Annualisierte realisierte Volatilität in Prozent.
"""
def realized_vol(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100


"""
Diebold-Mariano-West-Test auf die mittlere Verlustdifferenz d_t = L_a - L_b.
Die Langfristvarianz wird mit einem Bartlett-Kern (Newey-West) geschätzt, die
Lag-Länge nach der ueblichen Faustregel 1,5*n^(1/3). Ein negativer Mittelwert
bedeutet, dass Schätzer a den kleineren Verlust aufweist.
"""
def dmw_test(diff):
    d = np.asarray(pd.Series(diff).dropna(), dtype=float)
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    mean = d.mean()
    e = d - mean
    lag = int(np.floor(1.5 * n ** (1.0 / 3.0)))
    var = (e * e).mean()
    for l in range(1, lag + 1):
        var += 2.0 * (1.0 - l / (lag + 1.0)) * (e[l:] * e[:-l]).mean()
    if var <= 0:
        return mean, np.nan, np.nan, n
    t = mean / np.sqrt(var / n)
    p = 2.0 * stats.norm.sf(abs(t))
    return mean, t, p, n


"""
Jobson-Korkie-Test mit der Korrektur von Memmel (2003) auf die Differenz zweier
Sharpe Ratios, die auf derselben Stichprobe gemessen wurden. a und b sind
ausgerichtete Arrays taeglicher Überschussrenditen. Zurückgegeben werden die
beiden annualisierten Sharpe Ratios, die z-Statistik, der p-Wert und der
Stichprobenumfang.
"""
def jobson_korkie(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    sr_a = a.mean() / a.std(ddof=1)
    sr_b = b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (1.0 / n) * (2.0 - 2.0 * rho
                         + 0.5 * (sr_a ** 2 + sr_b ** 2 - 2.0 * sr_a * sr_b * rho ** 2))
    z = (sr_a - sr_b) / np.sqrt(theta)
    p = 2.0 * stats.norm.sf(abs(z))
    ann = np.sqrt(TRADING_DAYS)
    return sr_a * ann, sr_b * ann, z, p, n


"""
Größe des investierbaren Universums je Formationstermin. Die Auswahlregel ist
dieselbe wie in backtest.run_window: mindestens 90 % Beobachtungen im
Trainingsfenster und eine gueltige Beobachtung am letzten Trainingstag.
Aus Laufzeitgründen wird nur jeder step-te Termin ausgewertet.
"""
def universe_sizes(log_returns, train, step=5):
    sizes = []
    dates = []
    for start in range(0, len(log_returns) - train, step):
        train_slice = log_returns.iloc[start: start + train]
        train_slice = train_slice.loc[:, train_slice.notna().mean() >= 0.9]
        train_slice = train_slice.loc[:, train_slice.iloc[-1].notna()]
        sizes.append(train_slice.shape[1])
        dates.append(log_returns.index[start + train - 1])
    return pd.Series(sizes, index=pd.DatetimeIndex(dates))


# =============================================================================
# Tabellen: Daten
# =============================================================================

"""
Deskriptive Statistik des TRBC-Sektorpanels: Anzahl Beobachtungen sowie
annualisierte Rendite und Volatilität als Querschnittsverteilung über die
Sektorindizes.
"""
def table_daten_deskriptiv(log_returns):
    d = log_returns
    stat = pd.DataFrame({
        "Beobachtungen": d.notna().sum(),
        "Rendite p.a. (\\%)": d.mean() * TRADING_DAYS * 100,
        "Volatilität p.a. (\\%)": d.std() * np.sqrt(TRADING_DAYS) * 100,
    })
    rows = []
    for col in stat.columns:
        k = 0 if col == "Beobachtungen" else 2
        q = stat[col]
        rows.append(" & ".join([col, num(q.min(), k), num(q.quantile(0.25), k),
                                num(q.median(), k), num(q.quantile(0.75), k),
                                num(q.max(), k)]))

    corr = d.corr().values
    iu = np.triu_indices_from(corr, 1)
    pair = corr[iu]
    pair = pair[np.isfinite(pair)]

    note = (f"Tägliche logarithmierte Renditen von {d.shape[1]} TRBC-Sektorindizes, "
            f"{datum(d.index.min())} bis {datum(d.index.max())}; durchschnittliche "
            f"paarweise Korrelation {num(pair.mean(), 3)}.")

    write_table("tab_daten_deskriptiv",
                "Deskriptive Statistik der TRBC-Sektorrenditen",
                "tab:daten_deskriptiv",
                "lrrrrr",
                "Kennzahl & Min. & 25\\,\\% & Median & 75\\,\\% & Max.",
                rows, note)


# =============================================================================
# Tabellen: statistische Prognosequalität
# =============================================================================

"""
Prognosequalität der drei Kovarianzschätzer im Basisfall: QLIKE und MSE.
"""
def table_statistik_basis():
    qlike = read_qlike(DATASET, BASE_TRAIN, BASE_PRED).mean()
    mse = read_cov_mse(DATASET, BASE_TRAIN, BASE_PRED).mean()

    rows = []
    for cov in COV_ORDER:
        rows.append(" & ".join([COV_LABEL[cov],
                                num(qlike[cov], 2),
                                num(mse[cov] * 1e7, 2)]))

    write_table("tab_statistik_basis",
                "Prognosequalität der Kovarianzschätzer im Basisfall",
                "tab:statistik_basis",
                "lrr",
                "Kovarianzschätzer & QLIKE & MSE ($\\times 10^{-7}$)",
                rows,
                note=(f"{DATASET}, Training {BASE_TRAIN} Handelstage, Prognosehorizont "
                      f"{BASE_PRED} Tag; niedrigere Werte bedeuten eine bessere Prognose."))


"""
Diebold-Mariano-West-Tests auf die paarweisen Verlustdifferenzen im Basisfall,
getrennt für QLIKE und MSE.
"""
def table_dmw():
    qlike = read_qlike(DATASET, BASE_TRAIN, BASE_PRED)
    mse = read_cov_mse(DATASET, BASE_TRAIN, BASE_PRED)
    pairs = [("GARCH", "Historical"), ("DCC", "Historical"), ("DCC", "GARCH")]

    rows = []
    for label, losses, scale, digits in [("QLIKE", qlike, 1.0, 3),
                                         ("MSE ($\\times 10^{-7}$)", mse, 1e7, 3)]:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}}")
        for a, b in pairs:
            mean, t, p, n = dmw_test((losses[a] - losses[b]) * scale)
            rows.append(" & ".join([f"\\quad {COV_LABEL[a]} vs.\\ {COV_LABEL[b]}",
                                    num(mean, digits), num(t, 2),
                                    pval(p), stars(p), str(n)]))
        rows.append("MIDRULE")
    rows = rows[:-1]

    write_table("tab_dmw",
                "Diebold-Mariano-West-Tests auf die Verlustdifferenzen (Basisfall)",
                "tab:dmw",
                "lrrr@{}lr",
                ("Vergleich & $\\varnothing\\,\\Delta$ Verlust & $t$-Statistik & "
                 "\\multicolumn{2}{c}{$p$-Wert} & Termine"),
                rows,
                note=("Negative Differenzen bedeuten, dass der erstgenannte Schätzer besser"
                      " prognostiziert; $^{***}$, $^{**}$, $^{*}$ = 1-, 5-, 10-\\%-Niveau."))


"""
QLIKE und MSE über alle Trainingsfenster und Prognosehorizonte, bewertet auf dem
gemeinsamen Zeitraum, jeweils mit der Differenz zur historischen Matrix.
"""
def table_qlike_grid():
    rows = []
    specs = ([(str(t), t, 1) for t in TRAIN_WINDOWS]
             + [("MIDRULE", None, None)]
             + [(f"$h={p}$", BASE_TRAIN, p) for p in PRED_WINDOWS])
    for label, train, pred in specs:
        if train is None:
            rows.append("MIDRULE")
            continue
        q = read_qlike(DATASET, train, pred).loc[COMMON_START:]
        m = read_cov_mse(DATASET, train, pred).loc[COMMON_START:] * 1e7
        rows.append(" & ".join([label,
                                num(q["Historical"].mean(), 1),
                                num((q["GARCH"] - q["Historical"]).mean(), 2),
                                num((q["DCC"] - q["Historical"]).mean(), 2),
                                num(m["Historical"].mean(), 2),
                                num(m["GARCH"].mean(), 2),
                                num(m["DCC"].mean(), 2)]))
    write_table("tab_qlike_grid",
                "Prognosequalität über Trainingsfenster und Prognosehorizonte (TRBC)",
                "tab:qlike_grid",
                "lrrrrrr",
                ["& \\multicolumn{3}{c}{QLIKE} & \\multicolumn{3}{c}{MSE ($\\times 10^{-7}$)} \\\\",
                 "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                 ("Spezifikation & Historisch & $\\Delta$ GARCH & $\\Delta$ DCC & "
                  "Historisch & GARCH & DCC-GARCH \\\\")],
                rows,
                note=(f"Oberer Block: Trainingsfenster in Handelstagen ($h=1$), unterer Block:"
                      f" Prognosehorizonte, bewertet ab {datum(COMMON_START)}; $\\Delta$ ist die"
                      " Differenz zur historischen Matrix."))


# =============================================================================
# Tabellen: ökonomische Performance
# =============================================================================

"""
Out-of-Sample-Performance aller Modell-Schätzer-Kombinationen im Basisfall.
"""
def table_oekonomie_basis():
    summary = read_summary(DATASET, BASE_TRAIN, BASE_PRED)
    rows = []
    for model in MODEL_ORDER:
        for cov in COV_ORDER:
            r = summary.loc[f"{model} {cov}"]
            rows.append(" & ".join([model, COV_LABEL[cov],
                                    num(r["Ann. Return"] * 100, 2),
                                    num(r["Ann. Std"] * 100, 2),
                                    num(r["Ann. Sharpe (rf=0)"], 3),
                                    num(r["Ann. Sharpe"], 3),
                                    num(r["Avg Turnover"], 3)]))
        rows.append("MIDRULE")
    r = summary.loc["Naive"]
    rows.append(" & ".join(["1/N", "--",
                            num(r["Ann. Return"] * 100, 2),
                            num(r["Ann. Std"] * 100, 2),
                            num(r["Ann. Sharpe (rf=0)"], 3),
                            num(r["Ann. Sharpe"], 3),
                            num(r["Avg Turnover"], 3)]))

    write_table("tab_oekonomie_basis",
                "Out-of-Sample-Performance im Basisfall",
                "tab:oekonomie_basis",
                "llrrrrr",
                ("Modell & Kovarianz & Rendite (\\%) & Vola (\\%) & "
                 "Sharpe ($r_f=0$) & Sharpe & Turnover"),
                rows,
                note=(f"{DATASET}, Training {BASE_TRAIN} Handelstage, Prognosehorizont {BASE_PRED} Tag;"
                      " Rendite, Volatilität und Sharpe Ratio annualisiert, Turnover ohne Kosten."))


"""
Jobson-Korkie-Tests (mit Memmel-Korrektur) für die ökonomisch relevanten
Sharpe-Differenzen im Basisfall.
"""
def table_jobson_korkie():
    returns = read_returns(DATASET, BASE_TRAIN, BASE_PRED)
    rf = backtest.load_risk_free(returns.index).fillna(0.0)
    excess = returns.sub(rf, axis=0)

    wanted = [("MVP Historical", "MVP DCC"), ("HRP Historical", "HRP DCC"),
              ("ERC Historical", "ERC DCC"), ("MVP GARCH", "MVP DCC"),
              ("MVP Historical", "Naive"), ("HRP Historical", "Naive"),
              ("ERC Historical", "Naive")]
    rows = []
    n_obs = 0
    for a, b in wanted:
        pair = excess[[a, b]].dropna()
        sr_a, sr_b, z, p, n = jobson_korkie(pair[a].values, pair[b].values)
        n_obs = n
        rows.append(" & ".join([f"{a} vs.\\ {b}",
                                num(sr_a, 3), num(sr_b, 3), num(sr_a - sr_b, 4),
                                num(z, 2), pval(p), stars(p)]))
    write_table("tab_jobson_korkie",
                "Tests auf Unterschiede der Sharpe Ratios (Basisfall)",
                "tab:jobson_korkie",
                "lrrrrr@{}l",
                ("Vergleich & Sharpe A & Sharpe B & Differenz & $z$-Statistik & "
                 "\\multicolumn{2}{c}{$p$-Wert}"),
                rows,
                note=("Auf Überschussrenditen über die Fed Funds Rate; $^{***}$, $^{**}$, $^{*}$ ="
                      " 1-, 5-, 10-\\%-Niveau."))


"""
Sharpe Ratios aller Portfoliomodelle über die Trainingsfenster.
"""
def table_sharpe_train():
    rows = []
    for train in TRAIN_WINDOWS:
        r = read_returns(DATASET, train, 1).loc[COMMON_START:]
        cells = [str(train)]
        for model in MODEL_ORDER:
            for cov in COV_ORDER:
                cells.append(num(sharpe0(r[f"{model} {cov}"]), 3))
        cells.append(num(sharpe0(r["Naive"]), 3))
        rows.append(" & ".join(cells))

    header = [
        ("Training & \\multicolumn{3}{c}{MVP} & \\multicolumn{3}{c}{HRP} & "
         "\\multicolumn{3}{c}{ERC} & 1/N \\\\"),
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        (" & Hist. & GARCH & DCC & Hist. & GARCH & DCC & Hist. & GARCH & DCC & \\\\"),
    ]

    write_table("tab_sharpe_train",
                "Sharpe Ratios über die Trainingsfenster (TRBC, $h=1$)",
                "tab:sharpe_train",
                "r" + "r" * 10,
                header, rows,
                note=(f"Annualisierte Sharpe Ratio ($r_f=0$), bewertet ab {datum(COMMON_START)}."))


"""
Sharpe Ratios und Turnover über die Prognosehorizonte.
"""
def table_sharpe_horizont():
    rows = []
    for pred in PRED_WINDOWS:
        s = read_summary(DATASET, BASE_TRAIN, pred)
        r = read_returns(DATASET, BASE_TRAIN, pred).loc[COMMON_START:]
        cells = [str(pred)]
        for model in MODEL_ORDER:
            for cov in ["Historical", "DCC"]:
                cells.append(num(sharpe0(r[f"{model} {cov}"]), 3))
        cells.append(num(sharpe0(r["Naive"]), 3))
        cells.append(num(s.loc["MVP Historical", "Avg Turnover"], 3))
        cells.append(num(s.loc["MVP DCC", "Avg Turnover"], 3))
        rows.append(" & ".join(cells))

    header = [
        ("$h$ (Tage) & \\multicolumn{2}{c}{MVP} & \\multicolumn{2}{c}{HRP} & "
         "\\multicolumn{2}{c}{ERC} & 1/N & \\multicolumn{2}{c}{Turnover MVP} \\\\"),
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){9-10}",
        (" & Hist. & DCC & Hist. & DCC & Hist. & DCC & & Hist. & DCC \\\\"),
    ]

    write_table("tab_sharpe_horizont",
                "Sharpe Ratios und Turnover über die Prognosehorizonte "
                f"(TRBC, Training {BASE_TRAIN} Tage)",
                "tab:sharpe_horizont",
                "r" + "r" * 9,
                header, rows,
                note=(f"Annualisierte Sharpe Ratio ($r_f=0$), bewertet ab {datum(COMMON_START)};"
                      " Turnover je Rebalancierung, also nicht auf gleiche Frequenz normiert."))


"""
Ergebnisse über alle 40 empirischen Läufe: wie oft ein Modell das
1/N-Portfolio schlägt, wie oft ein dynamischer Schätzer die historische Matrix
ökonomisch schlägt und wie oft er sie statistisch schlägt.
"""
def table_bilanz():
    beats_naive = {}
    beats_hist_sharpe = {}
    beats_qlike = {"GARCH": 0, "DCC": 0}
    beats_mse = {"GARCH": 0, "DCC": 0}
    n_runs = 0

    for train in TRAIN_WINDOWS:
        for pred in PRED_WINDOWS:
            if not os.path.exists(os.path.join(run_dir(DATASET, train, pred), "summary.csv")):
                continue
            n_runs += 1
            r = read_returns(DATASET, train, pred).loc[COMMON_START:]
            base = sharpe0(r["Naive"])
            for model in MODEL_ORDER:
                ref = sharpe0(r[f"{model} Historical"])
                for cov in COV_ORDER:
                    key = f"{model} {cov}"
                    sharpe = sharpe0(r[key])
                    beats_naive[key] = beats_naive.get(key, 0) + int(sharpe > base)
                    if cov != "Historical":
                        beats_hist_sharpe[key] = beats_hist_sharpe.get(key, 0) + int(sharpe > ref)
            q = read_qlike(DATASET, train, pred).loc[COMMON_START:].mean()
            m = read_cov_mse(DATASET, train, pred).loc[COMMON_START:].mean()
            for cov in ["GARCH", "DCC"]:
                beats_qlike[cov] += int(q[cov] < q["Historical"])
                beats_mse[cov] += int(m[cov] < m["Historical"])

    rows = []
    for model in MODEL_ORDER:
        for cov in COV_ORDER:
            key = f"{model} {cov}"
            eco = beats_hist_sharpe.get(key)
            rows.append(" & ".join([model, COV_LABEL[cov],
                                    f"{beats_naive[key]}/{n_runs}",
                                    "--" if eco is None else f"{eco}/{n_runs}",
                                    "--" if cov == "Historical"
                                    else f"{beats_qlike[cov]}/{n_runs}",
                                    "--" if cov == "Historical"
                                    else f"{beats_mse[cov]}/{n_runs}"]))
        rows.append("MIDRULE")
    rows = rows[:-1]

    write_table("tab_bilanz",
                "Ergebnisse über alle empirischen Läufe",
                "tab:bilanz",
                "llrrrr",
                ["Modell & Kovarianz & vs.\\ 1/N & "
                 "\\multicolumn{3}{c}{vs.\\ historische Matrix} \\\\",
                 "\\cmidrule(lr){4-6}",
                 " & & Sharpe & Sharpe & QLIKE & MSE \\\\"],
                rows,
                note=(f"Alle {n_runs} Kombinationen aus Trainingsfenster und Prognosehorizont,"
                      f" bewertet ab {datum(COMMON_START)}."))


"""
Sharpe Ratios nach Abzug proportionaler Transaktionskosten. Die Kosten je
Rebalancierung sind c mal Turnover, umgerechnet auf einen Handelstag. Da der
Backtest selbst kostenfrei rechnet, ist dies eine Ex-post-Rechnung auf Basis des
ausgewiesenen mittleren Turnovers.
"""
def table_transaktionskosten():
    summary = read_summary(DATASET, BASE_TRAIN, BASE_PRED)
    returns = read_returns(DATASET, BASE_TRAIN, BASE_PRED)

    rows = []
    for model in MODEL_ORDER + ["Naive"]:
        covs = ["N/A"] if model == "Naive" else COV_ORDER
        for cov in covs:
            key = "Naive" if model == "Naive" else f"{model} {cov}"
            r = np.asarray(returns[key], dtype=float)
            mean, sd = r.mean(), r.std(ddof=1)
            turnover = summary.loc[key, "Avg Turnover"]
            cells = ["1/N" if model == "Naive" else model,
                     "--" if cov == "N/A" else COV_LABEL[cov],
                     num(turnover, 3)]
            for cost_bp in COST_LEVELS:
                daily_cost = (cost_bp / 10000.0) * turnover / BASE_PRED
                cells.append(num((mean - daily_cost) * TRADING_DAYS
                                 / (sd * np.sqrt(TRADING_DAYS)), 3))
            rows.append(" & ".join(cells))
        if model != "Naive":
            rows.append("MIDRULE")

    header = ("Modell & Kovarianz & Turnover & "
              + " & ".join(f"{c} Bp." for c in COST_LEVELS))
    write_table("tab_transaktionskosten",
                "Sharpe Ratios nach proportionalen Transaktionskosten (Basisfall)",
                "tab:transaktionskosten",
                "llr" + "r" * len(COST_LEVELS),
                header, rows,
                note=("Sharpe Ratio ($r_f=0$) nach Abzug von $c$ Basispunkten auf das je"
                      " Rebalancierung umgeschlagene Volumen."))


# =============================================================================
# Tabellen: Strukturbruch
# =============================================================================

"""
Kennzahlen je Teilperiode des Basisfalls, zusammen mit der Größe des
investierbaren Universums. Zeigt den Niveaubruch des QLIKE und die
Instabilität der ökonomischen Rangfolge.
"""
def table_strukturbruch_teilperioden(log_returns):
    qlike = read_qlike(DATASET, BASE_TRAIN, BASE_PRED)
    mse = read_cov_mse(DATASET, BASE_TRAIN, BASE_PRED) * 1e7
    returns = read_returns(DATASET, BASE_TRAIN, BASE_PRED)
    sizes = universe_sizes(log_returns, BASE_TRAIN, step=1)

    rows = []
    for label, start, end in SUB_PERIODS:
        q = qlike.loc[start:end]
        m = mse.loc[start:end]
        r = returns.loc[start:end]
        n = sizes.loc[start:end]
        if len(q) == 0:
            continue
        rows.append(" & ".join([label,
                                f"{int(n.min())}--{int(n.max())}",
                                num(q["Historical"].mean(), 1),
                                num((q["DCC"] - q["Historical"]).mean(), 2),
                                num(m["Historical"].mean(), 2),
                                num(m["DCC"].mean(), 2),
                                num(sharpe0(r["MVP Historical"]), 3),
                                num(sharpe0(r["MVP DCC"]), 3),
                                num(sharpe0(r["Naive"]), 3)]))
    write_table("tab_strukturbruch_teilperioden",
                "Teilperioden des Basisfalls: Universum, Prognosequalität und Performance",
                "tab:strukturbruch_teilperioden",
                "llrrrrrrr",
                ["Periode & $N$ & \\multicolumn{2}{c}{QLIKE} & "
                 "\\multicolumn{2}{c}{MSE ($\\times 10^{-7}$)} & "
                 "\\multicolumn{3}{c}{Sharpe} \\\\",
                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
                 (" & & Hist. & $\\Delta$ DCC & Hist. & DCC & MVP Hist. & MVP DCC & 1/N \\\\")],
                rows,
                note=("$N$ ist die Spannweite der investierbaren Sektoren; QLIKE-Niveaus sind"
                      " zwischen Perioden mit unterschiedlichem $N$ nicht vergleichbar."))


"""
Vergleich von vollem und gemeinsamem Bewertungszeitraum über die
Trainingsfenster. Trennt den Stichprobeneffekt vom Schätzeffekt.
"""
def table_strukturbruch_gemeinsam(log_returns):
    rows = []
    n_full = []
    n_common = []
    for train in TRAIN_WINDOWS:
        q = read_qlike(DATASET, train, 1)
        r = read_returns(DATASET, train, 1)
        qc = q.loc[COMMON_START:]
        rc = r.loc[COMMON_START:]
        sizes = universe_sizes(log_returns, train)
        rows.append(" & ".join([str(train),
                                num(sizes.loc[COMMON_START:].mean(), 1),
                                num(q["DCC"].mean(), 1), num(qc["DCC"].mean(), 1),
                                num(sharpe0(r["MVP Historical"]), 3),
                                num(sharpe0(rc["MVP Historical"]), 3),
                                num(sharpe0(r["Naive"]), 3),
                                num(sharpe0(rc["Naive"]), 3)]))
        n_full.append(sharpe0(r["MVP Historical"]))
        n_common.append(sharpe0(rc["MVP Historical"]))

    # Korrelation zwischen mittlerer Universumsgröße und QLIKE-Niveau
    sizes_mean = [universe_sizes(log_returns, t).loc[COMMON_START:].mean()
                  for t in TRAIN_WINDOWS]
    qlike_level = [read_qlike(DATASET, t, 1).loc[COMMON_START:, "DCC"].mean()
                   for t in TRAIN_WINDOWS]
    rho = np.corrcoef(sizes_mean, qlike_level)[0, 1]

    note = (f"\\emph{{voll}} bezeichnet die volle Historie des jeweiligen Fensters,"
            f" \\emph{{gem.}} den gemeinsamen Zeitraum ab {datum(COMMON_START)}.")

    write_table("tab_strukturbruch_gemeinsam",
                "Voller versus gemeinsamer Bewertungszeitraum (TRBC, $h=1$)",
                "tab:strukturbruch_gemeinsam",
                "rrrrrrrr",
                ["Training & $\\varnothing\\,N$ & \\multicolumn{2}{c}{QLIKE DCC} & "
                 "\\multicolumn{2}{c}{Sharpe MVP Hist.} & "
                 "\\multicolumn{2}{c}{Sharpe 1/N} \\\\",
                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
                 (" & & voll & gem. & voll & gem. & voll & gem. \\\\")],
                rows, note)


# =============================================================================
# Tabellen: Anhang, synthetische Validierung
# =============================================================================

"""
Basisfall der drei synthetischen Datensätze: Prognosequalität und
MVP-Performance je Kovarianzschätzer.
"""
def table_synth_basis():
    rows = []
    for dataset in SYNTH_DATASETS:
        summary = read_summary(dataset, BASE_TRAIN, BASE_PRED)
        mse = read_cov_mse(dataset, BASE_TRAIN, BASE_PRED).mean() * 1e8
        for cov in COV_ORDER:
            mvp = summary.loc[f"MVP {cov}"]
            rows.append(" & ".join([SYNTH_LABEL[dataset] if cov == COV_ORDER[0] else "",
                                    COV_LABEL[cov],
                                    num(mvp["Avg QLIKE"], 2),
                                    num(mse[cov], 2),
                                    num(mvp["Ann. Std"] * 100, 2),
                                    num(mvp["Ann. Sharpe (rf=0)"], 3),
                                    num(mvp["Avg Turnover"], 3)]))
        rows.append("MIDRULE")
    rows = rows[:-1]
    write_table("tab_synth_basis",
                "Synthetische Validierung: Basisfall je Datengenerierungsprozess",
                "tab:synth_basis",
                "llrrrrr",
                ("Datensatz & Kovarianz & QLIKE & MSE ($\\times 10^{-8}$) & "
                 "Vola (\\%) & Sharpe & Turnover"),
                rows,
                note=(f"20 simulierte Anlagen, Training {BASE_TRAIN} Handelstage, Prognosehorizont"
                      f" {BASE_PRED} Tag; Performancekennzahlen für das MVP."))


"""
DMW-Tests auf den synthetischen Datensätzen: bestätigt, dass der jeweils
korrekt spezifizierte Schätzer gewinnt.
"""
def table_synth_dmw():
    rows = []
    for dataset in SYNTH_DATASETS:
        q = read_qlike(dataset, BASE_TRAIN, BASE_PRED)
        cells = [SYNTH_LABEL[dataset]]
        for a, b in [("GARCH", "Historical"), ("DCC", "Historical"), ("DCC", "GARCH")]:
            mean, t, p, _ = dmw_test(q[a] - q[b])
            cells.append(num(mean, 3))
            cells.append(stars(p))
            cells.append(num(t, 2))
        rows.append(" & ".join(cells))
    write_table("tab_synth_dmw",
                "Synthetische Validierung: DMW-Tests auf die QLIKE-Differenzen",
                "tab:synth_dmw",
                "lr@{}lrr@{}lrr@{}lr",
                ("Datensatz & \\multicolumn{2}{c}{$\\Delta$ G--H} & $t$ & "
                 "\\multicolumn{2}{c}{$\\Delta$ D--H} & $t$ & "
                 "\\multicolumn{2}{c}{$\\Delta$ D--G} & $t$"),
                rows,
                note=("G = GARCH, D = DCC-GARCH, H = historisch; negative Differenzen bedeuten eine"
                      " bessere Prognose. $^{***}$, $^{**}$, $^{*}$ = 1-, 5-, 10-\\%-Niveau."))


"""
Stabilität der synthetischen Ergebnisse über die Trainingsfenster. Gegenstück
zu Tabelle tab:strukturbruch_gemeinsam: Hier existiert kein Strukturbruch.
"""
def table_synth_stabilitaet():
    rows = []
    spans = {}
    for dataset in ["GARCH_sim", "DCC_sim"]:
        levels = []
        for train in TRAIN_WINDOWS:
            path = os.path.join(run_dir(dataset, train, 1), "qlike.csv")
            if not os.path.exists(path):
                continue
            q = read_qlike(dataset, train, 1).loc[COMMON_START:]
            m = read_cov_mse(dataset, train, 1).loc[COMMON_START:] * 1e8
            r = read_returns(dataset, train, 1).loc[COMMON_START:]
            levels.append(q["Historical"].mean())
            rows.append(" & ".join([SYNTH_LABEL[dataset] if train == TRAIN_WINDOWS[0] else "",
                                    str(train), "20",
                                    num(q["Historical"].mean(), 1),
                                    num((q["DCC"] - q["Historical"]).mean(), 2),
                                    num(m["Historical"].mean(), 2),
                                    num(m["DCC"].mean(), 2),
                                    num(sharpe0(r["MVP Historical"]), 3),
                                    num(sharpe0(r["MVP DCC"]), 3)]))
        spans[dataset] = max(levels) - min(levels)
        rows.append("MIDRULE")
    rows = rows[:-1]
    write_table("tab_synth_stabilitaet",
                "Synthetische Validierung: Stabilität über die Trainingsfenster",
                "tab:synth_stabilitaet",
                "lrrrrrrrr",
                ["Datensatz & Training & $N$ & \\multicolumn{2}{c}{QLIKE} & "
                 "\\multicolumn{2}{c}{MSE ($\\times 10^{-8}$)} & "
                 "\\multicolumn{2}{c}{Sharpe MVP} \\\\",
                 "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                 (" & & & Hist. & $\\Delta$ DCC & Hist. & DCC & Hist. & DCC \\\\")],
                rows,
                note=(f"Bewertet ab {datum(COMMON_START)}; das Universum ist mit 20 Anlagen konstant."))


# =============================================================================
# Ablauf
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Lade TRBC-Kursdaten ...")
    _, log_returns, _ = backtest.load_dataset(DATASET)

    print("Daten:")
    table_daten_deskriptiv(log_returns)

    print("Statistische Ergebnisse:")
    table_statistik_basis()
    table_dmw()
    table_qlike_grid()

    print("Ökonomische Ergebnisse:")
    table_oekonomie_basis()
    table_jobson_korkie()
    table_sharpe_train()
    table_sharpe_horizont()
    table_bilanz()
    table_transaktionskosten()

    print("Strukturbruch:")
    table_strukturbruch_teilperioden(log_returns)
    table_strukturbruch_gemeinsam(log_returns)

    print("Anhang (synthetische Validierung):")
    table_synth_basis()
    table_synth_dmw()
    table_synth_stabilitaet()

    print("Fertig.")


if __name__ == "__main__":
    main()
