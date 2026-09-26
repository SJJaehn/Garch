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

# Gemeinsamer Bewertungszeitraum für die empirischen Vergleiche über
# Trainingsfenster und Prognosehorizonte hinweg. Das längste Fenster (2520
# Tage) startet erst im November 2008 und würde den Einbruch der Finanzkrise
# nicht mehr, die Erholung danach aber vollständig enthalten -- das allein hebt
# seine Sharpe Ratio gegenüber kurzen Fenstern an. Ab 2010 sind Einbruch und
# Erholung für alle Fenster gleichermaßen ausgeschlossen. Die synthetischen
# Tabellen rechnen dagegen über die volle Stichprobe, da dort das Universum
# konstant ist und die Parameter des Prozesses sich nicht ändern.
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
    # breite Tabellen werden auf die Textbreite gestaucht, statt in den Rand zu
    # laufen. Schmalere Tabellen bleiben unveraendert, sonst wuerde resizebox
    # sie hochskalieren und ihre Schrift groesser setzen als den Fliesstext.
    if wide:
        lines.append("  \\resizebox{\\ifdim\\width>\\linewidth \\linewidth"
                     "\\else \\width\\fi}{!}{%")
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
Der Kursstand der Fed Funds Rate wird nur einmal gelesen und danach
wiederverwendet, weil die Sharpe Ratio fuer sehr viele Laeufe gebraucht wird.
"""
RF_LEVEL = None


"""
Taegliche einfache risikofreie Rendite, ausgerichtet auf die uebergebenen Daten.
"""
def risk_free(dates):
    global RF_LEVEL
    if RF_LEVEL is None:
        RF_LEVEL = backtest.read_risk_free_level()
    return RF_LEVEL.reindex(dates).ffill().pct_change().fillna(0.0)


"""
Annualisierte Sharpe Ratio auf Ueberschussrenditen ueber die Fed Funds Rate,
identisch zur Spalte "Ann. Sharpe" in summary.csv.
"""
def sharpe(returns):
    excess = pd.Series(returns) - risk_free(pd.Series(returns).index)
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return (r.mean() * TRADING_DAYS) / sd if sd > 0 else np.nan


"""
Notiz-Baustein fuer die Tabellen des Basisfalls. Diese werden ueber die volle
Historie des Basisfensters gerechnet und nicht ab COMMON_START, weshalb ihre
Werte von den Tabellen ueber mehrere Trainingsfenster abweichen.
"""
def basis_zeitraum():
    index = read_returns(DATASET, BASE_TRAIN, BASE_PRED).index
    return f"bewertet über die volle Historie vom {datum(index[0])} bis {datum(index[-1])}"


"""
Annualisierte realisierte Volatilität in Prozent.
"""
def realized_vol(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100


"""
Diebold-Mariano-Test auf die mittlere Verlustdifferenz d_t = L_a - L_b.
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

Die historische Kovarianzmatrix ist an einzelnen Terminen singulär, weshalb
QLIKE dort nicht definiert ist (NaN). Ein spaltenweiser Mittelwert würde
GARCH und DCC dann über mehr Termine mitteln als die historische Matrix und
die drei Schätzer wären nicht mehr auf derselben Terminmenge vergleichbar.
Gemittelt wird deshalb nur über die Termine, an denen alle drei QLIKE-Werte
definiert sind (dieselbe Terminmenge wie im DM-Test von tab_dmw). Der MSE hat
an keinem Termin fehlende Werte.
"""
def table_statistik_basis():
    mse_df = read_cov_mse(DATASET, BASE_TRAIN, BASE_PRED)[COV_ORDER]
    qlike_common = read_qlike(DATASET, BASE_TRAIN, BASE_PRED)[COV_ORDER].dropna()
    qlike = qlike_common.mean()
    mse = mse_df.mean()
    n_qlike = len(qlike_common)
    n_mse = len(mse_df)

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
                      f"{BASE_PRED} Tag, {basis_zeitraum()};"
                      f" niedrigere Werte bedeuten eine bessere Prognose."
                      f" QLIKE gemittelt über {n_qlike} Termine, an denen alle drei Schätzer"
                      f" definiert sind (die historische Matrix ist an einzelnen Terminen"
                      f" singulär); MSE über alle {n_mse} Termine, an denen dieser Ausfall"
                      f" nicht auftritt."))


"""
Diebold-Mariano-Tests auf die paarweisen Verlustdifferenzen im Basisfall,
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
                "Diebold-Mariano-Tests auf die Verlustdifferenzen (Basisfall)",
                "tab:dmw",
                "lrrr@{}lr",
                ("Vergleich & $\\varnothing\\,\\Delta$ Verlust & $t$-Statistik & "
                 "\\multicolumn{2}{c}{$p$-Wert} & Termine"),
                rows,
                note=(f"Basisfall, {basis_zeitraum()}."
                      " Negative Differenzen bedeuten, dass der erstgenannte Schätzer besser"
                      " prognostiziert; $^{***}$, $^{**}$, $^{*}$ = 1-, 5-, 10-\\%-Niveau."))


"""
QLIKE und MSE über alle Trainingsfenster und Prognosehorizonte, bewertet auf dem
gemeinsamen Zeitraum.
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
        q = read_qlike(DATASET, train, pred).loc[COMMON_START:][COV_ORDER].dropna()
        m = read_cov_mse(DATASET, train, pred).loc[COMMON_START:] * 1e7
        rows.append(" & ".join([label,
                                num(q["Historical"].mean(), 1),
                                num(q["GARCH"].mean(), 1),
                                num(q["DCC"].mean(), 1),
                                num(m["Historical"].mean(), 2),
                                num(m["GARCH"].mean(), 2),
                                num(m["DCC"].mean(), 2)]))
    write_table("tab_qlike_grid",
                "Prognosequalität über Trainingsfenster und Prognosehorizonte (TRBC)",
                "tab:qlike_grid",
                "lrrrrrr",
                ["& \\multicolumn{3}{c}{QLIKE} & \\multicolumn{3}{c}{MSE ($\\times 10^{-7}$)} \\\\",
                 "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                 ("Spezifikation & Historisch & GARCH & DCC-GARCH & "
                  "Historisch & GARCH & DCC-GARCH \\\\")],
                rows,
                note=(f"Beide Blöcke bewertet ab {datum(COMMON_START)}. Oberer Block:"
                      f" Trainingsfenster in Handelstagen bei $h=1$, unterer Block:"
                      f" Prognosehorizonte bei {BASE_TRAIN} Tagen Training."))


# =============================================================================
# Tabellen: ökonomische Performance
# =============================================================================

"""
Durchschnittliche Anzahl der Anlagen mit einem Gewicht von mehr als 1 % je
Formationstermin. Gelesen wird die Gewichtshistorie des Python-Backtests
(weights.csv, wird bei LOG_WEIGHTS = True in main.py geschrieben).
Zuvor stand hier die Gewichtshistorie der R-Validierungsrechnung
(weights_R.csv). Beide stimmen bei Historisch und bei ERC exakt überein, bei
MVP mit GARCH zählt Python im Schnitt 0,3 bis 0,7 Anlagen mehr über der
1-%-Schwelle -- derselbe Unterschied zwischen arch und rugarch, den die
Validierung ohnehin ausweist.
"""
def read_n_positions(dataset, train, pred, threshold=0.01):
    path = os.path.join(run_dir(dataset, train, pred), "weights.csv")
    w = pd.read_csv(path, parse_dates=["Date"])
    w["Covariance Type"] = w["Covariance Type"].fillna("N/A")
    n = w.groupby(["Date", "Model", "Covariance Type"])["Weight"].apply(
        lambda x: int((np.asarray(x, dtype=float) > threshold).sum()))
    return n.groupby(["Model", "Covariance Type"]).mean()


"""
Out-of-Sample-Performance aller Modell-Schätzer-Kombinationen im Basisfall.
"""
def table_oekonomie_basis():
    summary = read_summary(DATASET, BASE_TRAIN, BASE_PRED)
    n_pos = read_n_positions(DATASET, BASE_TRAIN, BASE_PRED)
    rows = []
    for model in MODEL_ORDER:
        for cov in COV_ORDER:
            r = summary.loc[f"{model} {cov}"]
            rows.append(" & ".join([model, COV_LABEL[cov],
                                    num(r["Ann. Return"] * 100, 2),
                                    num(r["Ann. Std"] * 100, 2),
                                    num(r["Ann. Sharpe"], 3),
                                    num(r["Avg Turnover"], 3),
                                    num(n_pos[(model, cov)], 1)]))
        rows.append("MIDRULE")
    r = summary.loc["Naive"]
    rows.append(" & ".join(["1/N", "--",
                            num(r["Ann. Return"] * 100, 2),
                            num(r["Ann. Std"] * 100, 2),
                            num(r["Ann. Sharpe"], 3),
                            num(r["Avg Turnover"], 3),
                            num(n_pos[("Naive", "N/A")], 1)]))

    write_table("tab_oekonomie_basis",
                "Out-of-Sample-Performance im Basisfall",
                "tab:oekonomie_basis",
                "llrrrrr",
                ("Modell & Kovarianz & Rendite (\\%) & Vola (\\%) & "
                 "Sharpe & Turnover & Anlagen $>1\\,\\%$"),
                rows,
                note=(f"{DATASET}, Training {BASE_TRAIN} Handelstage, Prognosehorizont {BASE_PRED} Tag,"
                      f" {basis_zeitraum()};"
                      " Rendite, Volatilität und Sharpe Ratio annualisiert, Turnover ohne Kosten;"
                      " Anlagen $>1\\,\\%$ ist die durchschnittliche Zahl der Sektoren mit einem"
                      " Gewicht von mehr als 1\\,\\% je Formationstermin."))


"""
Übersetzt eine interne Strategiebezeichnung ("MVP Historical", "Naive", ...)
in die deutsche Beschriftung, wie sie auch in den übrigen Tabellen verwendet
wird ("MVP Historisch", "1/N").
"""
def strategie_label(key):
    if key == "Naive":
        return "1/N"
    model, cov = key.split(" ", 1)
    return f"{model} {COV_LABEL[cov]}"


"""
Jobson-Korkie-Tests (mit Memmel-Korrektur) für die ökonomisch relevanten
Sharpe-Differenzen im Basisfall.
"""
def table_jobson_korkie():
    returns = read_returns(DATASET, BASE_TRAIN, BASE_PRED)
    rf = backtest.load_risk_free(returns.index).fillna(0.0)
    excess = returns.sub(rf, axis=0)

    wanted = []
    for model in MODEL_ORDER:
        wanted.append((f"{model} Historical", f"{model} GARCH"))
        wanted.append((f"{model} Historical", f"{model} DCC"))
        wanted.append((f"{model} GARCH", f"{model} DCC"))
    for model in MODEL_ORDER:
        for cov in COV_ORDER:
            wanted.append((f"{model} {cov}", "Naive"))
    rows = []
    n_obs = 0
    for a, b in wanted:
        pair = excess[[a, b]].dropna()
        sr_a, sr_b, z, p, n = jobson_korkie(pair[a].values, pair[b].values)
        n_obs = n
        rows.append(" & ".join([f"{strategie_label(a)} vs.\\ {strategie_label(b)}",
                                num(sr_a - sr_b, 4),
                                num(z, 2), pval(p), stars(p)]))
    write_table("tab_jobson_korkie",
                "Tests auf Unterschiede der Sharpe Ratios (Basisfall)",
                "tab:jobson_korkie",
                "lrrr@{}l",
                ("Vergleich & $\\Delta$ Sharpe & $z$-Statistik & "
                 "\\multicolumn{2}{c}{$p$-Wert}"),
                rows,
                note=(f"Basisfall, {basis_zeitraum()}."
                      " Auf Überschussrenditen über die Fed Funds Rate; $^{***}$, $^{**}$, $^{*}$ ="
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
                cells.append(num(sharpe(r[f"{model} {cov}"]), 3))
        cells.append(num(sharpe(r["Naive"]), 3))
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
                note=(f"Annualisierte Sharpe Ratio, bewertet ab {datum(COMMON_START)}."))


"""
Sharpe Ratios und Turnover über die Prognosehorizonte.
"""
def table_sharpe_horizont():
    rows = []
    for pred in PRED_WINDOWS:
        s = read_summary(DATASET, BASE_TRAIN, pred)
        r = read_returns(DATASET, BASE_TRAIN, pred).loc[COMMON_START:]
        for k, cov in enumerate(COV_ORDER):
            cells = [str(pred) if k == 0 else "",
                     num(sharpe(r["Naive"]), 3) if k == 0 else "",
                     COV_LABEL[cov]]
            for model in MODEL_ORDER:
                cells.append(num(sharpe(r[f"{model} {cov}"]), 3))
            cells.append(num(s.loc[f"MVP {cov}", "Avg Turnover"], 3))
            rows.append(" & ".join(cells))
        rows.append("MIDRULE")
    rows = rows[:-1]

    header = "$h$ (Tage) & 1/N & Kovarianz & MVP & HRP & ERC & Turnover MVP"

    write_table("tab_sharpe_horizont",
                "Sharpe Ratios und Turnover über die Prognosehorizonte "
                f"(TRBC, Training {BASE_TRAIN} Tage)",
                "tab:sharpe_horizont",
                "rrlrrrr",
                header, rows,
                note=(f"Annualisierte Sharpe Ratio, bewertet ab {datum(COMMON_START)};"
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
            base = sharpe(r["Naive"])
            for model in MODEL_ORDER:
                ref = sharpe(r[f"{model} Historical"])
                for cov in COV_ORDER:
                    key = f"{model} {cov}"
                    wert = sharpe(r[key])
                    beats_naive[key] = beats_naive.get(key, 0) + int(wert > base)
                    if cov != "Historical":
                        beats_hist_sharpe[key] = beats_hist_sharpe.get(key, 0) + int(wert > ref)
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
                                    "--" if eco is None else f"{eco}/{n_runs}"]))
        rows.append("MIDRULE")
    rows = rows[:-1]

    write_table("tab_bilanz",
                "Ergebnisse über alle empirischen Läufe",
                "tab:bilanz",
                "llrr",
                "Modell & Kovarianz & vs.\\ 1/N & vs.\\ historischer Matrix",
                rows,
                note=(f"Alle {n_runs} Kombinationen aus Trainingsfenster und Prognosehorizont,"
                      f" bewertet ab {datum(COMMON_START)}; ausgewiesen ist, wie oft die"
                      f" Sharpe Ratio höher ausfällt als beim jeweiligen Vergleichsmaßstab."))


"""
Sharpe Ratios nach Abzug proportionaler Transaktionskosten. Die Kosten je
Rebalancierung sind c mal Turnover, umgerechnet auf einen Handelstag und von
der Rendite vor Bildung der Überschussrendite abgezogen, damit die 0-Bp.-Spalte
exakt derselben Sharpe-Definition folgt wie tab_oekonomie_basis (Gleichung 15,
Überschussrendite über die Fed Funds Rate). Da der Backtest selbst kostenfrei
rechnet, ist dies eine Ex-post-Rechnung auf Basis des ausgewiesenen mittleren
Turnovers.
"""
def table_transaktionskosten():
    summary = read_summary(DATASET, BASE_TRAIN, BASE_PRED)
    returns = read_returns(DATASET, BASE_TRAIN, BASE_PRED)

    rows = []
    for model in MODEL_ORDER + ["Naive"]:
        covs = ["N/A"] if model == "Naive" else COV_ORDER
        for cov in covs:
            key = "Naive" if model == "Naive" else f"{model} {cov}"
            r = pd.Series(returns[key])
            rf = risk_free(r.index)
            turnover = summary.loc[key, "Avg Turnover"]
            cells = ["1/N" if model == "Naive" else model,
                     "--" if cov == "N/A" else COV_LABEL[cov],
                     num(turnover, 3)]
            for cost_bp in COST_LEVELS:
                daily_cost = (cost_bp / 10000.0) * turnover / BASE_PRED
                excess = np.asarray(r - daily_cost - rf, dtype=float)
                excess = excess[np.isfinite(excess)]
                sd = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
                sr = (excess.mean() * TRADING_DAYS) / sd if sd > 0 else np.nan
                cells.append(num(sr, 3))
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
                note=("Sharpe Ratio auf Überschussrenditen über die Fed Funds Rate nach Abzug"
                      " von $c$ Basispunkten auf das je Rebalancierung umgeschlagene Volumen;"
                      " die 0-Bp.-Spalte entspricht damit exakt"
                      " \\hyperref[tab:oekonomie_basis]{Tabelle~\\ref*{tab:oekonomie_basis}}."))


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
                                num(q["GARCH"].mean(), 1),
                                num(q["DCC"].mean(), 1),
                                num(m["Historical"].mean(), 2),
                                num(m["GARCH"].mean(), 2),
                                num(m["DCC"].mean(), 2),
                                num(sharpe(r["MVP Historical"]), 3),
                                num(sharpe(r["MVP GARCH"]), 3),
                                num(sharpe(r["MVP DCC"]), 3),
                                num(sharpe(r["Naive"]), 3)]))
    write_table("tab_strukturbruch_teilperioden",
                "Teilperioden des Basisfalls: Universum, Prognosequalität und Performance",
                "tab:strukturbruch_teilperioden",
                "llrrrrrrrrrr",
                ["Periode & $N$ & \\multicolumn{3}{c}{QLIKE} & "
                 "\\multicolumn{3}{c}{MSE ($\\times 10^{-7}$)} & "
                 "\\multicolumn{4}{c}{Sharpe} \\\\",
                 "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-12}",
                 (" & & Hist. & GARCH & DCC & Hist. & GARCH & DCC"
                  " & MVP Hist. & MVP GARCH & MVP DCC & 1/N \\\\")],
                rows, wide=True,
                note=("$N$ ist die Spannweite der investierbaren Sektoren; QLIKE-Niveaus sind"
                      " zwischen Perioden mit unterschiedlichem $N$ nicht vergleichbar."))


"""
Voller versus gemeinsamer Bewertungszeitraum, QLIKE-Teil: je Trainingsfenster
der QLIKE-Verlust aller drei Schätzer, einmal über die volle Historie des
Fensters und einmal ab COMMON_START. Gemittelt wird über die gemeinsame
Terminmenge aller drei Schätzer (siehe table_statistik_basis).
"""
def table_strukturbruch_qlike(log_returns):
    rows = []
    for train in TRAIN_WINDOWS:
        q = read_qlike(DATASET, train, 1)[COV_ORDER].dropna()
        qc = q.loc[COMMON_START:]
        sizes = universe_sizes(log_returns, train)
        cells = [str(train), num(sizes.loc[COMMON_START:].mean(), 1)]
        for cov in COV_ORDER:
            cells += [num(q[cov].mean(), 1), num(qc[cov].mean(), 1)]
        rows.append(" & ".join(cells))

    note = (f"\\emph{{voll}} bezeichnet die volle Historie des jeweiligen Fensters,"
            f" \\emph{{gem.}} den gemeinsamen Zeitraum ab {datum(COMMON_START)}."
            f" $\\varnothing\\,N$ ist die mittlere Zahl investierbarer Sektoren im"
            f" gemeinsamen Zeitraum.")

    write_table("tab_strukturbruch_qlike",
                "QLIKE über vollen und gemeinsamen Bewertungszeitraum (TRBC, $h=1$)",
                "tab:strukturbruch_qlike",
                "rr" + "rr" * len(COV_ORDER),
                ["Training & $\\varnothing\\,N$ & "
                 + " & ".join(f"\\multicolumn{{2}}{{c}}{{{COV_LABEL[c]}}}" for c in COV_ORDER)
                 + " \\\\",
                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
                 " & & " + " & ".join(["voll & gem."] * len(COV_ORDER)) + " \\\\"],
                rows, note, wide=True)


"""
Voller versus gemeinsamer Bewertungszeitraum, ökonomischer Teil: die Sharpe
Ratio des MVP mit allen drei Kovarianzschätzern und des 1/N-Portfolios.
"""
def table_strukturbruch_sharpe():
    rows = []
    for train in TRAIN_WINDOWS:
        r = read_returns(DATASET, train, 1)
        rc = r.loc[COMMON_START:]
        cells = [str(train)]
        for cov in COV_ORDER:
            cells += [num(sharpe(r[f"MVP {cov}"]), 3), num(sharpe(rc[f"MVP {cov}"]), 3)]
        cells += [num(sharpe(r["Naive"]), 3), num(sharpe(rc["Naive"]), 3)]
        rows.append(" & ".join(cells))

    note = (f"\\emph{{voll}} bezeichnet die volle Historie des jeweiligen Fensters,"
            f" \\emph{{gem.}} den gemeinsamen Zeitraum ab {datum(COMMON_START)}."
            f" Ausgewiesen ist die annualisierte Sharpe Ratio.")

    write_table("tab_strukturbruch_sharpe",
                "Sharpe Ratio über vollen und gemeinsamen Bewertungszeitraum (TRBC, $h=1$)",
                "tab:strukturbruch_sharpe",
                "r" + "rr" * (len(COV_ORDER) + 1),
                ["Training & "
                 + " & ".join(f"\\multicolumn{{2}}{{c}}{{MVP {COV_LABEL[c]}}}" for c in COV_ORDER)
                 + " & \\multicolumn{2}{c}{1/N} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
                 "\\cmidrule(lr){8-9}",
                 " & " + " & ".join(["voll & gem."] * (len(COV_ORDER) + 1)) + " \\\\"],
                rows, note, wide=True)


"""
Kompakte Fassung des Vergleichs von vollem und gemeinsamem Bewertungszeitraum
für den Fliesstext: je Kennzahl nur die Spannweite über die zehn
Trainingsfenster. Die vollständige Tabelle steht im Anhang.
"""
def table_strukturbruch_kompakt(log_returns):
    garch_full = []
    garch_common = []
    qlike_full = []
    qlike_common = []
    mvp_full = []
    mvp_common = []
    naive_full = []
    naive_common = []
    for train in TRAIN_WINDOWS:
        q = read_qlike(DATASET, train, 1)
        r = read_returns(DATASET, train, 1)
        garch_full.append(q["GARCH"].mean())
        garch_common.append(q.loc[COMMON_START:, "GARCH"].mean())
        qlike_full.append(q["DCC"].mean())
        qlike_common.append(q.loc[COMMON_START:, "DCC"].mean())
        mvp_full.append(sharpe(r["MVP Historical"]))
        mvp_common.append(sharpe(r.loc[COMMON_START:, "MVP Historical"]))
        naive_full.append(sharpe(r["Naive"]))
        naive_common.append(sharpe(r.loc[COMMON_START:, "Naive"]))

    def spanne(werte):
        a = np.asarray(werte, dtype=float)
        return np.nanmax(a) - np.nanmin(a)

    rows = [" & ".join(["QLIKE GARCH", num(spanne(garch_full), 1),
                        num(spanne(garch_common), 1)]),
            " & ".join(["QLIKE DCC-GARCH", num(spanne(qlike_full), 1),
                        num(spanne(qlike_common), 1)]),
            " & ".join(["Sharpe MVP (historisch)", num(spanne(mvp_full), 3),
                        num(spanne(mvp_common), 3)]),
            " & ".join(["Sharpe 1/N", num(spanne(naive_full), 3),
                        num(spanne(naive_common), 3)])]

    note = (f"Spannweite (Maximum minus Minimum) über die zehn Trainingsfenster."
            f" \\emph{{voll}} bezeichnet die volle Historie des jeweiligen Fensters,"
            f" \\emph{{gem.}} den gemeinsamen Zeitraum ab {datum(COMMON_START)}."
            f" Die vollständigen Werte stehen in"
            f" \\hyperref[tab:strukturbruch_qlike]{{Tabelle~\\ref*{{tab:strukturbruch_qlike}}}}.")

    write_table("tab_strukturbruch_kompakt",
                "Spannweite der Kennzahlen über die Trainingsfenster (TRBC, $h=1$)",
                "tab:strukturbruch_kompakt",
                "lrr",
                "Kennzahl & voll & gem.",
                rows, note)


"""
Auszählung über alle 40 Kombinationen: wie oft liefert welcher
Kovarianzschätzer den besten Wert. Für die Sharpe Ratio je Portfoliomodell der
höchste Wert, für QLIKE und den Kovarianz-MSE der niedrigste Verlust.
"""
def table_bestzaehlung():
    zaehler = {}
    for key in MODEL_ORDER + ["QLIKE", "MSE"]:
        zaehler[key] = dict((cov, 0) for cov in COV_ORDER)
    dcc_vs_hist = {"QLIKE": 0, "MSE": 0}

    for train in TRAIN_WINDOWS:
        for pred in PRED_WINDOWS:
            ret = read_returns(DATASET, train, pred).loc[COMMON_START:]
            for model in MODEL_ORDER:
                werte = dict((cov, sharpe(ret[f"{model} {cov}"])) for cov in COV_ORDER)
                zaehler[model][max(werte, key=werte.get)] += 1

            # QLIKE nur über Termine gemittelt, an denen alle drei Schätzer
            # definiert sind (die historische Matrix ist an einzelnen Terminen
            # singulär, siehe table_statistik_basis).
            q = read_qlike(DATASET, train, pred).loc[COMMON_START:][COV_ORDER].dropna().mean()
            m = read_cov_mse(DATASET, train, pred).loc[COMMON_START:].mean()
            zaehler["QLIKE"][min(COV_ORDER, key=lambda c: q[c])] += 1
            zaehler["MSE"][min(COV_ORDER, key=lambda c: m[c])] += 1
            dcc_vs_hist["QLIKE"] += int(q["DCC"] < q["Historical"])
            dcc_vs_hist["MSE"] += int(m["DCC"] < m["Historical"])

    n = len(TRAIN_WINDOWS) * len(PRED_WINDOWS)
    beschriftung = dict((m, f"Sharpe {m}") for m in MODEL_ORDER)
    beschriftung["QLIKE"] = "QLIKE"
    beschriftung["MSE"] = "Kovarianz-MSE"

    rows = []
    for key in MODEL_ORDER:
        rows.append(" & ".join([beschriftung[key]]
                               + [f"{zaehler[key][cov]}/{n}" for cov in COV_ORDER]))
    rows.append("MIDRULE")
    for key in ["QLIKE", "MSE"]:
        rows.append(" & ".join([beschriftung[key]]
                               + [f"{zaehler[key][cov]}/{n}" for cov in COV_ORDER]))
    rows.append("MIDRULE")
    rows.append(" & ".join(["QLIKE (DCC vs.\\ Hist.)", "--", "--",
                            f"{dcc_vs_hist['QLIKE']}/{n}"]))
    rows.append(" & ".join(["MSE (DCC vs.\\ Hist.)", "--", "--",
                            f"{dcc_vs_hist['MSE']}/{n}"]))

    write_table("tab_bestzaehlung",
                "Häufigkeit des besten Kovarianzschätzers je Kennzahl",
                "tab:bestzaehlung",
                "lrrr",
                "Kennzahl & " + " & ".join(COV_LABEL[c] for c in COV_ORDER),
                rows,
                note=(f"Alle {n} Kombinationen aus Trainingsfenster und Prognosehorizont,"
                      f" bewertet ab {datum(COMMON_START)}. In den oberen beiden Blöcken ist"
                      f" ausgewiesen, wie oft der jeweilige Schätzer die höchste Sharpe Ratio"
                      f" beziehungsweise den niedrigsten Verlust liefert; der untere Block"
                      f" zählt zusätzlich direkt, in wie vielen der {n} Läufe DCC-GARCH die"
                      f" historische Matrix schlägt, unabhängig davon, ob GARCH dort noch"
                      f" besser abschneidet."))


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
                                    num(mvp["Ann. Sharpe"], 3),
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
DM-Tests auf den synthetischen Datensätzen: bestätigt, dass der jeweils
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
                "Synthetische Validierung: DM-Tests auf die QLIKE-Differenzen",
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
zu den Tabellen tab:strukturbruch_qlike und tab:strukturbruch_sharpe: Hier existiert kein Strukturbruch.
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
            q = read_qlike(dataset, train, 1)
            m = read_cov_mse(dataset, train, 1) * 1e8
            r = read_returns(dataset, train, 1)
            levels.append(q["Historical"].mean())
            rows.append(" & ".join([SYNTH_LABEL[dataset] if train == TRAIN_WINDOWS[0] else "",
                                    str(train), "20",
                                    num(q["Historical"].mean(), 1),
                                    num(q["GARCH"].mean(), 1),
                                    num(q["DCC"].mean(), 1),
                                    num(m["Historical"].mean(), 2),
                                    num(m["GARCH"].mean(), 2),
                                    num(m["DCC"].mean(), 2),
                                    num(sharpe(r["MVP Historical"]), 3),
                                    num(sharpe(r["MVP GARCH"]), 3),
                                    num(sharpe(r["MVP DCC"]), 3)]))
        spans[dataset] = max(levels) - min(levels)
        rows.append("MIDRULE")
    rows = rows[:-1]
    write_table("tab_synth_stabilitaet",
                "Synthetische Validierung: Stabilität über die Trainingsfenster",
                "tab:synth_stabilitaet",
                "lrrrrrrrrrrr",
                ["Datensatz & Training & $N$ & \\multicolumn{3}{c}{QLIKE} & "
                 "\\multicolumn{3}{c}{MSE ($\\times 10^{-8}$)} & "
                 "\\multicolumn{3}{c}{Sharpe MVP} \\\\",
                 "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
                 (" & & & Hist. & GARCH & DCC & Hist. & GARCH & DCC"
                  " & Hist. & GARCH & DCC \\\\")],
                rows, wide=True,
                note=("Bewertet über die volle simulierte Stichprobe, da das Universum mit"
                      " 20 Anlagen konstant ist und die Parameter des Prozesses über die"
                      " gesamte Reihe unverändert bleiben."))


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
    table_strukturbruch_kompakt(log_returns)
    table_bestzaehlung()
    table_strukturbruch_qlike(log_returns)
    table_strukturbruch_sharpe()

    print("Anhang (synthetische Validierung):")
    table_synth_basis()
    table_synth_dmw()
    table_synth_stabilitaet()

    print("Fertig.")


if __name__ == "__main__":
    main()
