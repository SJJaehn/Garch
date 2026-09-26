"""
Erzeugt die Abbildungen der Arbeit aus den Ergebnis-CSVs.

Gelesen wird ausschliesslich aus CSV-Dateien:
  Ergebnisse/<Datensatz>/<train>_<pred>/summary.csv
  Ergebnisse/<Datensatz>/<train>_<pred>/returns.csv

Geschrieben wird je Abbildung eine PDF-Datei und eine .tex-Datei nach
paper/Abbildungen/. Die .tex-Dateien enthalten eine vollstaendige
figure-Umgebung und werden im Manuskript per \\input{Abbildungen/<name>}
eingebunden.

    python paper/abbildungen.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "paper"))
import tabellen


# =============================================================================
# Settings
# =============================================================================

OUT_DIR = tabellen.OUT_DIR

# Schrift und Farben passen zum Fliesstext: Times-Klon in 9pt, gedeckte
# Farben, die auch im Graustufendruck unterscheidbar bleiben.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
})

# Standardfarben von matplotlib (C0, C1, C2)
COV_COLOR = {"Historical": "#1f77b4", "GARCH": "#ff7f0e", "DCC": "#2ca02c"}
COV_HATCH = {"Historical": "", "GARCH": "", "DCC": ""}


"""
Schreibt die figure-Umgebung zu einer bereits erzeugten PDF-Datei.
"""
def write_figure(name, caption, label, note=None, width="\\textwidth"):
    lines = ["% automatisch erzeugt von paper/abbildungen.py -- nicht von Hand editieren",
             "\\begin{figure}[htbp]",
             "  \\centering",
             f"  \\includegraphics[width={width}]{{Abbildungen/{name}.pdf}}",
             "  \\caption{" + caption + "}",
             "  \\label{" + label + "}"]
    if note:
        lines.append("  \\begin{minipage}{\\linewidth}\\vspace{0.4em}\\centering\\footnotesize")
        lines.append("  " + note)
        lines.append("  \\end{minipage}")
    lines.append("\\end{figure}")
    path = os.path.join(OUT_DIR, name + ".tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  geschrieben:", os.path.relpath(path, REPO_DIR))


"""
Sharpe Ratio je Strategie auf einem eingeschraenkten Bewertungszeitraum,
gerechnet aus returns.csv statt aus summary.csv.
"""
def sharpe_ab(dataset, train, pred, start):
    ret = tabellen.read_returns(dataset, train, pred)
    ret = ret.loc[ret.index >= pd.Timestamp(start)]
    return ret.apply(tabellen.sharpe)


# =============================================================================
# Abbildung 1: Sharpe Ratio im Basisfall
# =============================================================================

"""
Balkendiagramm der Sharpe Ratio im Basisfall, gruppiert nach Portfoliomodell
und eingefaerbt nach Kovarianzschaetzer. Die gestrichelte Linie markiert das
1/N-Portfolio.
"""
def abb_sharpe_basis():
    summary = tabellen.read_summary(tabellen.DATASET, tabellen.BASE_TRAIN,
                                    tabellen.BASE_PRED)
    naive = summary.loc["Naive", "Ann. Sharpe"]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    width = 0.26
    x = np.arange(len(tabellen.MODEL_ORDER))
    for k, cov in enumerate(tabellen.COV_ORDER):
        vals = [summary.loc[f"{m} {cov}", "Ann. Sharpe"]
                for m in tabellen.MODEL_ORDER]
        ax.bar(x + (k - 1) * width, vals, width,
               label=tabellen.COV_LABEL[cov], color=COV_COLOR[cov],
               edgecolor="white", linewidth=0.6, hatch=COV_HATCH[cov])

    ax.axhline(naive, color="black", linestyle="--", linewidth=1.0, zorder=0,
               label=f"1/N ({naive:.3f})".replace(".", ","))
    ax.set_xticks(x)
    ax.set_xticklabels(tabellen.MODEL_ORDER)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_ylim(0, max(summary["Ann. Sharpe"].max(), naive) * 1.18)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              loc="upper right", frameon=False, ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "abb_sharpe_basis.pdf"), bbox_inches="tight")
    plt.close(fig)

    write_figure("abb_sharpe_basis",
                 "Sharpe Ratio der Portfoliomodelle im Basisfall",
                 "abb:sharpe_basis",
                 note=(f"{tabellen.DATASET}, Training {tabellen.BASE_TRAIN} Handelstage, "
                       f"Prognosehorizont {tabellen.BASE_PRED} Tag; "
                       "die gestrichelte Linie markiert das 1/N-Portfolio."),
                 width="0.88\\textwidth")


# =============================================================================
# Abbildung 2: Sharpe Ratio ueber die Trainingsfenster
# =============================================================================

"""
Sharpe Ratio je Portfoliomodell ueber alle Trainingsfenster, gerechnet auf dem
gemeinsamen Bewertungszeitraum. Ein Panel je Modell, eine Linie je
Kovarianzschaetzer, dazu das 1/N-Portfolio als gestrichelte Linie.
"""
def abb_sharpe_train():
    werte = {}
    for train in tabellen.TRAIN_WINDOWS:
        werte[train] = sharpe_ab(tabellen.DATASET, train, tabellen.BASE_PRED,
                                 tabellen.COMMON_START)
    jahre = [t / 252.0 for t in tabellen.TRAIN_WINDOWS]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    for ax, model in zip(axes, tabellen.MODEL_ORDER):
        for cov in tabellen.COV_ORDER:
            y = [werte[t].get(f"{model} {cov}", np.nan) for t in tabellen.TRAIN_WINDOWS]
            ax.plot(jahre, y, marker="o", markersize=3, linewidth=1.2,
                    color=COV_COLOR[cov], label=tabellen.COV_LABEL[cov])
        y = [werte[t].get("Naive", np.nan) for t in tabellen.TRAIN_WINDOWS]
        ax.plot(jahre, y, linestyle="--", linewidth=1.0, color="black", label="1/N")
        ax.set_title(model)
        ax.set_xlabel("Training (Jahre)")
        ax.set_xticks([1, 3, 5, 7, 9])
    axes[0].set_ylabel("Sharpe Ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "abb_sharpe_train.pdf"), bbox_inches="tight")
    plt.close(fig)

    write_figure("abb_sharpe_train",
                 "Sharpe Ratio über die Trainingsfenster",
                 "abb:sharpe_train",
                 note=(f"{tabellen.DATASET}, Prognosehorizont {tabellen.BASE_PRED} Tag, "
                       f"Bewertung ab "
                       f"{pd.Timestamp(tabellen.COMMON_START).strftime('%d.%m.%Y')} "
                       "für alle Fenster."))


# =============================================================================
# Abbildung 3: Bewertungszeitraum voll gegen gemeinsam
# =============================================================================

"""
Sharpe Ratio ueber die Trainingsfenster, links auf der vollen Historie des
jeweiligen Fensters und rechts auf dem gemeinsamen Zeitraum. Der Sprung beim
laengsten Fenster im linken Panel ist ein reiner Stichprobeneffekt, weil er
auch beim 1/N-Portfolio auftritt, das gar nicht schaetzt.
"""
def abb_sharpe_zeitraum():
    voll = {}
    gemeinsam = {}
    for train in tabellen.TRAIN_WINDOWS:
        ret = tabellen.read_returns(tabellen.DATASET, train, tabellen.BASE_PRED)
        voll[train] = ret.apply(tabellen.sharpe)
        gemeinsam[train] = ret.loc[tabellen.COMMON_START:].apply(tabellen.sharpe)
    jahre = [t / 252.0 for t in tabellen.TRAIN_WINDOWS]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    titel = ["volle Historie", f"gemeinsam ab {tabellen.COMMON_START[:4]}"]
    for ax, werte, name in zip(axes, [voll, gemeinsam], titel):
        for cov in tabellen.COV_ORDER:
            y = [werte[t].get(f"MVP {cov}", np.nan) for t in tabellen.TRAIN_WINDOWS]
            ax.plot(jahre, y, marker="o", markersize=3, linewidth=1.2,
                    color=COV_COLOR[cov], label="MVP " + tabellen.COV_LABEL[cov])
        y = [werte[t].get("Naive", np.nan) for t in tabellen.TRAIN_WINDOWS]
        ax.plot(jahre, y, linestyle="--", linewidth=1.2, color="black", label="1/N")
        ax.set_title(name)
        ax.set_xlabel("Training (Jahre)")
        ax.set_xticks([1, 3, 5, 7, 9])
    axes[0].set_ylabel("Sharpe Ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "abb_sharpe_zeitraum.pdf"), bbox_inches="tight")
    plt.close(fig)

    write_figure("abb_sharpe_zeitraum",
                 "Sharpe Ratio über die Trainingsfenster, voller und gemeinsamer Bewertungszeitraum",
                 "abb:sharpe_zeitraum",
                 note=(f"{tabellen.DATASET}, Prognosehorizont {tabellen.BASE_PRED} Tag, "
                       "links die volle Historie des jeweiligen "
                       "Fensters, rechts der für alle Fenster identische Zeitraum ab "
                       f"{pd.Timestamp(tabellen.COMMON_START).strftime('%d.%m.%Y')}."))


def main():
    print("Abbildungen werden erzeugt ...")
    abb_sharpe_basis()
    abb_sharpe_train()
    abb_sharpe_zeitraum()
    print("fertig.")


if __name__ == "__main__":
    main()
