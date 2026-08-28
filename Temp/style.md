# Schreibstil — Referenz

Abgeleitet aus der Bachelorarbeit (`Temp/main.tex`, Replikationsstudie zu Goutte et al. 2023).
Alle Zahlenangaben sind am Fließtext dieser Arbeit gemessen (356 Sätze, ohne Tabellen und Formeln).
Dient als Vorlage für weitere Arbeiten.

Die gemessenen Häufigkeiten beschreiben, was den Stil ausmacht — sie sind keine Zielwerte.
Wer sie mechanisch nachbaut, erzeugt Monotonie. Abschnitt~2 hält deshalb fest, wo bewusst
variiert werden soll.

---

## 1. Satzbau und Informationsdichte

**Ein Satz pro Quelltextzeile.** Kein Umbruch nach Spaltenbreite. Ein Satz endet, die
nächste Zeile beginnt. Das erleichtert Diffs und das Umstellen von Sätzen.

**Sätze sind lang und tragen mehrere Informationen.** Median 21 Wörter, Mittel 21,8,
Maximum 44. Ein Satz besteht typischerweise aus Hauptaussage plus angehängter Begründung
oder Folge:

> Da profitable Handelsstrategien, ähnlich wie Faktoren, auf der Annahme wiederholbarer
> Ineffizienzen im Markt basieren, betrifft das Problem der Replication Crisis auch
> Studien zu Machine-Learning-Modellen.

**Verkettung über Nebensätze, nicht über Satzzeichen.** Die Arbeit enthält
**null Semikolons** und **null Gedankenstriche**. Was andere mit `;` oder `—` verbinden
würden, wird zu einem Nebensatz:

| Konnektor | Häufigkeit | Funktion |
|---|---|---|
| `wobei` | 44 | Einschränkung, Nebenbefund, Präzisierung |
| `da` | 32 | Begründung (vorangestellt oder nachgestellt) |
| `wodurch` | 16 | Folge |
| `weshalb` | 14 | Schlussfolgerung |

**Satzanfänge zur Steuerung des Gedankengangs:** `So`, `Dabei`, `Zudem`, `Jedoch`,
`Daher`, `Hierbei`, `Hierfür`, `Des Weiteren`, `Stattdessen`, `Beispielsweise`.
`So` leitet die Konkretisierung einer eben aufgestellten Behauptung ein.

---

## 2. Variation — damit die Dichte nicht ermüdet

Die Merkmale aus Abschnitt~1 sind das Gerüst, nicht das Rezept. Innerhalb dieses Gerüsts
gilt:

**Satzlänge variieren.** Der Median liegt bei 21 Wörtern, die Spannweite reicht von 10
bis 44. Ein kurzer Satz nach zwei langen setzt den Punkt, der sonst untergeht. Drei
gleich lange Sätze hintereinander lesen sich als Fließband.

**Konnektoren rotieren.** `wobei` ist das häufigste Mittel, aber zwei `wobei` in
aufeinanderfolgenden Sätzen fallen auf. Als Alternativen stehen `da`, `wodurch`,
`weshalb`, `sodass`, `während`, `zumal`, `sofern` zur Verfügung, außerdem der schlichte
Punkt, wenn der Gedanke ohnehin abgeschlossen ist.

**Die Stellung des Begründungssatzes wechseln.** Voranstellen betont die Ursache
(„Da die Reihen unterschiedlich lang sind, variiert …"), Nachstellen betont das Ergebnis
(„Die Zahl der Anlagen variiert, da die Reihen unterschiedlich lang sind."). Beides
abwechselnd verwenden.

**Höchstens zwei Nebensätze pro Satz.** Dicht heißt informationsreich, nicht verschachtelt.
Wenn ein Satz drei Ebenen tief geht, gehört er geteilt.

**Satzanfänge nicht wiederholen.** Zwei Sätze hintereinander mit `Dabei` oder `Zudem`
zu beginnen, wirkt wie eine Aufzählung ohne Aufzählungszeichen.

**Verben statt Nominalisierungen**, wo es ohne Verlust geht: „weil die Korrelation
steigt" statt „aufgrund des Anstiegs der Korrelation".

---

## 3. Absatz- und Abschnittsaufbau

**Kurze Absätze von zwei bis fünf Sätzen.** Ein Absatz trägt einen Gedanken.

**Absatzabstand kommt aus dem Layout, nicht aus dem Text.** Mit `parskip` (oder
entsprechend gesetztem `\parskip`) genügt die Leerzeile im Quelltext. Manuelle
`\vspace{\baselineskip}` zwischen Absätzen sind ein Workaround und entfallen.

**Jeder Abschnitt beginnt mit einem Satz, der sagt, was er leistet:**

> In diesem Abschnitt wird das für die Replikation verwendete Vorgehen erläutert.

**Die Einleitung endet mit einem Fahrplan**, der jeden Abschnitt in einem Satz ankündigt
und dabei per `\hyperref` verlinkt.

---

## 4. Wortwahl und Register

**Unpersönlich, aber nicht steif.** Kein `wir`. Passiv oder `man` sind die Standardformen.

**Ich-Form nur dort, wo die Person unvermeidlich ist** — bei eigener Korrespondenz oder
eigenem Vorgehen, das keine andere Formulierung zulässt:

> Auf Nachfrage wurde mir von den Autoren mitgeteilt, dass …

**Fachbegriffe in der Originalsprache**, nicht eingedeutscht: Precision, Recall, Feature
Engineering, Replication Crisis, Data-Snooping, Buy-and-Hold, out-of-sample. Deutsche
Syntax mit englischen Termini.

**Abkürzungen bei Erstnennung ausschreiben**, Kurzform in Klammern: „Long Short-Term
Memory (LSTM)".

**Keine Auszeichnung im Fließtext.** Kein `\emph`, kein `\textit`, kein `\textbf`.
Betonung entsteht über Satzstellung, nicht über Formatierung.

**Keine Aufzählungslisten.** Auch Mehrfachnennungen laufen im Fließtext, bei zwei Punkten
mit `zum einen … zum anderen`.

---

## 5. Was weggelassen wird

**Keine Wiederholung von Tabellenwerten im Text.** Der Text beschreibt das Muster, die
Tabelle liefert die Zahlen. Eine Zahl steht nur dann im Text, wenn sie das Argument trägt
und sonst nirgends steht.

**Keine Methodenlehrbuch-Erklärungen.** Bekannte Verfahren werden benannt und belegt,
nicht hergeleitet. Erklärt wird nur, was von der Standardanwendung abweicht oder für die
eigenen Ergebnisse nötig ist.

**Keine Meta-Kommentare** wie „Es ist wichtig zu betonen". Die Aussage steht direkt.

**Entscheidungen werden begründet, aber knapp** — meist als Nebensatz:

> Anfallende Transaktionskosten werden wie auch in der Basisstudie nicht berücksichtigt,
> da diese stark vom gehandelten Volumen sowie indirekten Faktoren abhängen, zu denen es
> keine verlässlichen Daten gibt.

---

## 6. Umgang mit Ergebnissen und Zahlen

**Befund zuerst, Erklärung danach.**

**Abweichungen und Probleme werden offen benannt**, nicht kaschiert — Fehler der
Basisstudie ebenso wie Grenzen des eigenen Vorgehens.

**Unsicherheit wird konsequent modalisiert:** `könnte`, `vermutlich`, `wahrscheinlich`,
`liegt die Vermutung nahe`, `tendenziell`. Vermutungen werden als solche markiert.
Auch hier gilt Variation: nicht dreimal hintereinander `vermutlich`.

**Prozent mit `\%`, Dezimaltrennzeichen Komma**, Tausender mit Punkt („10.000-mal").

---

## 7. Zitation

Stil: `biblatex`, Autor-Jahr.

| Befehl | Verwendung |
|---|---|
| `\textcite{}` | Autor ist Satzsubjekt: „\textcite{Harvey2016} argumentierten, dass …" |
| `\citeauthor{}` | Wiederaufnahme im Folgesatz ohne erneute Jahreszahl |
| `\parencite{}` | reiner Beleg am Satzende |
| `\citeyear{}` | wenn nur das Jahr in den Satz gehört |
| `\citetitle{}` | wenn der Titel der Studie genannt wird |

Erstnennung `\textcite`, Wiederaufnahme `\citeauthor` — das hält den Bezug, ohne die
Jahreszahl zu wiederholen.

**Quellen tragen Inhalt, nicht Dekoration.** Zu jeder zitierten Studie steht, was sie
gemacht und was sie gefunden hat.

---

## 8. Querverweise

Immer `\hyperref` mit sprechendem Text, nie ein nacktes `\ref`:

```latex
\hyperref[sec:Methodik]{\ref*{sec:Methodik}.~Abschnitt}
\hyperref[tab:results-agg]{Tab.~\ref*{tab:results-agg}}
\hyperref[eq:sharpe]{Gleichung~\ref*{eq:sharpe}}
\hyperref[subsec:Evaluation]{Abschnitt zur Evaluation}
```

Innerhalb von `\hyperref` steht `\ref*` (Stern), damit kein verschachtelter Link entsteht.
Bei benachbarten Abschnitten genügt „folgende Abschnitt" statt einer Nummer — auch hier
lohnt der Wechsel zwischen Nummer und Bezeichnung.

---

## 9. Tabellen und Abbildungen

**Tabellen werden aus CSV gelesen**, nicht von Hand gesetzt — `\csvreader` aus
`csvsimple`, Trennzeichen Semikolon, `respect all`:

```latex
\begin{table}[h]
    \centering
    \begin{adjustbox}{width=\textwidth}
        \csvreader[tabular = {lcccccccccc},
            table head = {\toprule … \midrule},
            table foot = \bottomrule,
            late after line = \\, respect all, separator=semicolon]
        {Abbildungen/Ergebnisse/results-base.csv}{}
        {\csvcoli & \csvcolii & …}
    \end{adjustbox}
    \caption{…}{\footnotesize …}
    \label{tab:…}
\end{table}
```

- Breite Tabellen in `adjustbox` mit `width=\textwidth`
- `booktabs`-Linien, keine Vertikallinien
- Mehrstufige Köpfe über `\multicolumn` plus `\cmidrule(r){}`/`\cmidrule(l){}`
- **Beschriftung unter der Tabelle**
- Ergänzende Erläuterung als zweites `\caption`-Argument in `\footnotesize`, ohne Präfix
  wie „Anmerkung", und als knapper Satz, der die Spalten benennt

---

## 10. LaTeX-Konventionen

- Ein Satz pro Zeile
- Absatzabstand über `parskip`, keine manuellen `\vspace`
- Akronyme über `\DeclareAcronym` im Präambel-Block
- Formeln in `align`/`equation` mit `\label`, im Text per `\hyperref` referenziert
- Formelvariablen mit sprechenden deutschen Namen, wo es die Lesbarkeit erhöht:
  `Gesamtrendite`, `Vorhersage_t`
- Kommentare im Quelltext auf Deutsch, sparsam

---

## 11. Checkliste

**Substanz**

- [ ] Keine Zahl im Text, die schon in einer Tabelle steht — außer sie trägt das Argument
- [ ] Jede Vermutung modalisiert
- [ ] Jede Entscheidung knapp begründet
- [ ] Abschnitt beginnt mit einem Satz, der sagt, was er leistet

**Form**

- [ ] Ein Satz pro Zeile
- [ ] Keine Semikolons, keine Gedankenstriche
- [ ] Kein `\textbf`/`\emph` im Fließtext, keine Aufzählungslisten
- [ ] Kein `wir`; Passiv oder `man`
- [ ] Fachbegriffe englisch, Abkürzungen bei Erstnennung ausgeschrieben
- [ ] Erstnennung `\textcite`, Wiederaufnahme `\citeauthor`
- [ ] Querverweise als `\hyperref` mit `\ref*`

**Klang** — laut lesen, dann prüfen

- [ ] Keine drei gleich langen Sätze hintereinander
- [ ] Kein `wobei` in zwei aufeinanderfolgenden Sätzen
- [ ] Keine zwei gleichen Satzanfänge hintereinander
- [ ] Kein Satz mit mehr als zwei Nebensätzen
- [ ] Begründungen mal vorangestellt, mal nachgestellt
