# Ergebnisteil – Vortragsskript (ca. 12 Minuten)

> Empirischer Fokus: **ausschließlich TRBC**. Synthetische Daten (MonteCarlo,
> GARCH_sim, DCC_sim) dienen als Modell-Validierung.
> Kovarianzschätzer: Historisch · GARCH(1,1) konst. Korrelation · DCC-GARCH.
> Portfoliomethoden: MVP · ERC · HRP · naives 1/N.
> Zahlen aus `Ergebnisse/Zusammenfassung/aggregate_results.xlsx`.
> **Sharpe Ratio = `Ann. Sharpe (rf=0)`** – annualisierte Rendite / Volatilität,
> ohne Abzug des risikofreien Zinses.

---

## 0 | Fahrplan  (ca. 0:40)

„Ich baue den Ergebnisteil in vier Schritten auf. **Erstens** nehme ich *eine*
konkrete, typische Konstellation unter die Lupe – vier Jahre Training, ein Tag
Prognose auf dem TRBC-Datensatz – und schaue, ob die Ergebnisse so ausfallen,
wie man es erwarten würde. **Zweitens** löse ich das – wie sich zeigen wird –
überraschende Ergebnis auf, indem ich hinter die Sharpe Ratio schaue.
**Drittens** prüfe ich an synthetischen Daten, ob die Modelle überhaupt
funktionieren. Und **viertens** verallgemeinere ich über verschiedene Trainings-
und Prognosezeiträume.“

---

## 1 | Der Einstiegsfall: TRBC, 4 Jahre Training, 1 Tag Prognose  (ca. 3:00)

**Slide: Ergebnistabelle TRBC 1008_1 (Sharpe, Turnover je Modell × Kovarianz)**

„Fangen wir konkret an. Ich wähle bewusst eine realistische, gängige
Konstellation: **TRBC, vier Jahre Trainingsfenster – also 1008 Handelstage –,
und ein Prognose- und Rebalancing-Horizont von einem Tag.**

Die naheliegende Erwartung wäre: Das aufwendigste Modell, DCC-GARCH mit
dynamischer Korrelation, sollte die beste risikoadjustierte Rendite liefern.
Schauen wir auf das Minimum-Variance-Portfolio, weil dort der Kovarianzschätzer
am stärksten durchschlägt:

- **Historische Kovarianz: Sharpe 0,695**
- **GARCH: Sharpe 0,681**
- **DCC-GARCH: Sharpe 0,671**

Das ist – gelinde gesagt – **unerwartet**. Die Rangfolge ist *genau umgekehrt*
zur Modellkomplexität. Der einfachste Schätzer, die simple historische Matrix,
gewinnt; das theoretisch überlegene DCC-Modell liegt hinten. Und das ist kein
Ausreißer bei MVP – bei ERC und HRP sieht es genauso aus: Historisch vorne, DCC
dahinter. Selbst das **naive 1/N-Portfolio** ist mit Sharpe 0,58 nicht weit weg
von den optimierten Verfahren.

Der Grund wird sofort sichtbar, wenn man die **Handelsstrategie** danebenlegt,
also den Turnover – wie viel jede Strategie pro Periode umschichtet:

- **MVP historisch: Turnover 0,008** – praktisch statisch.
- **MVP GARCH: 0,28. MVP DCC: 0,26.**

Das ist ein Faktor **dreißig**. Die dynamischen Modelle reagieren auf jede neue
Volatilitätsschätzung und schichten permanent um; die historische Matrix ist
träge und lässt das Portfolio in Ruhe. Und wohlgemerkt: In diesen Sharpe-Zahlen
sind **noch gar keine Transaktionskosten** enthalten. Rechnet man die ein, zieht
die historische Variante *weiter* davon. Ökonomisch ist das Ergebnis also
eindeutig: Der teuerste Schätzer ist hier der schlechteste.“

---

## 2 | Auflösung: statistische vs. ökonomische Relevanz  (ca. 3:00)

**Slide: TRBC 1008_1 – QLIKE, Real/Fcst-Std und realisierte Vola nebeneinander**

„Jetzt die entscheidende Frage: Ist das DCC-Modell also einfach *schlecht*? Nein
– und hier trennt sich statistische von ökonomischer Relevanz. Schauen wir auf
das, wofür das Modell eigentlich gebaut ist: die **Prognosegüte** der Kovarianz,
gemessen am QLIKE-Loss – kleiner ist besser.

- **DCC: −156,7 · GARCH: −154,6 · Historisch: −152,8.**

Statistisch ist DCC hier klar **der beste Prognostiker** – und das ist kein
Zufall dieses einen Falls: Zählt man über den gesamten Datensatz aus, gewinnt
DCC den QLIKE-Vergleich bei TRBC in **allen 40** getesteten Konstellationen. Als
Prognosemodell tut DCC also genau das, was es soll.

Und auch im Portfolio erfüllt es seine eigentliche Aufgabe. MVP soll die
**Varianz minimieren** – nicht die Sharpe Ratio maximieren. Schaut man auf die
tatsächlich realisierte Volatilität:

- **MVP DCC: 13,81 % · MVP Historisch: 13,82 %.**

DCC liefert also die **niedrigste** realisierte Vola – hauchdünn, aber es
erreicht sein Ziel. Das Modell ‚funktioniert‘ in seinem eigenen Zielmaß. Dass es
trotzdem bei der Sharpe Ratio verliert, liegt allein daran, dass die etwas
höhere Rendite und der dramatisch niedrigere Turnover der historischen Variante
den winzigen Vola-Vorteil überkompensieren.

Ein zweiter Beleg für die Grenzen auf realen Daten: der **Vola-Bias**,
realisierte geteilt durch prognostizierte Vola. Bei MVP-DCC liegt der bei
**1,28** – das Modell **unterschätzt** die tatsächliche Schwankung um fast 30 %.
Die historische Matrix ist mit 1,05 deutlich näher an der Wahrheit.

**Die Kernbotschaft dieses Abschnitts:** Statistische Überlegenheit – bester
QLIKE, niedrigste Varianz – und ökonomischer Nutzen – höchste Sharpe, geringste
Kosten – fallen hier **auseinander**. DCC gewinnt die statistische Disziplin und
verliert die ökonomische. Genau diese Lücke ist der interessante Befund: Ein
besseres Prognosemodell ist auf echten Marktdaten nicht automatisch die bessere
Anlageentscheidung.“

---

## 3 | Funktionieren die Modelle überhaupt? Validierung an synthetischen Daten  (ca. 3:00)

**Slide: MVP-Sharpe & QLIKE für MonteCarlo / GARCH_sim / DCC_sim (train=1008, pred=1)**

„Jetzt könnte man einwenden: Vielleicht ist das DCC-Modell schlicht falsch
implementiert oder generell wertlos. Um das auszuschließen, brauche ich Daten,
bei denen ich den **wahren datengenerierenden Prozess kenne**. Genau dafür sind
die drei synthetischen Datensätze – ein kontrolliertes Experiment mit drei
Stufen:

**Erstens MonteCarlo** – eine geometrische Brownsche Bewegung mit *konstanter*
Volatilität, also *ohne* jede dynamische Struktur. Erwartung: Hier darf DCC
keinen Vorteil haben. Genau so ist es – der QLIKE ist für alle drei Schätzer
praktisch identisch, und beim MVP-Sharpe liegt die historische Matrix mit
**1,010** sogar minimal vorn vor DCC mit 1,002. Das ist der Kontrollfall: Wo es
nichts zu modellieren gibt, bringt das komplexe Modell nichts – korrekt.

**Zweitens GARCH_sim** – jetzt mit echter GARCH-Volatilitätsdynamik. Erwartung:
Die GARCH-basierten Schätzer müssen die historische Matrix schlagen. Und das tun
sie deutlich: **MVP-Sharpe DCC 0,98 und GARCH 0,98 gegen Historisch 0,87.** Die
realisierte Vola sinkt von 11,4 % (historisch) auf 9,9 %. Sobald die Struktur da
ist, wird sie erkannt und in besseres Risiko übersetzt.

**Drittens DCC_sim** – zusätzlich mit *dynamischer Korrelation*, dem
Alleinstellungsmerkmal von DCC. Erwartung: Jetzt muss DCC auch GARCH schlagen.
Auch das bestätigt sich, und zwar am klarsten: **MVP-Sharpe DCC 0,93 > GARCH
0,89 > Historisch 0,67** – der größte Abstand von allen, und DCC hat mit −162
den mit Abstand besten QLIKE.

**Fazit dieses Blocks:** Die Modelle sind **nicht** kaputt. Sie liefern
*exakt dann* einen Mehrwert, wenn der Datenprozess ihre Annahmen erfüllt – und
der Mehrwert wächst genau in dem Maße, wie die Struktur komplexer wird. Der
Umkehrschluss für Abschnitt 1 und 2 ist damit klar: Die realen TRBC-Daten
verhalten sich in der Praxis eher wie der MonteCarlo-Fall – die ausbeutbare
Struktur ist nach Kosten zu gering, um den Aufwand zu rechtfertigen.“

---

## 4 | Verallgemeinerung: Trainings- und Prognosezeiträume  (ca. 2:00)

**Slide: TRBC – MVP-Sharpe vs. Trainingsfenster (pred=1)**

„Bleibt die Frage, ob der Einstiegsfall repräsentativ ist. Zwei
Verallgemeinerungen, weiter auf TRBC und MVP.

**Trainingsfenster, bei einem Tag Prognose.** Ich habe von einem Jahr (252 Tage)
bis zehn Jahre (2520) durchvariiert. Zwei Dinge: Erstens bleibt die **Rangfolge
stabil** – die historische Matrix ist in fast allen Fensterlängen gleichauf oder
vorn; der DCC-Vorsprung aus der Statistik materialisiert sich in *keinem* Fenster
als Sharpe-Vorteil. Zweitens ist der Effekt der Fensterlänge **nicht monoton**:
die Sharpe schwankt zwischen 0,47 und 0,70, mit Bestwerten bei 1008 und bei 2520
Tagen. Es gibt kein universell optimales Fenster – ein Teil des Ergebnisses ist
schätzunsicherheits- und periodengetrieben, weshalb man Einzelwerte nicht
überinterpretieren sollte.

**Slide: TRBC – MVP-Sharpe & Turnover vs. Prognosehorizont**

**Prognosehorizont, gemittelt über die Fenster.** Das ist der aufschlussreichste
Teil. Verlängere ich den Horizont von 1 auf 21 Tage:

- **DCC-Sharpe fällt von 0,58 auf 0,50** – die dynamischen Modelle werden aktiv
  *schlechter*.
- **Historisch bleibt praktisch konstant bei ~0,56** – und überholt DCC bei
  längeren Horizonten sogar klar.

Der Grund steht im Turnover: Er explodiert für DCC von 0,26 auf **0,60**, während
er für die historische Matrix mit 0,06 winzig bleibt. Bei längeren Horizonten
jagen die GARCH-Modelle veraltete Prognosen und schichten umso heftiger um – die
träge historische Matrix wird dadurch relativ immer besser. Die statistische
Überlegenheit von DCC übersetzt sich also nicht nur *nicht* in ökonomischen
Nutzen – bei längerem Horizont **kehrt sie sich ins Gegenteil**.“

---

## 5 | Zusammenfassung  (ca. 0:50)

**Slide: 3 Take-aways**

„Drei Botschaften zum Mitnehmen:

**Erstens:** Im Einstiegsfall TRBC schlägt der einfachste Schätzer, die
historische Kovarianz, das aufwendige DCC-GARCH bei der Sharpe Ratio – bei einem
Dreißigstel des Handelsvolumens. Das ist unerwartet, aber robust über
Trainingsfenster und Horizonte.

**Zweitens:** Das ist kein Modellfehler. Statistisch ist DCC überlegen – bester
QLIKE, niedrigste Varianz – und die synthetischen Daten zeigen sauber, dass die
Modelle *genau dann* Mehrwert liefern, wenn der Prozess ihre Annahmen erfüllt.
Statistische und ökonomische Relevanz fallen auf realen Daten auseinander.

**Drittens:** Der Preis der Dynamik ist der Turnover – und er wird bei längeren
Prognosehorizonten so hoch, dass sich das Vorzeichen des Nutzens umdreht. Auf
echten Daten ist das komplexere Modell hier nicht die bessere Anlageentscheidung.

Vielen Dank – gerne Ihre Fragen.“

---

### Backup-Zahlen (TRBC, sofern nicht anders genannt)
- **1008_1 MVP Sharpe:** Hist 0,695 · GARCH 0,681 · DCC 0,671.  (Naive 0,581)
- **1008_1 MVP Turnover:** Hist 0,008 · GARCH 0,28 · DCC 0,26.
- **1008_1 MVP Vola:** DCC 13,81 % · Hist 13,82 %; **QLIKE** DCC −156,7 < Hist −152,8.
- **1008_1 MVP Vola-Bias (Real/Fcst):** DCC 1,28 · Hist 1,05.
- **QLIKE-Sieger DCC bei TRBC:** 40/40 Konstellationen.
- **Synth MVP-Sharpe (train=1008, pred=1):**
  MonteCarlo Hist 1,010 ≥ DCC 1,002 · GARCH_sim DCC 0,98 > Hist 0,87 · DCC_sim DCC 0,93 > GARCH 0,89 > Hist 0,67.
- **TRBC MVP Sharpe vs. Horizont (1→21):** DCC 0,58→0,50 · Hist 0,57→0,56.
- **TRBC MVP Turnover vs. Horizont (1→21):** DCC 0,26→0,60 · Hist 0,009→0,06.
- **TRBC MVP Sharpe vs. Fenster:** min 0,47 (2016) … max 0,70 (2520); Bestwerte 1008 (0,695) & 2520 (0,700).
- **5-J-Check (1260_1 MVP):** Hist 0,624 · GARCH 0,601 · DCC 0,599 (Naive 0,507) – durchweg schwächer als 4 J.
