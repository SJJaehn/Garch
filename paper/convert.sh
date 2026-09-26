#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALL_DIR="$(pwd)"
if [[ $# -eq 0 ]]; then
  OUTPUT="$SCRIPT_DIR/paper.docx"
elif [[ "$1" = /* ]]; then
  OUTPUT="$1"
else
  OUTPUT="$CALL_DIR/$1"
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc is required but was not found in PATH." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Arbeitskopie. Die breiten Tabellen sind im Manuskript in \resizebox gewickelt,
# damit sie im PDF nicht in den Rand laufen. Pandoc kann das nicht parsen und
# verwirft die betroffenen Tabellen stillschweigend, deshalb wird die Huelle in
# der Kopie entfernt. Das Manuskript selbst bleibt unveraendert.
cp paper.tex literatur.bib "$WORK/"
cp paper.aux "$WORK/" 2>/dev/null || true
cp -R Abbildungen "$WORK/"
python3 - "$WORK" <<'PYEOF'
import glob, io, os, re, sys

# Die Fussnote unter Tabellen und Abbildungen steht im Manuskript in einer
# minipage. Pandoc kennt dafuer kein eigenes Format und setzt sie als normalen
# Fliesstext. Als quote-Umgebung landet sie dagegen im Word-Stil "Block Text",
# der unten im Referenzdokument klein und zentriert definiert wird.
NOTE = re.compile(
    r"[ \t]*\\begin\{minipage\}\{\\linewidth\}[^\n]*\n(.*?)\n[ \t]*\\end\{minipage\}\n",
    re.S)

for path in glob.glob(os.path.join(sys.argv[1], "Abbildungen", "*.tex")):
    text = io.open(path, encoding="utf-8").read()
    out = text

    # Breite Tabellen: die \resizebox-Huelle entfernen, sonst verwirft Pandoc
    # die Tabelle stillschweigend.
    if "\\resizebox" in out:
        out = re.sub(r"[ \t]*\\resizebox\{[^\n]*\}\{!\}\{%\n", "", out)
        out = re.sub(r"\n[ \t]*\}\n(\s*\\caption)", r"\n\1", out)

    # Abbildungen: die Fussnote aus der figure-Umgebung herausziehen. Bleibt sie
    # drin, deutet Pandoc Bild und Fussnote als zwei Teilabbildungen und setzt
    # sie nebeneinander, wodurch das Bild nur die halbe Seitenbreite bekommt.
    note = "\\begin{quote}\n%s\n\\end{quote}\n"
    if "\\end{figure}" in out:
        # Bei Abbildungen muss die Fussnote zusaetzlich aus der figure-Umgebung
        # heraus. Bleibt sie drin, deutet Pandoc Bild und Fussnote als zwei
        # Teilabbildungen und setzt sie nebeneinander, wodurch das Bild nur die
        # halbe Seitenbreite bekommt.
        out = NOTE.sub("", out)
        body = NOTE.search(text)
        if body:
            out = out.replace("\\end{figure}\n",
                              "\\end{figure}\n\n" + note % body.group(1).strip())
    else:
        out = NOTE.sub(lambda m: note % m.group(1).strip(), out)

    # \cmidrule kennt Pandoc nicht. Die Zeile wird nicht ignoriert, sondern ihr
    # Inhalt landet als Text in der ersten Kopfzelle ("3-4(lr)5-6(lr)7-8").
    out = re.sub(r"(?m)^[ \t]*(?:\\cmidrule(?:\([lr]+\))?\{\d+-\d+\})+[ \t]*\n", "", out)

    # Signifikanzsterne stehen als $^{***}$ im Mathemodus, also als Hochstellung
    # ohne Basis. Pandoc erzeugt daraus ein leeres Formelobjekt, das in Word
    # unsichtbar ist -- die Tabellen verlieren dadurch saemtliche Sterne.
    out = re.sub(r"\$\^\{(\*+)\}\$", r"\\textsuperscript{\1}", out)

    if out != text:
        io.open(path, "w", encoding="utf-8").write(out)
PYEOF

# Pandoc nummeriert Gleichungen nicht, deshalb laufen \ref{eq:...}-Verweise ins
# Leere und erscheinen in Word als "Gleichung [eq:qlike]". Die Nummern stehen in
# paper.aux vom letzten LaTeX-Lauf und werden hier fest eingesetzt.
python3 - "$WORK" <<'AUXEOF'
import io, os, re, sys

work = sys.argv[1]
path = os.path.join(work, "paper.tex")
body = io.open(path, encoding="utf-8").read()

# Gleichungsnummern aus dem letzten LaTeX-Lauf fest einsetzen.
aux = os.path.join(work, "paper.aux")
if os.path.exists(aux):
    text = io.open(aux, encoding="utf-8", errors="ignore").read()
    numbers = dict(re.findall(r"\\newlabel\{(eq:[a-z0-9-]+)\}\{\{(\d+)\}", text))
    pattern = r"\\hyperref\[(eq:[a-z0-9-]+)\]\{Gleichung~\\ref\*\{eq:[a-z0-9-]+\}\}"
    body = re.sub(pattern, lambda m: "Gleichung~" + numbers.get(m.group(1), m.group(1)), body)

# Das Transponiert-Zeichen als Apostroph zieht Pandoc in den Index, aus r_t'
# wird dann r_{t'}. Explizit als \prime geschrieben bleibt es eine Hochstellung.
body = re.sub(r"([A-Za-z])_([A-Za-z0-9])'", r"\1_\2^{\\prime}", body)

io.open(path, "w", encoding="utf-8").write(body)
AUXEOF

# Referenzdokument fuer die Word-Formatierung. Der Standardstil von Pandoc setzt
# Tabellen ohne Linien; hier bekommen sie die Booktabs-Anmutung des PDF, also
# eine kraeftige Linie oben und unten, eine duenne unter dem Kopf, keine
# senkrechten Linien, dazu 10 pt Schrift und enge Zeilen.
REFDOC="$WORK/reference.docx"
pandoc --print-default-data-file reference.docx > "$WORK/reference_default.docx"
python3 - "$WORK/reference_default.docx" "$REFDOC" <<'PYEOF'
import re, sys, zipfile

NOTE_STYLE = '''<w:style w:type="paragraph" w:styleId="BlockText">
    <w:name w:val="Block Text"/>
    <w:basedOn w:val="BodyText"/>
    <w:next w:val="BodyText"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="60" w:after="180" w:line="240" w:lineRule="auto"/>
      <w:ind w:firstLine="0" w:left="0" w:right="0"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>
  </w:style>'''

TABLE_STYLE = '''<w:style w:type="table" w:default="1" w:styleId="Table">
    <w:name w:val="Table"/>
    <w:basedOn w:val="TableNormal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
    <w:tblPr>
      <w:tblInd w:w="0" w:type="dxa"/>
      <w:tblBorders>
        <w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>
        <w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>
    <w:tblStylePr w:type="firstRow">
      <w:tcPr><w:tcBorders>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
      </w:tcBorders><w:vAlign w:val="bottom"/></w:tcPr>
    </w:tblStylePr>
  </w:style>'''

# Seitenformat wie im Manuskript: A4, Rand links 3,5 cm, rechts 1,5 cm, oben
# 2,5 cm, unten 2 cm (Angaben in Twips). Das Referenzdokument von Pandoc setzt
# gar kein Format, Word nimmt dann Letter. Die Seitenbreite ist zugleich die
# Obergrenze, auf die Pandoc Grafiken skaliert -- ohne diesen Block bleiben die
# Abbildungen deutlich schmaler als der Satzspiegel.
SECT_PR = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1417" w:right="850" w:bottom="1134" w:left="1984" '
           'w:header="709" w:footer="709" w:gutter="0"/>'
           '<w:footnotePr><w:numRestart w:val="eachSect"/></w:footnotePr></w:sectPr>')

src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/styles.xml":
            text = data.decode("utf-8")
            text, hits = re.subn(
                r'<w:style w:type="table" w:default="1" w:styleId="Table">.*?</w:style>',
                TABLE_STYLE, text, flags=re.S)
            if hits != 1:
                sys.exit("Table-Stil im Referenzdokument nicht gefunden")
            text, hits = re.subn(
                r'<w:style w:type="paragraph" w:styleId="BlockText">.*?</w:style>',
                NOTE_STYLE, text, flags=re.S)
            if hits != 1:
                sys.exit("BlockText-Stil im Referenzdokument nicht gefunden")
            data = text.encode("utf-8")
        elif item.filename == "word/document.xml":
            text = data.decode("utf-8")
            text, hits = re.subn(r"<w:sectPr>.*?</w:sectPr>", SECT_PR, text, flags=re.S)
            if hits != 1:
                sys.exit("sectPr im Referenzdokument nicht gefunden")
            data = text.encode("utf-8")
        zout.writestr(item, data)
PYEOF

cd "$WORK"
pandoc paper.tex \
  --from=latex \
  --to=docx \
  --citeproc \
  --bibliography=literatur.bib \
  --resource-path=.:Abbildungen \
  --reference-doc="$REFDOC" \
  -o "$OUTPUT"


# Pandoc setzt jede Tabelle auf dieselbe Breite (5,5 Zoll) und ignoriert dabei
# die Spaltenzahl. Bei den breiten Tabellen bleiben so nur knapp 0,5 Zoll pro
# Spalte, wodurch Word negative Zahlen hinter dem Minus umbricht. Hier werden
# die Tabellen auf den vollen Satzspiegel gezogen und breite zusaetzlich
# kleiner gesetzt -- das entspricht dem \resizebox des Manuskripts.
python3 - "$OUTPUT" <<'TBLEOF'
import re, shutil, sys, zipfile

TEXT_WIDTH = 9072          # Twips, A4 abzueglich der Raender aus paper.tex
src = sys.argv[1]
tmp = src + ".tmp"

zin = zipfile.ZipFile(src)
parts = {name: zin.read(name) for name in zin.namelist()}
infos = zin.infolist()
zin.close()

doc = parts["word/document.xml"].decode("utf-8")


def resize(match):
    table = match.group(0)
    widths = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)"', table)]
    if not widths:
        return table
    total = sum(widths)
    scaled = [max(400, round(w * TEXT_WIDTH / total)) for w in widths]
    it = iter(scaled)
    table = re.sub(r'<w:gridCol w:w="\d+"',
                   lambda m: '<w:gridCol w:w="%d"' % next(it), table)
    table = re.sub(r'<w:tblW w:w="\d+" w:type="\w+"/>',
                   '<w:tblW w:w="%d" w:type="dxa"/>' % TEXT_WIDTH, table)
    it2 = iter(scaled * (table.count('<w:tcW') // max(1, len(scaled)) + 1))
    table = re.sub(r'<w:tcW w:w="\d+" w:type="\w+"/>',
                   lambda m: '<w:tcW w:w="%d" w:type="dxa"/>' % next(it2), table)

    # Ab acht Spalten wird die Schrift verkleinert, damit Zahlen nicht umbrechen
    if len(widths) >= 8:
        size = "14" if len(widths) >= 11 else "16"
        sz = '<w:sz w:val="%s"/><w:szCs w:val="%s"/>' % (size, size)
        table = table.replace("<w:r><w:rPr>", "<w:r><w:rPr>" + sz)
        table = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr>" + sz + "</w:rPr>", table)
    return table


doc = re.sub(r"<w:tbl>.*?</w:tbl>", resize, doc, flags=re.S)
parts["word/document.xml"] = doc.encode("utf-8")

with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for info in infos:
        zout.writestr(info, parts[info.filename])
shutil.move(tmp, src)
TBLEOF

echo "Wrote $OUTPUT"
