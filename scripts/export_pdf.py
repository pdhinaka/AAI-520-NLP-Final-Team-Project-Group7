"""Export the executed notebook to PDF through LaTeX.

Run from the repo root after executing the notebook:

    python scripts/export_pdf.py

nbconvert's LaTeX route prints DataFrames as their plain-text repr,
which wraps badly on a page. This script works on an in-memory copy of
the notebook: every HTML table output gets a LaTeX longtable with
wrapped columns, and the preamble gets a smaller monospace font and
slimmer margins so 100-column text output fits without re-wrapping,
and symbols the default font lacks (>= and <=) are mapped to math.
Section numbers come from the headings, so LaTeX numbering is off. The
notebook file itself is not changed. Writes reports/final_notebook.pdf.
"""

import argparse
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import nbformat
from nbconvert import LatexExporter

from finagent import config

NOTEBOOK = config.ROOT / "notebooks" / "final_notebook.ipynb"
OUT = config.ROOT / "reports" / "final_notebook.pdf"
TITLE = "Multi-Agent Financial Analysis System"
AUTHORS = ["Eric Hernandez", "Nolan Robbins", "Pranav Dhinakar"]

PREAMBLE = r"""
\usepackage{newunicodechar}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\setcounter{secnumdepth}{0}
\fvset{fontsize=\footnotesize}
\renewcommand{\arraystretch}{1.15}
"""

# nbconvert hard-wraps text output at 80 characters, sized for its
# default font and margins; footnotesize fits about 105.
TEMPLATE = """((*- extends 'index.tex.j2' -*))
((* set charlim = 104 *))
"""

# Characters of footnotesize text that fit across the text width.
LINE_CHARS = 120
LONGEST_WORD = 22


class _TableParser(HTMLParser):
    """Collect header and body rows of the first <table> as text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.head, self.body = [], []
        self._row = self._cell = None
        self._in_head = self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self._skip = True
        elif tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._skip = False
        elif tag == "thead":
            self._in_head = False
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            (self.head if self._in_head else self.body).append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None and not self._skip:
            self._cell.append(data)


def _escape(text):
    out = []
    for i, ch in enumerate(text):
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
        # Let URLs and key=value lists break inside a narrow column.
        if ch in "/=," and i + 1 < len(text) and text[i + 1] != " ":
            out.append(r"\allowbreak{}")
    return "".join(out)


def _widths(rows):
    """Column widths as fractions of the line, or None if 'l' fits."""
    ncols = max(len(r) for r in rows)
    cols = [[r[i] for r in rows if i < len(r)] for i in range(ncols)]
    nat = [min(max(len(c) for c in col), 80) for col in cols]
    if sum(nat) + 2 * ncols <= LINE_CHARS:
        return None
    # Every column is at least as wide as its longest word (capped), and
    # the rest of the line goes to the columns with long text.
    floor = [min(max(len(w) for c in col for w in c.split() or [""]),
                 LONGEST_WORD, n) + 1 for col, n in zip(cols, nat)]
    avail = LINE_CHARS - 2 * ncols
    rest = max(avail - sum(floor), 0)
    extra = [n - f for n, f in zip(nat, floor)]
    total = sum(extra) or 1
    chars = [f + rest * e / total for f, e in zip(floor, extra)]
    usable = 1.0 - 0.017 * ncols
    return [usable * c / sum(chars) for c in chars]


def html_table_to_latex(html):
    """A booktabs longtable for a pandas HTML table, or None."""
    parser = _TableParser()
    parser.feed(html)
    if not parser.body:
        return None
    ncols = max(len(r) for r in parser.head + parser.body)

    def pad(row):
        return row + [""] * (ncols - len(row))

    # pandas writes the index name on a second header row; merge them.
    header = [""] * ncols
    for row in parser.head:
        for i, cell in enumerate(pad(row)):
            if cell:
                header[i] = cell
    body = [pad(r) for r in parser.body]
    widths = _widths([header] + body)
    if widths is None:
        spec = "l" * ncols
    else:
        spec = "".join(
            r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % w
            for w in widths)

    def line(row):
        return " & ".join(_escape(c) for c in row) + r" \\"

    return "\n".join([
        r"\begingroup\footnotesize",
        r"\begin{longtable}{%s}" % spec,
        r"\toprule", line(header), r"\midrule", r"\endhead",
        *[line(r) for r in body],
        r"\bottomrule",
        r"\end{longtable}",
        r"\endgroup",
    ])


def prepare(nb):
    """Add LaTeX tables and title metadata to an in-memory notebook."""
    tables = 0
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            html = out.get("data", {}).get("text/html")
            if html and "<table" in html:
                tex = html_table_to_latex(html)
                if tex:
                    out["data"]["text/latex"] = tex
                    tables += 1
    nb.metadata["title"] = TITLE
    nb.metadata["authors"] = [{"name": a} for a in AUTHORS]
    return tables


def to_latex(nb, tmp):
    template = tmp / "export.tex.j2"
    template.write_text(TEMPLATE)
    exporter = LatexExporter(template_file=str(template))
    body, resources = exporter.from_notebook_node(nb)
    body = body.replace(
        "tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in",
        "tmargin=0.8in,bmargin=0.8in,lmargin=0.75in,rmargin=0.75in")
    # The notebook's own title cell opens the document.
    body = body.replace(r"\maketitle", "")
    body = body.replace(r"\begin{document}",
                        PREAMBLE + "\n" + r"\begin{document}", 1)
    return body, resources


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--keep-tex", action="store_true",
                        help="also write the .tex next to the PDF")
    args = parser.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    tables = prepare(nb)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        body, resources = to_latex(nb, tmp)
        tex = tmp / "notebook.tex"
        tex.write_text(body)
        for name, data in resources.get("outputs", {}).items():
            (tmp / name).write_bytes(data)
        # Three passes so longtable column widths settle.
        for _ in range(3):
            run = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                 tex.name], cwd=tmp, capture_output=True, text=True)
            if run.returncode != 0:
                print(run.stdout[-3000:])
                raise SystemExit("xelatex failed")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "notebook.pdf", args.out)
        if args.keep_tex:
            shutil.copy(tex, args.out.with_suffix(".tex"))
    print(f"Wrote {args.out} ({tables} tables converted).")


if __name__ == "__main__":
    main()
