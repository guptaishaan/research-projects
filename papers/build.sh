#!/usr/bin/env bash
# Build one or all papers: figures -> LaTeX (with BibTeX) -> PDF -> source zip.
#
#   ./build.sh                 # build every paper
#   ./build.sh raddose-triage  # build one
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$ROOT")"
PY="${PAPER_PYTHON:-python3}"

PAPERS=(
  heart-disease-cross-hospital
  clip-forgetting-forecasting
  bankruptcy-stress-test
  exoplanet-model-comparison
  raddose-triage
  eeg-controls-audit
)
[ $# -gt 0 ] && PAPERS=("$@")

for name in "${PAPERS[@]}"; do
  dir="$ROOT/$name"
  echo "=== $name"
  cp "$ROOT/_common/paperstyle.sty" "$dir/paperstyle.sty"

  if [ -f "$dir/make_figures.py" ]; then
    (cd "$REPO" && "$PY" "$dir/make_figures.py" >/dev/null)
  fi

  (
    cd "$dir"
    rm -f main.aux main.bbl main.blg main.out main.toc main.log \
          main.fls main.fdb_latexmk
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    bibtex main >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    pdflatex -interaction=nonstopmode main.tex >/dev/null

    bad=$(grep -cE "Warning: (Citation|Reference).*undefined" main.log || true)
    over=$(grep -c "Overfull" main.log || true)
    pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
    echo "    pages=$pages  undefined-refs=$bad  overfull=$over"

    mv -f main.pdf "$name.pdf"
    rm -f main.aux main.blg main.out main.toc main.fls main.fdb_latexmk

    rm -f "$name-latex.zip"
    zip -qr "$name-latex.zip" \
        main.tex refs.bib paperstyle.sty main.bbl figures \
        make_figures.py README.md \
        -x '*.DS_Store'
    echo "    -> $name.pdf, $name-latex.zip"
  )
done
