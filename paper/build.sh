#!/bin/bash
# build.sh -- regenerate numbers/tables/figure and build main.pdf (pdflatex -> bibtex -> pdflatex x2)
set -e
cd "$(dirname "$0")"
python3 gen/mk_ijcai.py > /dev/null
python3 gen/mk_two.py > /dev/null
python3 gen/mk_hires.py > /dev/null
python3 gen/mk_fig3_en.py > /dev/null
python3 gen/mk_fig1_en.py > /dev/null
python3 gen/mk_fig2_en.py > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
grep -E "Warning|Overfull|undefined" build.log | grep -v "Font shape" | sort | uniq -c | head -30 || true
pdfinfo main.pdf | grep Pages
