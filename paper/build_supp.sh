#!/bin/bash
# build_supp.sh -- regenerate the supplementary tables and build supp.pdf
# The ``named'' citation macros are part of ijcai27.sty; supp.tex is a plain article, so they are
# copied once into named.sty (the block keeps its original notice).
set -e
cd "$(dirname "$0")"
sed -n '/^%%%% named.sty/,/^%%%% use parentheses/p' ijcai27.sty | sed '$d' > named.sty
python3 gen/mk_supp.py > /dev/null
python3 gen/mk_ijcai.py > /dev/null     # tab_fusion.tex
python3 gen/mk_two.py > /dev/null       # numbers_two.tex, supp_two*.tex
pdflatex -interaction=nonstopmode -halt-on-error supp.tex > build_supp.log
bibtex supp >> build_supp.log
pdflatex -interaction=nonstopmode -halt-on-error supp.tex >> build_supp.log
pdflatex -interaction=nonstopmode -halt-on-error supp.tex >> build_supp.log
grep -E "Warning|Overfull|undefined" build_supp.log | grep -v "Font shape" | sort | uniq -c | head -30 || true
pdfinfo supp.pdf | grep Pages
