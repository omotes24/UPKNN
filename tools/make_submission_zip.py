# make_submission_zip.py -- builds submission/upknn_technical_appendix.zip, the anonymous technical appendix of the submission:
#   the supplementary material (PDF) and the code, protocols, benchmark definitions, analysis scripts and result files of this
#   repository, with tools/subs.py ANON_SUBS applied to every text file (and to the path eu_build/Euphonium), no licence holder
#   and no repository URL.  Fails if an identifying string remains.   usage (repository root): python tools/make_submission_zip.py
import os, re, sys, io, zipfile, fnmatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subs import ANON_SUBS
OUT = 'submission/upknn_technical_appendix.zip'
INCLUDE = ['README.md', 'code', 'eu', 'runs/ok', 'protocols', 'analysis', 'tools', 'environment', 'third_party', 'checkpoints/README.md',
           'data/README.md', 'paper/gen', 'paper/supp.tex', 'paper/build_supp.sh']
SKIP = ['*/__pycache__/*', '*.pyc', 'tools/make_submission_zip.py', 'tools/anon_terms.txt']
TEXT = ('.py', '.sh', '.md', '.txt', '.json', '.jsonl', '.tex', '.md5', '.bib')
# identifying terms (names, user names, affiliations) are read from tools/anon_terms.txt, one per line (not part of the
#   repository); generic patterns (home paths, e-mail addresses, repository URLs, the substituted names) are always checked
_terms = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anon_terms.txt'))] if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anon_terms.txt')) else []
BAD = re.compile('|'.join([re.escape(t) for t in _terms if t] + [r'dragon_glider', r'\bhades\b', r'/Users/', r'/home/[a-z]',
                 r'[A-Za-z0-9._-]+@[A-Za-z0-9-]+\.[A-Za-z.]+', r'euphonium']), re.I)
LICENCE = ('MIT License\n\nCopyright (c) 2026 Anonymous Authors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and '
           'associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, '
           'copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished '
           'to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or '
           'substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT '
           'LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT '
           'HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN '
           'CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n')
NOTE = ('Technical appendix of the submission "Neighbours and Orientation: Out-of-Distribution Detection with Frozen Diffusion\n'
        'Features" (anonymous; the detector is called Upright kNN).\n\n'
        '- supplementary.pdf: the supplementary material (protocol details, comparison methods, the proof of Proposition 2, the weight\n'
        '  sweep, the two-notions analysis with its controls, all pairs, FPR95, ablations, the mirror test, the ImageNet tables incl. the\n'
        '  bank-size and resampling controls, fusions, CSI, EuroSAT / chest X-rays / histology / rotation-trained generators, compute).\n'
        '- README.md and the folders: code (GPU side, exactly as run), protocols (each written before its evaluation; FREEZE_UKNN.md5 with the\n'
        '  checksums recorded before the final evaluation, verified by tools/check_freeze.py), benchmark definitions (runs/ok, eu/runs/jobs),\n'
        '  analysis scripts and the result files of the paper (paper/gen/data).\n'
        '- "PriorE" is an earlier detector of the authors, renamed for anonymity; it is not part of the submission. Because of this\n'
        '  renaming, tools/check_freeze.py reports protocols/PROTOCOL_UKNN.md as different from its recorded checksum (the code files match);\n'
        '  the non-anonymous release verifies all six files.\n')


def anon(t):
    for a, b in ANON_SUBS: t = t.replace(a, b)
    return t


def files():
    for inc in INCLUDE:
        if os.path.isfile(inc): yield inc; continue
        for dp, _, fs in os.walk(inc):
            for f in fs:
                p = os.path.join(dp, f)
                if not any(fnmatch.fnmatch(p, s) for s in SKIP): yield p


os.makedirs('submission', exist_ok=True); bad = []
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(files()):
        arc = 'upknn/' + anon(p)
        if p.endswith(TEXT):
            t = anon(open(p, encoding='utf-8').read())
            if p == 'tools/subs.py':        # the substitution lists themselves name what they replace: keep only the generic part
                t = t.split('\n# Additional substitutions')[0] + '\n'
            for i, line in enumerate(t.splitlines(), 1):
                if BAD.search(line) and p != 'tools/subs.py': bad.append(f'{p}:{i}: {line.strip()[:120]}')
            z.writestr(arc, t)
        else:
            z.write(p, arc)
    z.writestr('upknn/LICENSE', LICENCE); z.writestr('upknn/APPENDIX_README.txt', NOTE)
    z.write('paper/supp.pdf', 'upknn/supplementary.pdf')
if bad:
    print('IDENTIFYING STRINGS:\n' + '\n'.join(bad[:40])); os.remove(OUT); sys.exit(1)
print(OUT, f'{os.path.getsize(OUT) / 2**20:.1f} MB', len(zipfile.ZipFile(OUT).namelist()), 'files')
