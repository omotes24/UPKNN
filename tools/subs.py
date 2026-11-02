# subs.py -- the only edits made to the GPU-side files (code/, eu/experiments/, code/jobs/, protocols/) when they were copied from the
#   GPU server into this repository: the project root and the Python interpreter of the server are replaced by neutral ones, and the
#   server's host name by a generic word.  Every substitution is exact and invertible (tools/check_freeze.py undoes them to verify the
#   checksums recorded when the protocol was frozen).
SUBS = [
    ('source ~/miniforge3/etc/profile.d/conda.sh && conda activate dg && ', ': && '),     # activate your own environment instead
    ('PY=$HOME/miniforge3/envs/dg/bin/python', 'PY=${PY:-python}'),
    ('PY=~/miniforge3/envs/dg/bin/python', 'PY=${PY:-python}  '),
    ('export PATH=$HOME/miniforge3/envs/dg/bin:$PATH', 'export PATH=$PATH'),
    ('~/dragon_glider', '~/upknn'),
    ('$HOME/dragon_glider', '$HOME/upknn'),
    ('{HOME}/dragon_glider', '{HOME}/upknn'),
    ('hades', 'GPU-server'),
]
INVERTIBLE = SUBS

# Additional substitutions for the ANONYMOUS technical appendix only (tools/make_submission_zip.py): the name of an earlier,
# publicly released detector of the same code base, and the directory that holds its analysis modules.
ANON_SUBS = [('Euphonium', 'PriorE'), ('euphonium', 'priore'), ('EUPHONIUM', 'PRIORE')]
