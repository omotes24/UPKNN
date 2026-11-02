# patch_hires.py -- one-off edit (kept for the record): adds the dit256 tier to run_hires.py, hires_seed.sh, assemble_rotb.py and
#   kx_export_x.py (PROTOCOL_HIRES.md, DiT addendum); refuses to run twice.
import os
C = os.path.expanduser('~/upknn/code')
def edit(f, old, new):
    s = open(f'{C}/{f}').read(); assert s.count(old) == 1, (f, old[:60]); open(f'{C}/{f}', 'w').write(s.replace(old, new))
edit('run_hires.py', "ct_extract_defs.SPEC['e512'] = ('edm2:edm2-img512-xs-uncond-2147483-0.045.pkl', 'lz512_bank_r0', 64)\n",
     "ct_extract_defs.SPEC['e512'] = ('edm2:edm2-img512-xs-uncond-2147483-0.045.pkl', 'lz512_bank_r0', 64)\n"
     "ct_extract_defs.SPEC['dit256'] = ('dit:256', 'lzdit_bank_r0', 32)\n"
     "_groups_for = ct_extract_defs.groups_for\n"
     "def groups_for(net, den, bankset, dev):                                 # DiT: the probe modules of ditwrap.py\n"
     "    return net.dg_groups if hasattr(net, 'dg_groups') else _groups_for(net, den, bankset, dev)\n"
     "ct_extract_defs.groups_for = groups_for\n")
edit('run_hires.py', "    if spec == 'adm:256':\n",
     "    if spec == 'dit:256':\n        import ditwrap; return ditwrap.load_dit(device, fp16=os.environ.get('DG_FP16', '1') != '0')\n    if spec == 'adm:256':\n")
edit('hires_seed.sh', "[ $tag = e512 ] && until [ -f $L/ENC512_DONE ]; do sleep 60; done\nif [ $tag = in256c ]; then PFX=q256-; else PFX=q512-; fi\n",
     "[ $tag = e512 ] && until [ -f $L/ENC512_DONE ]; do sleep 60; done\n[ $tag = dit256 ] && until [ -f $L/ENCDIT_DONE ]; do sleep 60; done\n"
     "case $tag in in256c) PFX=q256-;; e512) PFX=q512-;; dit256) PFX=qdit-;; esac\n")
edit('assemble_rotb.py', "ROOT = os.path.expanduser('~/upknn'); tag, s = sys.argv[1], int(sys.argv[2])\n",
     "ROOT = os.path.expanduser('~/upknn'); tag, s = sys.argv[1], int(sys.argv[2]); RB = {'e512': 'rb512-', 'dit256': 'rbdit-'}[tag]\n")
s = open(f'{C}/assemble_rotb.py').read(); assert s.count("rb512-r{a}") == 2; open(f'{C}/assemble_rotb.py', 'w').write(s.replace("f'{D}/rb512-r{a}", "f'{D}/{RB}r{a}"))
edit('kx_export_x.py', "'e512': ('L012', 1000)}", "'e512': ('L012', 1000), 'dit256': ('L012', 1000)}")
print('PATCH_HIRES_DONE')
