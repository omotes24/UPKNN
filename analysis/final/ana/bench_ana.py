# bench_ana.py -- Table A2: CPU seconds per image of every method in ONE session (runs/bench/bench_cpu2.jsonl from the GPU server)
import json, os
rows = [json.loads(l) for l in open(f'{os.path.dirname(os.path.abspath(__file__))}/../data/runs/bench/bench_cpu2.jsonl')]
last = {}
for r in rows: last[r['method']] = r                       # the latest measurement of every method
json.dump(last, open('bench_cpu2.json', 'w'), indent=1)
for m, r in last.items(): print(f"{m:10s} {r['sec_per_img']:9.2f} s/img  NFE {r['nfe_fwd_per_img']:8.0f}  VJP {r['vjp_per_img']:5.0f}  threads {r['threads']}  {r.get('cpu','')}")
