import json
R={}
for l in open('/home/hittu/kv-offload/results/exp/main.jsonl'):
    r=json.loads(l); R[(r['method'],r['task'],r['length'])]=r
for L in (4096,8192,16384,20480):
  for t in ('niah_multivalue','vt','niah_single_2','niah_multikey_1'):
    s=json.loads(open(f'/home/hittu/kv-offload/data/ruler/{L}/{t}/validation.jsonl').readline())
    inp=s['input']; n=len(inp)
    pos={a: round(inp.find(a)/n,2) for a in s['outputs']}
    found={m:[a for a in s['outputs'] if a in R[(m,t,L)]['output']] for m in ('stream_1024','stream_2048') if (m,t,L) in R}
    print(L,t,'answer pos (fraction of input chars):',pos,'| stream found:',found)
