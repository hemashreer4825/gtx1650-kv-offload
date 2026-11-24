import json, statistics as st, collections
R=[json.loads(l) for l in open('/home/hittu/kv-offload/results/exp/main.jsonl')]
# keep the final record per (method,task,length) (error was redone)
fin={}
for r in R: fin[(r['method'],r['task'],r['length'])]=r
ok=[r for r in fin.values() if r['status']=='ok']
print('records',len(R),'final',len(fin),collections.Counter(r['status'] for r in R),'final status',collections.Counter(r['status'] for r in fin.values()))
print('time span',min(r['time'] for r in R),max(r['time'] for r in R))
# 1 token identity gpu vs offload
for L in (4096,8192):
  for t in ('niah_single_2','niah_multikey_1','niah_multivalue','vt'):
    g=fin[('gpu',t,L)]; o=fin[('offload',t,L)]
    print('identity',L,t,'same_output',g['output']==o['output'],'gen',g['gen_tokens'],o['gen_tokens'])
# 2 input tokens per task/length
for L in (4096,8192,16384,20480):
  for t in ('niah_single_2','niah_multikey_1','niah_multivalue','vt'):
    xs=[(r['ctx_tokens'],r['q_tokens']) for r in ok if r['length']==L and r['task']==t]
    print('tokens',L,t,'ctx',sorted(set(x[0] for x in xs)),'q',sorted(set(x[1] for x in xs)))
# 3 per-run clocks/temps/waits
print('sm_avg per run',min(r['gpu_sm_clock_avg'] for r in ok),max(r['gpu_sm_clock_avg'] for r in ok))
print('temp max per run',min(r['gpu_temp_max'] for r in ok),max(r['gpu_temp_max'] for r in ok))
print('cool wait',min(r['cool_wait_s'] for r in ok),max(r['cool_wait_s'] for r in ok))
print('throttle frac',min(r['thermal_throttle_frac'] for r in ok),'power cap frac',min(r['power_cap_frac'] for r in ok))
print('swap max',max(r['swap_growth_mib'] for r in ok))
# 4 per-task decode spread vs mean, gpu & offload
for m in ('gpu','offload'):
  for L in (4096,8192,16384,20480):
    xs=[r['decode_tok_s'] for r in ok if r['method']==m and r['length']==L]
    if xs: mu=st.mean(xs); print('spread',m,L,round(mu,3),'max dev %',round(100*max(abs(x-mu) for x in xs)/mu,1))
# 5 peaks: allocated vs reserved
for (m,L) in [('gpu',4096),('offload',20480),('snapkv_2048',20480)]:
  xs=[r for r in ok if r['method']==m and r['length']==L]
  print('peak',m,L,'alloc',max(r['peak_gpu_alloc_mib'] for r in xs),'reserved',max(r['peak_gpu_reserved_mib'] for r in xs))
# 6 snapkv outputs
for r in ok:
  if r['method'].startswith('snapkv') and r['task']=='niah_single_2': print('snap',r['length'],r['method'],r['refs'],repr(r['output'][:60]))
# 7 offload 20480 gen tokens
for r in ok:
  if r['method']=='offload' and r['length']==20480: print('off20k',r['task'],'gen',r['gen_tokens'],'decode_s',r['decode_s'],'prefill',r['prefill_s'],'score',r['score'])
# 8 kept tokens
print('kept',sorted(set((r['method'],r['kv_kept']) for r in ok if r['method']!='gpu' and r['method']!='offload')))
