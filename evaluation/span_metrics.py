import json, argparse
def iou(a,b):
  s1,e1=a; s2,e2=b
  inter=max(0, min(e1,e2)-max(s1,s2))
  uni=max(e1,e2)-min(s1,s2)
  return inter/uni if uni>0 else 0.0
def match_spans(ts, ps, thr=0.5):
  mt=set(); mp=set(); tp=0
  for i,(tl,ts1,te1) in enumerate(ts):
    for j,(pl,ps1,pe1) in enumerate(ps):
      if j in mp or tl!=pl: continue
      if iou((ts1,te1),(ps1,pe1))>=thr:
        tp+=1; mt.add(i); mp.add(j); break
  fp=len(ps)-len(mp); fn=len(ts)-len(mt); return tp,fp,fn
def prf(tp,fp,fn):
  p=tp/(tp+fp) if tp+fp>0 else 0.0
  r=tp/(tp+fn) if tp+fn>0 else 0.0
  f=2*p*r/(p+r) if p+r>0 else 0.0
  return p,r,f
def evaluate(gold, preds):
  import json
  G=[json.loads(l) for l in open(gold,encoding="utf-8")]
  P=[json.loads(l) for l in open(preds,encoding="utf-8")]
  labels=["EMAIL","SECRET"]
  tot={lbl:{"tp":0,"fp":0,"fn":0} for lbl in labels}
  all={"tp":0,"fp":0,"fn":0}
  for g,p in zip(G,P):
    gs=[tuple(s) for s in g["spans"] if s[0] in labels]
    ps=[tuple(s) for s in p.get("spans",[]) if s[0] in labels]
    for lbl in labels:
      g1=[s for s in gs if s[0]==lbl]; p1=[s for s in ps if s[0]==lbl]
      tp,fp,fn=match_spans(g1,p1)
      tot[lbl]["tp"]+=tp; tot[lbl]["fp"]+=fp; tot[lbl]["fn"]+=fn
    tp,fp,fn=match_spans(gs,ps)
    all["tp"]+=tp; all["fp"]+=fp; all["fn"]+=fn
  def fmt(d):
    p,r,f=prf(d["tp"],d["fp"],d["fn"]); return {**d,"precision":p,"recall":r,"f1":f}
  out={"per_label":{k:fmt(v) for k,v in tot.items()}, "overall":fmt(all)}
  print(json.dumps(out, indent=2))
if __name__=='__main__':
  ap=argparse.ArgumentParser(); ap.add_argument('--gold'); ap.add_argument('--preds'); a=ap.parse_args(); evaluate(a.gold,a.preds)
