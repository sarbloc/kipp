"""Benchmark untagged cosine matching against live tagged events (issue #33).

Tagged events ([type:id] prefix) carry ground truth for which entity a sentence
belongs to. This strips the tag, runs the per-fact cosine matcher against an
index of agent-written (mcp:/cli:) facts, and reports precision/coverage across
threshold x margin (best entity minus runner-up entity). Near-identical facts
(cos > 0.95) are excluded to avoid leakage from facts extracted from the same
events. Read-only against the configured Qdrant.

Run: python scripts/bench_untagged_matching.py
"""
import re, numpy as np
from entity_memory.client import get_client, collection_name, scroll_entities, DOMAINS
from entity_memory.extract import split_sentences
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2"); c = get_client()
LEAD = re.compile(r"^\s*((?:\[[^\]]*\]\s*)+)")
TAG = re.compile(r"\[([a-z_]+:[^\]\s]+)\]")
rows=[]  # (best, runner, picked_correct)
for d in DOMAINS:
    ents = scroll_entities(c, domain=d)
    facts=[(e.id,f.text) for e in ents for f in e.facts if f.source.startswith(("mcp:","cli:"))]
    if not facts: continue
    owners=np.array([o for o,_ in facts]); M=m.encode([t for _,t in facts], normalize_embeddings=True, batch_size=64)
    ids=set(owners)
    pts,_=c.scroll(collection_name(d,"events"), limit=1000, with_payload=True)
    for p in pts:
        t=p.payload["text"]; lm=LEAD.match(t)
        if not lm: continue
        tags=TAG.findall(lm.group(1))
        if not tags or tags[0] not in ids: continue
        target=tags[0]; body=t[lm.end():]
        sents=split_sentences(body)
        if not sents: continue
        S=m.encode(sents, normalize_embeddings=True)
        for s in S:
            sc=M@s
            sc=np.where(sc>0.95,-1,sc)  # drop near-identical (leakage)
            b=int(np.argmax(sc)); bo=owners[b]
            other=sc[owners!=bo]; ru=float(other.max()) if other.size else -1
            rows.append((float(sc[b]), float(sc[b])-ru, bo==target))
rows=np.array(rows,dtype=float); print("sentences:",len(rows))
for th in (0.32,0.4,0.45,0.5):
    for mg in (0,0.02,0.04,0.06,0.08,0.1):
        sel=(rows[:,0]>=th)&(rows[:,1]>=mg)
        cov=sel.mean(); prec=rows[sel,2].mean() if sel.any() else float('nan')
        print(f"th={th:.2f} margin={mg:.2f} coverage={cov:.2f} precision={prec:.2f}")
