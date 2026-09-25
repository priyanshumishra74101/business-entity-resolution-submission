#!/usr/bin/env python3
"""Local two-stage entity resolution: capped blocking -> logistic pair classifier.

Only supplied TSVs are used.  `candidate_pairs.tsv` is the union of capped blocks;
`matching_results.tsv` contains only candidates above a validation-tuned threshold.
"""
from __future__ import annotations

import argparse, csv, hashlib, math, re, sqlite3, sys, unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:  # dependency is pinned; graceful fallback keeps reproducibility
    sanscript = None

LEGAL = {"inc","incorporated","corp","corporation","co","company","llc","ltd","limited","llp","plc","pvt","private","gmbh","sa","sas","sarl","bv","ag","holdings","group"}
ABBR = {"rd":"road","st":"street","ave":"avenue","blvd":"boulevard","ln":"lane","dr":"drive","hwy":"highway","pvt":"private","ltd":"limited","corp":"corporation","co":"company","ste":"suite","apt":"apartment","fl":"floor"}
ADDRESS_STOP = {"road","street","avenue","lane","drive","boulevard","building","floor","near","the","and","of","at","block","sector","nagar","main","west","east","north","south","apt","suite","unit"}

def bucket(s: str, n: int) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) % n

def latin(s: str) -> str:
    # Rule-based script conversion; never performs an identity lookup.
    if sanscript and any('\u0900' <= c <= '\u097f' for c in s):
        s = transliterate(s, sanscript.DEVANAGARI, sanscript.IAST)
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii").lower()
    return s.replace("&", " and ")

def tokens(s: str) -> list[str]:
    return [ABBR.get(x, x) for x in re.findall(r"[a-z0-9]+", latin(s))]

def norm(name: str, address: str, country: str) -> dict[str, str]:
    nt, at = tokens(name), tokens(address)
    core = [x for x in nt if x not in LEGAL]
    nums = [x for x in at if x.isdigit()]
    postal = next((x for x in reversed(nums) if 4 <= len(x) <= 8), "")
    street = max((x for x in at if len(x) > 2 and x not in ADDRESS_STOP and not x.isdigit()), key=lambda x:(len(x),x), default="")
    return {"country": latin(country).strip() or "unknown", "name": " ".join(nt), "core": " ".join(core),
            "addr": " ".join(at), "addr_sorted": " ".join(sorted(at)), "postal": postal,
            "number": nums[0] if nums else "", "street": street, "zipstreet": postal+"|"+street[:6] if postal and street else ""}

def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f: yield from csv.DictReader(f, delimiter="\t")

SCHEMA = """CREATE TABLE records (entity_id TEXT PRIMARY KEY,country TEXT,name TEXT,core TEXT,addr TEXT,addr_sorted TEXT,postal TEXT,number TEXT,street TEXT,zipstreet TEXT);"""

def build_index(db: Path, files: list[Path]) -> None:
    # An interrupted SQLite construction must never be reused as a complete index.
    # Expected row count is stored only after all source rows and all block indexes
    # are committed.
    expected = sum(1 for f in files for _ in rows(f))
    if db.exists():
        try:
            chk = sqlite3.connect(db)
            ready = chk.execute("SELECT value FROM metadata WHERE key='record_count'").fetchone()
            chk.close()
            if ready and int(ready[0]) == expected:
                return
        except sqlite3.Error:
            pass
        raise RuntimeError(f"Incomplete/stale index at {db}; use a new --work-dir or remove it before rerunning.")
    db.parent.mkdir(parents=True, exist_ok=True); con=sqlite3.connect(db)
    con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF;"+SCHEMA)
    b=[]; total=0
    for f in files:
        for r in rows(f):
            n=norm(r["business_name"],r["business_address"] or "",r["country"] or "unknown")
            b.append((r["entity_id"],n["country"],n["name"],n["core"],n["addr"],n["addr_sorted"],n["postal"],n["number"],n["street"],n["zipstreet"]))
            if len(b)>=50000: con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",b); con.commit();total+=len(b);b=[]
    if b: con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",b);total+=len(b)
    con.executescript("CREATE INDEX core_ix ON records(country,core); CREATE INDEX addr_ix ON records(country,addr_sorted); CREATE INDEX zip_ix ON records(country,zipstreet); CREATE INDEX postal_ix ON records(country,postal,core);")
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
    con.execute("INSERT INTO metadata VALUES ('record_count',?)",(str(total),))
    con.commit();con.close(); print(f"indexed {total:,}: {db.name}",file=sys.stderr)

def candidates(con: sqlite3.Connection, n: dict[str,str], cap: int=25, legacy: bool=False) -> list[tuple]:
    out={}
    table = "records"
    projection = "*" if not legacy else "entity_id,country,business_name,name_core,address_key,address_sorted,postal,house_number,street_anchor,zip_street"
    def add(sql, p):
        got=list(con.execute(sql+" LIMIT ?",(*p,cap+1)))
        if len(got)<=cap:
            for r in got: out[r[0]]=r
    # Composite / capped blocks. Empty keys are never used.
    core_col = "name_core" if legacy else "core"
    addr_col = "address_sorted" if legacy else "addr_sorted"
    zip_col = "zip_street" if legacy else "zipstreet"
    if n["core"]: add(f"SELECT {projection} FROM {table} WHERE country=? AND {core_col}=?",(n["country"],n["core"]))
    if n["addr_sorted"]: add(f"SELECT {projection} FROM {table} WHERE country=? AND {addr_col}=?",(n["country"],n["addr_sorted"]))
    if n["zipstreet"]: add(f"SELECT {projection} FROM {table} WHERE country=? AND {zip_col}=?",(n["country"],n["zipstreet"]))
    if n["postal"] and n["core"]: add(f"SELECT {projection} FROM {table} WHERE country=? AND postal=? AND {core_col}=?",(n["country"],n["postal"],n["core"]))
    return [out[k] for k in sorted(out)]

def jaccard(a: str,b: str)->float:
    x,y=set(a.split()),set(b.split()); return len(x&y)/len(x|y) if x or y else 0.0

def make_idf(texts: list[str]) -> dict[str,float]:
    df=Counter(); N=len(texts)
    for s in texts:
        df.update(set(s[i:i+3] for i in range(max(1,len(s)-2))))
    return {g:math.log((N+1)/(c+1))+1 for g,c in df.items()}

def tfidf_cos(a: str,b: str,idf: dict[str,float])->float:
    ca=Counter(a[i:i+3] for i in range(max(1,len(a)-2))); cb=Counter(b[i:i+3] for i in range(max(1,len(b)-2)))
    dot=sum(c*cb.get(g,0)*idf.get(g,1.0)**2 for g,c in ca.items())
    na=math.sqrt(sum(c*c*idf.get(g,1.0)**2 for g,c in ca.items())); nb=math.sqrt(sum(c*c*idf.get(g,1.0)**2 for g,c in cb.items()))
    return dot/(na*nb) if na and nb else 0.0

def feat(s: dict[str,str], n: dict[str,str], r: tuple, idf: dict[str,float]) -> list[float]:
    # record: id,country,name,core,addr,addr_sorted,postal,number,street,zipstreet
    cn,cc,ca,cpostal,cnum,cstreet=r[2],r[3],r[4],r[6],r[7],r[8]
    return [fuzz.ratio(n["name"],cn)/100, fuzz.token_sort_ratio(n["core"],cc)/100, jaccard(n["name"],cn),
            fuzz.token_sort_ratio(n["addr"],ca)/100, jaccard(n["addr"],ca), tfidf_cos(n["name"],cn,idf),
            float(n["country"]==r[1]),float(bool(n["addr"]) and bool(ca)),float(bool(n["postal"] and n["postal"]==cpostal)),
            float(bool(n["number"] and n["number"]==cnum)),float(bool(n["street"] and n["street"]==cstreet)),
            abs(len(n["name"])-len(cn))/max(1,max(len(n["name"]),len(cn))),abs(len(n["name"].split())-len(cn.split()))/max(1,max(len(n["name"].split()),len(cn.split())))]

class LR:
    def __init__(self,d): self.w=[0.0]*d;self.b=-2.0
    def p(self,x):
        z=max(-30,min(30,self.b+sum(a*b for a,b in zip(self.w,x))));return 1/(1+math.exp(-z))
    def fit(self,ex):
        for epoch in range(10):
            rate=.08/(1+.25*epoch)
            for x,y in ex:
                d=rate*(3 if y else 1)*(y-self.p(x));self.b+=d
                for i,v in enumerate(x):self.w[i]+=d*v

def truth_sample(path:Path, denom:int=200)->dict[str,set[str]]:
    ans={}
    for r in rows(path):
        if bucket(r["source1_entity_id"],denom)==0: ans[r["source1_entity_id"]]=set(filter(None,r["matched_entity_ids"].split(",")))
    return ans

def score(pred:set[str], actual:set[str])->float:
    if not actual:return float(not pred)
    if not pred:return 0.
    tp=len(pred&actual);p=tp/len(pred);rec=tp/len(actual);return 1.25*p*rec/(.25*p+rec) if tp else 0.

def train(train:Path,work:Path):
    db=work/'two_stage_train.sqlite';build_index(db,[train/'train_source2.tsv',train/'train_source3.tsv']); truth=truth_sample(train/'train_ground_truth.tsv')
    con=sqlite3.connect(db); raw=[]; texts=[]
    for s in rows(train/'train_source1.tsv'):
        actual=truth.get(s['entity_id'])
        if actual is None: continue
        n=norm(s['business_name'],s['business_address'] or '',s['country'] or 'unknown'); cs=candidates(con,n)
        raw.append((s,n,actual,cs));texts.append(n['name']);texts.extend(r[2] for r in cs)
    idf=make_idf(texts); ex=[]; val=[]; coverage=[]
    for s,n,actual,cs in raw:
        candidate_ids={r[0] for r in cs}; coverage.extend([x in candidate_ids for x in actual]); pairs=[(r[0],feat(s,n,r,idf),int(r[0] in actual)) for r in cs]
        # Salted split is independent from the deterministic sampling predicate.
        if bucket(s['entity_id']+'|validation',5)==0:val.append((actual,pairs))
        else:
            neg=0
            for _,x,y in pairs:
                if y or neg<12:ex.append((x,y));neg+=not y
    model=LR(13);model.fit(ex); best=(0.,.7)
    for i in range(65,96):
        t=i/100;v=sum(score({cid for cid,x,_ in ps if model.p(x)>=t},a) for a,ps in val)/len(val)
        if v>best[0]:best=(v,t)
    print(f"validation_f05={best[0]:.4f}; threshold={best[1]:.2f}; candidate_recall={sum(coverage)/len(coverage):.4f}; pairs={len(ex):,}",file=sys.stderr)
    con.close();return model,idf,best[1],best[0]

def predict(test:Path,work:Path,out:Path,model:LR,idf:dict[str,float],threshold:float,legacy_test_index:Path|None=None):
    legacy = legacy_test_index is not None
    db = legacy_test_index if legacy else work/'two_stage_test.sqlite'
    if not legacy: build_index(db,[test/'test_source2.tsv',test/'test_source3.tsv'])
    con=sqlite3.connect(db);out.mkdir(exist_ok=True,parents=True)
    total=cand_n=match_n=0
    with (out/'candidate_pairs.tsv').open('w',encoding='utf-8',newline='') as cf,(out/'matching_results.tsv').open('w',encoding='utf-8',newline='') as mf:
        cw=csv.writer(cf,delimiter='\t',lineterminator='\n');mw=csv.writer(mf,delimiter='\t',lineterminator='\n');cw.writerow(['source1_entity_id','candidate_entity_ids']);mw.writerow(['source1_entity_id','matched_entity_ids'])
        for s in rows(test/'test_source1.tsv'):
            n=norm(s['business_name'],s['business_address'] or '',s['country'] or 'unknown');cs=candidates(con,n,legacy=legacy);ids=[r[0] for r in cs];matches=[r[0] for r in cs if model.p(feat(s,n,r,idf))>=threshold]
            cw.writerow([s['entity_id'],','.join(ids)]);mw.writerow([s['entity_id'],','.join(matches)])
            total+=1;cand_n+=len(ids);match_n+=len(matches)
            if total%100000==0:print(f"scored {total:,}; candidates {cand_n:,}; matches {match_n:,}",file=sys.stderr,flush=True)
    con.close();print(f"final: rows={total:,}, candidates={cand_n:,}, matches={match_n:,}",file=sys.stderr)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--work-dir',type=Path,required=True);ap.add_argument('--legacy-test-index',type=Path);a=ap.parse_args()
    m,idf,t,_=train(a.data_root/'train',a.work_dir);predict(a.data_root/'test',a.work_dir,a.output_dir,m,idf,t,a.legacy_test_index)
if __name__=='__main__':main()
