"""
Agent 4 — Research Agreement Map Dashboard
============================================
Streamlit interface with Mistral AI integration.
Light background UI. Paper upload. expert-level analysis.
"""

import os
import sys
import json
import re
import math
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Mistral API key — set your key here ──────────────────────────────────────
MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY_HERE"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agent 4 — Agreement & Disagreement Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');
:root {
  --bg:#F7F4EF; --surface:#FFFFFF; --surface2:#F0EDE7;
  --border:#DDD9D1; --border-s:#C4BFB5;
  --text:#1A1714; --muted:#6B6560; --muted2:#9C9890;
  --green:#1B4332; --green-bg:#DCFCE7;
  --red:#7C2D12;   --red-bg:#FEF2F2;
  --yellow:#78350F;--yellow-bg:#FFFBEB;
  --purple:#3730A3;--purple-bg:#EEF2FF;
  --blue:#1E40AF;
}
html,body,.stApp{background-color:var(--bg)!important;color:var(--text)!important;font-family:'Instrument Sans',sans-serif!important;}
.stSidebar{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
.stSidebar *{color:var(--text)!important;}
.stButton>button{background:var(--text)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Instrument Sans',sans-serif!important;font-weight:600!important;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px 22px;margin:4px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.metric-card h2{font-family:'DM Serif Display',serif;font-size:2.2rem;margin:0;line-height:1;}
.metric-card p{color:var(--muted);font-size:.82rem;margin:6px 0 0;font-family:'DM Mono',monospace;}
.claim-card{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--blue);border-radius:6px;padding:12px 16px;margin:6px 0;font-family:'DM Mono',monospace;font-size:.82rem;line-height:1.65;color:var(--text);}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.73rem;font-weight:600;font-family:'DM Mono',monospace;margin-right:6px;letter-spacing:.04em;text-transform:uppercase;}
.badge-agreement{background:var(--green-bg);color:var(--green);border:1px solid #A7F3D0;}
.badge-contradiction{background:var(--red-bg);color:var(--red);border:1px solid #FECACA;}
.badge-partial{background:var(--yellow-bg);color:var(--yellow);border:1px solid #FDE68A;}
.badge-novel{background:var(--purple-bg);color:var(--purple);border:1px solid #C7D2FE;}
.section-header{font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.14em;color:var(--muted2);text-transform:uppercase;margin:22px 0 10px;border-bottom:1px solid var(--border);padding-bottom:6px;}
.evidence-box{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-family:'DM Mono',monospace;font-size:.78rem;color:var(--muted);margin:4px 0;line-height:1.6;}
.gap-callout{background:#ECFDF5;border:1px solid #6EE7B7;border-left:3px solid var(--green);border-radius:6px;padding:10px 14px;font-size:.83rem;line-height:1.6;color:var(--green);margin-top:8px;}
.page-title{font-family:'DM Serif Display',serif;font-size:2rem;letter-spacing:-.3px;color:var(--text);margin-bottom:2px;}
div[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
div[data-testid="stRadio"] label{color:var(--text)!important;}
div[data-testid="stSidebar"] div[data-testid="stMetric"] label,
div[data-testid="stSidebar"] div[data-testid="stMetric"] div{color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)


# ── Inline Agent 4 logic ─────────────────────────────────────────────────────
CLAIM_PATTERNS = [
    r"we (show|demonstrate|prove|find|observe|propose|present|establish|confirm|verify)[^.]+\.",
    r"(results|experiments|analysis|findings|study|data) (show|indicate|suggest|reveal|demonstrate)[^.]+\.",
    r"(our|the) (model|method|approach|framework|system|algorithm)[^.]+\.",
    r"(this|the) (paper|work|study|research)[^.]+\.",
    r"(we|our) (achieve|outperform|improve|increase|reduce|exceed|surpass)[^.]+\.",
    r"(accuracy|performance|precision|recall|f1|auc|loss|error)[^.]*(%|percent|\d)[^.]+\.",
    r"(unlike|in contrast|compared to|while|whereas)[^.]+\.",
    r"(however|although|despite|nevertheless)[^.]+\.",
]
NEGATION_WORDS = ["not","no","never","neither","nor","fail","cannot","doesn't","don't","didn't","isn't","aren't","wasn't","opposite","contrary","unlike","disagree","reject","disprove"]
CONTRAST_WORDS = ["however","but","although","yet","while","whereas","in contrast","on the other hand","despite","nevertheless","unlike","counter","instead","rather than"]

def _tok(t): return re.findall(r'\b[a-z]{3,}\b', t.lower())
def _vecs(texts):
    tok = [_tok(t) for t in texts]; N = len(texts)
    df = defaultdict(int)
    for ts in tok:
        for w in set(ts): df[w] += 1
    vecs = []
    for ts in tok:
        tf = defaultdict(float)
        for w in ts: tf[w] += 1
        v = {}
        for w, c in tf.items():
            if ts: v[w] = (c/len(ts)) * math.log((N+1)/(df[w]+1))
        vecs.append(v)
    return vecs
def _cos(v1,v2):
    k = set(v1)&set(v2); dot=sum(v1[x]*v2[x] for x in k)
    m1=math.sqrt(sum(x**2 for x in v1.values())); m2=math.sqrt(sum(x**2 for x in v2.values()))
    return dot/(m1*m2) if m1 and m2 else 0.0
def sem_sim(a,b): v=_vecs([a,b]); return _cos(v[0],v[1])
def _cat(t):
    t=t.lower()
    if any(w in t for w in ["outperform","improve","better","superior","exceed","surpass","achieve"]): return "performance"
    if any(w in t for w in ["propose","introduce","present","framework","method","model","algorithm"]): return "methodology"
    if any(w in t for w in ["find","observe","discover","reveal","show","demonstrate"]): return "finding"
    if any(w in t for w in ["however","unlike","contrast","whereas","but","although"]): return "contrast"
    if any(w in t for w in ["limit","fail","cannot","unable","weakness","drawback"]): return "limitation"
    return "general"
def extract_claims(text, pid):
    sents=re.split(r'(?<=[.!?])\s+',text); claims=[]; seen=set()
    for s in sents:
        s=s.strip()
        if len(s)<30 or s in seen: continue
        score=sum(1 for p in CLAIM_PATTERNS if re.search(p,s,re.IGNORECASE))
        if score>0:
            seen.add(s)
            claims.append({"claim_id":f"{pid}_c{len(claims)+1}","paper_id":pid,"text":s,"claim_score":min(score/3,1.0),"category":_cat(s)})
    return claims
def classify_rel(c1,c2):
    t1,t2=c1["text"],c2["text"]; sim=sem_sim(t1,t2)
    neg1=any(w in t1.lower() for w in NEGATION_WORDS); neg2=any(w in t2.lower() for w in NEGATION_WORDS)
    con1=any(w in t1.lower() for w in CONTRAST_WORDS); con2=any(w in t2.lower() for w in CONTRAST_WORDS)
    n1=[float(m) for m in re.findall(r'\b(\d+\.?\d*)',t1)]; n2=[float(m) for m in re.findall(r'\b(\d+\.?\d*)',t2)]
    num_conf=bool(n1 and n2 and abs(max(n1)-max(n2))>15)
    if sim<0.08: rel,conf="novel",0.75+0.1*c1["claim_score"]
    elif (neg1!=neg2 and sim>0.15) or num_conf or (con1 or con2): rel,conf="contradiction",0.6+0.2*sim
    elif sim>0.45 and c1["category"]==c2["category"]: rel,conf="agreement",0.65+0.3*sim
    elif sim>0.18: rel,conf="partial",0.5+0.2*sim
    else: rel,conf="novel",0.55
    gap_map={"contradiction":f"Empirical study needed to resolve conflict between {c1['paper_id']} and {c2['paper_id']}.","partial":f"Unified framework integrating '{c1['category']}' and '{c2['category']}' is missing.","novel":f"Novel claim from {c1['paper_id']} on '{c1['category']}' lacks corroboration.","agreement":f"Mechanisms of agreed '{c1['category']}' finding remain underexplored."}
    expls={"contradiction":f"Potential contradiction (sim={sim:.2f}). Opposing conclusions on shared subject.","agreement":f"Strong semantic alignment (sim={sim:.2f}) in '{c1['category']}'. Mutual evidential support.","partial":f"Moderate overlap (sim={sim:.2f}). Shared area, differing scope or framing.","novel":f"Low similarity (sim={sim:.2f}). Novel perspective not addressed by the other paper."}
    return {"claim":t1,"papers":[c1["paper_id"],c2["paper_id"]],"relationship":rel,"confidence":round(min(conf,0.99),3),"evidence":[t1[:200],t2[:200]],"uncertainty_score":round(1-min(conf,0.99),3),"research_gap":gap_map[rel],"explanation":expls[rel],"semantic_similarity":round(sim,4),"claim_pair":{"claim1_id":c1["claim_id"],"claim2_id":c2["claim_id"],"claim1_category":c1["category"],"claim2_category":c2["category"]}}

def run_local(papers):
    all_claims={pid:extract_claims(t,pid) for pid,t in papers.items()}
    comps=[]; pids=list(all_claims.keys())
    for i in range(len(pids)):
        for j in range(i+1,len(pids)):
            for c1 in all_claims[pids[i]][:10]:
                best,br=0,None
                for c2 in all_claims[pids[j]][:10]:
                    r=classify_rel(c1,c2)
                    if r["confidence"]>best: best,br=r["confidence"],r
                if br: comps.append(br)
    stats=defaultdict(int)
    for c in comps: stats[c["relationship"]]+=1
    total=len(comps) or 1
    return {"agent":"Agent_4","claims_by_paper":{pid:[c["text"] for c in cl] for pid,cl in all_claims.items()},"comparisons":comps,"summary":{"total_comparisons":len(comps),"agreement_count":stats["agreement"],"contradiction_count":stats["contradiction"],"partial_count":stats["partial"],"novel_count":stats["novel"],"agreement_ratio":round(stats["agreement"]/total,3),"contradiction_ratio":round(stats["contradiction"]/total,3),"avg_confidence":round(sum(c["confidence"] for c in comps)/total,3)}}

def call_mistral(papers, api_key):
    import urllib.request, urllib.error
    ctx="\n\n".join(f"=== PAPER: {pid} ===\n{txt[:2500]}" for pid,txt in papers.items())
    sys_p=('You are a senior program committee member. Extract key scientific claims, compare across paper pairs, classify as agreement/contradiction/partial/novel. '
           'Respond ONLY valid JSON: {"comparisons":[{"claim":"...","papers":["p1","p2"],"relationship":"...","confidence":0.8,"semantic_similarity":0.5,"explanation":"...","evidence":["...","..."],"research_gap":"..."}],"key_insights":{"strongest_agreement":"...","key_contradiction":"...","research_frontier":"..."}}')
    payload=json.dumps({"model":"mistral-large-latest","messages":[{"role":"system","content":sys_p},{"role":"user","content":f"Analyze:\n\n{ctx}"}],"temperature":0.2,"max_tokens":3000}).encode()
    req=urllib.request.Request("https://api.mistral.ai/v1/chat/completions",data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r:
        data=json.loads(r.read())
    content=data["choices"][0]["message"]["content"]
    content=re.sub(r"```(?:json)?","",content).strip().rstrip("`").strip()
    return json.loads(content)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Agent 4")
    st.markdown("**Agreement & Disagreement Analyst**")
    st.markdown("<small style='color:#9C9890;font-family:DM Mono'>Research Intelligence</small>", unsafe_allow_html=True)
    st.markdown("---")
    use_mistral = bool(MISTRAL_API_KEY and MISTRAL_API_KEY != "YOUR_MISTRAL_API_KEY_HERE")
    if use_mistral:
        st.markdown("🟢 **Mistral AI** — active")
    else:
        st.markdown("⚠️ Set `MISTRAL_API_KEY` in dashboard.py")
    st.markdown("---")
    if "results" in st.session_state and st.session_state.results:
        s0 = st.session_state.results.get("summary",{})
        st.metric("Papers Analyzed", len(st.session_state.results.get("claims_by_paper",{})))
        st.metric("Claim Pairs", s0.get("total_comparisons",0))
        st.metric("Avg Confidence", f"{s0.get('avg_confidence',0):.1%}")
        st.markdown("---")
        st.markdown("**Relationship Distribution**")
        for label,key,color in [("Agreement","agreement_count","#1B4332"),("Contradiction","contradiction_count","#7C2D12"),("Partial","partial_count","#78350F"),("Novel","novel_count","#3730A3")]:
            st.markdown(f"<span style='color:{color};font-weight:600'>{label}</span>: **{s0.get(key,0)}**",unsafe_allow_html=True)
    st.markdown("---")
    view = st.radio("View", ["📊 Overview","🔍 Claim Explorer","📈 Confidence Heatmap","💡 Key Insights"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='page-title'>🔬 Agreement & Disagreement Analyst</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6B6560;font-size:.92rem;margin-bottom:1.5rem'>Upload scientific papers — Agent 4 extracts claims, maps semantic relationships, and classifies agreement · contradiction · partial · novel contributions.</p>", unsafe_allow_html=True)

# Upload
st.markdown("<div class='section-header'>Upload Scientific Papers</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload papers (.txt · .md · .tex)", type=["txt","md","tex"], accept_multiple_files=True, label_visibility="collapsed")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__),"..","data","output")

def load_data_papers():
    d=os.path.join(os.path.dirname(__file__),"..","data"); res={}
    for fp in glob.glob(os.path.join(d,"*.txt")):
        pid=os.path.splitext(os.path.basename(fp))[0]
        with open(fp,encoding="utf-8") as f: res[pid]=f.read()
    return res

papers = load_data_papers()
if uploaded_files:
    for uf in uploaded_files:
        papers[os.path.splitext(uf.name)[0]] = uf.read().decode("utf-8","ignore")
    st.success(f"✓ {len(uploaded_files)} file(s) uploaded · {len(papers)} total papers ready")

if papers:
    with st.expander(f"📄 {len(papers)} paper(s) loaded"):
        for pid, text in papers.items():
            st.markdown(f"**{pid}** — {len(text):,} chars")

col_btn, col_info = st.columns([2,5])
with col_btn:
    run_clicked = st.button("▶ Run Analysis", use_container_width=True)
with col_info:
    if not papers: st.warning("Upload papers or add .txt files to `data/`.")
    elif not use_mistral: st.info("Local TF-IDF engine active. Set MISTRAL_API_KEY for LLM-enhanced analysis.")

if run_clicked:
    if not papers: st.error("No papers to analyze."); st.stop()
    with st.spinner("Extracting claims and mapping relationships…"):
        prog = st.progress(0, text="Extracting claims…")
        result = run_local(papers)
        prog.progress(50, text="Classifying relationships…")
        if use_mistral:
            try:
                prog.progress(70, text="Mistral AI enhancing…")
                mi = call_mistral(papers, MISTRAL_API_KEY)
                if mi.get("comparisons"):
                    for mc in mi["comparisons"]:
                        mc.setdefault("claim_pair",{"claim1_category":"general","claim2_category":"general"})
                        mc.setdefault("uncertainty_score",round(1-mc.get("confidence",0.7),3))
                    result["comparisons"] = mi["comparisons"] + result["comparisons"][:5]
                    from collections import Counter
                    cnt=Counter(c["relationship"] for c in result["comparisons"]); total=len(result["comparisons"]) or 1
                    result["summary"].update({"total_comparisons":total,"agreement_count":cnt["agreement"],"contradiction_count":cnt["contradiction"],"partial_count":cnt["partial"],"novel_count":cnt["novel"],"avg_confidence":round(sum(c.get("confidence",0.7) for c in result["comparisons"])/total,3)})
                result["mistral_insights"]=mi.get("key_insights",{})
            except Exception as e:
                st.warning(f"Mistral call failed: {e}")
        prog.progress(90, text="Saving…")
        os.makedirs(OUTPUT_DIR,exist_ok=True)
        with open(os.path.join(OUTPUT_DIR,"agent4_results.json"),"w") as f:
            json.dump({k:v for k,v in result.items() if k!="all_claims_structured"},f,indent=2)
        st.session_state.results = result
        prog.progress(100, text="Done.")
    st.success(f"✅ {result['summary']['total_comparisons']} claim pairs analyzed."); st.rerun()

res_file = os.path.join(OUTPUT_DIR,"agent4_results.json")
if "results" not in st.session_state:
    if os.path.exists(res_file):
        with open(res_file) as f: st.session_state.results=json.load(f)
    else: st.session_state.results=None

results = st.session_state.results
if not results:
    st.markdown("---"); st.info("Upload papers and click **Run Analysis** to begin."); st.stop()

s = results.get("summary",{}); comparisons = results.get("comparisons",[]); claims_by_paper = results.get("claims_by_paper",{})

if view == "📊 Overview":
    st.markdown("## 📊 Analysis Overview")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h2 style='color:#1B4332'>{s.get('agreement_count',0)}</h2><p>🟢 Agreements</p></div>",unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h2 style='color:#7C2D12'>{s.get('contradiction_count',0)}</h2><p>🔴 Contradictions</p></div>",unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h2 style='color:#78350F'>{s.get('partial_count',0)}</h2><p>🟡 Partial Agreements</p></div>",unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h2 style='color:#3730A3'>{s.get('novel_count',0)}</h2><p>🟣 Novel Claims</p></div>",unsafe_allow_html=True)
    if results.get("mistral_insights"):
        ins=results["mistral_insights"]; st.markdown("<div class='section-header'>🧠 Mistral AI Key Insights</div>",unsafe_allow_html=True)
        ma,mb,mc2=st.columns(3)
        with ma: st.success(f"**Strongest Agreement**\n\n{ins.get('strongest_agreement','—')}")
        with mb: st.error(f"**Key Contradiction**\n\n{ins.get('key_contradiction','—')}")
        with mc2: st.info(f"**Research Frontier**\n\n{ins.get('research_frontier','—')}")
    st.markdown("<div class='section-header'>Claims Per Paper</div>",unsafe_allow_html=True)
    for pid,claims in claims_by_paper.items():
        with st.expander(f"📄 {pid} — {len(claims)} claims"):
            for i,claim in enumerate(claims,1):
                st.markdown(f"<div class='claim-card'><b>{i}.</b> {claim}</div>",unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        fig=go.Figure(data=[go.Pie(labels=["Agreement","Contradiction","Partial","Novel"],values=[s.get(k,0) for k in ["agreement_count","contradiction_count","partial_count","novel_count"]],marker=dict(colors=["#1B4332","#7C2D12","#78350F","#3730A3"]),hole=0.42,textfont=dict(size=13,family="DM Mono"))])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#1A1714",family="Instrument Sans"),title="Relationship Distribution",showlegend=True,margin=dict(t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)
    except ImportError: st.info("pip install plotly for charts")

elif view == "🔍 Claim Explorer":
    st.markdown("## 🔍 Claim-by-Claim Explorer")
    if not comparisons: st.warning("No comparisons found."); st.stop()
    cf1,cf2=st.columns(2)
    with cf1: rel_f=st.multiselect("Filter by relationship",["agreement","contradiction","partial","novel"],default=["agreement","contradiction","partial","novel"])
    with cf2: conf_min=st.slider("Min confidence",0.0,1.0,0.0,0.05)
    filtered=[c for c in comparisons if c.get("relationship") in rel_f and c.get("confidence",0)>=conf_min]
    st.markdown(f"**{len(filtered)} comparisons** matching filters")
    for i,comp in enumerate(filtered):
        rel=comp.get("relationship","novel"); conf=comp.get("confidence",0)
        cc="#1B4332" if conf>0.75 else "#78350F" if conf>0.5 else "#7C2D12"
        with st.expander(f"Pair {i+1} | {rel.upper()} | conf={conf:.2f} | {' ↔ '.join(comp.get('papers',[]))}"):
            st.markdown(f"<span class='badge badge-{rel}'>{rel}</span><span style='font-size:.88rem;color:#6B6560'>{' ↔ '.join(comp.get('papers',[]))}</span>",unsafe_allow_html=True)
            st.markdown(f"<div style='margin:12px 0'><div style='font-size:.75rem;color:#9C9890;margin-bottom:4px;font-family:DM Mono'>Confidence: {conf:.1%}</div><div style='background:#F0EDE7;border-radius:4px;height:7px'><div style='background:{cc};width:{conf*100}%;height:7px;border-radius:4px'></div></div></div>",unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Claim</div>",unsafe_allow_html=True)
            st.markdown(f"<div class='claim-card'>{comp.get('claim','')}</div>",unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Evidence</div>",unsafe_allow_html=True)
            for ev in comp.get("evidence",[]): st.markdown(f"<div class='evidence-box'>» {ev}</div>",unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Reasoning Trace</div>",unsafe_allow_html=True)
            st.info(comp.get("explanation","—"))
            if comp.get("research_gap"): st.markdown(f"<div class='gap-callout'>💡 {comp['research_gap']}</div>",unsafe_allow_html=True)

elif view == "📈 Confidence Heatmap":
    st.markdown("## 📈 Confidence Heatmap")
    if not comparisons: st.warning("No data."); st.stop()
    try:
        import plotly.graph_objects as go
        pl=list(claims_by_paper.keys()); n=len(pl)
        matrix=[[0.0]*n for _ in range(n)]; counts=[[0]*n for _ in range(n)]; pi={p:i for i,p in enumerate(pl)}
        for comp in comparisons:
            ps=comp.get("papers",[]
            )
            if len(ps)>=2 and ps[0] in pi and ps[1] in pi:
                i,j=pi[ps[0]],pi[ps[1]]; matrix[i][j]+=comp.get("confidence",0); counts[i][j]+=1
        avg=[[matrix[i][j]/counts[i][j] if counts[i][j]>0 else 0 for j in range(n)] for i in range(n)]
        fig=go.Figure(data=go.Heatmap(z=avg,x=pl,y=pl,colorscale=[[0,"#F7F4EF"],[0.4,"#BFDBFE"],[0.7,"#3B82F6"],[1,"#1E3A8A"]],zmin=0,zmax=1,text=[[f"{v:.2f}" for v in row] for row in avg],texttemplate="%{text}",textfont=dict(size=12,family="DM Mono")))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#1A1714",family="Instrument Sans"),height=420,margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig,use_container_width=True)
    except ImportError: st.info("pip install plotly")

elif view == "💡 Key Insights":
    st.markdown("## 💡 Key Research Insights")
    agreements=sorted([c for c in comparisons if c.get("relationship")=="agreement"],key=lambda x:x.get("confidence",0),reverse=True)[:3]
    contradictions=sorted([c for c in comparisons if c.get("relationship")=="contradiction"],key=lambda x:x.get("confidence",0),reverse=True)[:3]
    st.markdown("<div class='section-header'>🟢 Strongest Agreements</div>",unsafe_allow_html=True)
    if agreements:
        for a in agreements: st.success(f"**{' ↔ '.join(a.get('papers',[]))}** (conf={a.get('confidence',0):.2f})\n\n{a.get('explanation','')}")
    else: st.info("No strong agreements detected.")
    st.markdown("<div class='section-header'>🔴 Key Contradictions</div>",unsafe_allow_html=True)
    if contradictions:
        for c in contradictions: st.error(f"**{' ↔ '.join(c.get('papers',[]))}** (conf={c.get('confidence',0):.2f})\n\n{c.get('explanation','')}")
    else: st.info("No strong contradictions detected.")
    st.markdown("<div class='section-header'>📌 Research Gaps</div>",unsafe_allow_html=True)
    seen=set()
    for comp in comparisons:
        g=comp.get("research_gap","")
        if g and g not in seen: seen.add(g); st.markdown(f"<div class='gap-callout'>💡 {g}</div>",unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Metrics</div>",unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    m1.metric("Agreement Rate",f"{s.get('agreement_ratio',0):.1%}"); m2.metric("Contradiction Rate",f"{s.get('contradiction_ratio',0):.1%}"); m3.metric("Avg Confidence",f"{s.get('avg_confidence',0):.1%}")
