"""
Agent 5 — Uncertainty & Research Gap Dashboard
================================================
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
    page_title="Agent 5 — Uncertainty & Gap Analyst",
    page_icon="🧭",
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
  --critical:#9F1239; --critical-bg:#FFF1F2;
  --high:#C2410C;    --high-bg:#FFF7ED;
  --medium:#A16207;  --medium-bg:#FEFCE8;
  --low:#166534;     --low-bg:#F0FDF4;
  --accent:#1B4332;  --accent-bg:#DCFCE7;
  --warn:#7C2D12;    --warn-bg:#FEF2F2;
}
html,body,.stApp{background-color:var(--bg)!important;color:var(--text)!important;font-family:'Instrument Sans',sans-serif!important;}
.stSidebar{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
.stSidebar *{color:var(--text)!important;}
.stButton>button{background:var(--text)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Instrument Sans',sans-serif!important;font-weight:600!important;}
.gap-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 22px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.gap-card:hover{border-color:var(--border-s);}
.gap-rank{font-family:'DM Serif Display',serif;font-size:2rem;font-weight:700;color:var(--muted2);line-height:1;}
.tier-badge{font-family:'DM Mono',monospace;font-size:.72rem;padding:3px 10px;border-radius:20px;font-weight:600;letter-spacing:.05em;}
.tier-critical{background:var(--critical-bg);color:var(--critical);border:1px solid #FECDD3;}
.tier-high{background:var(--high-bg);color:var(--high);border:1px solid #FED7AA;}
.tier-medium{background:var(--medium-bg);color:var(--medium);border:1px solid #FDE68A;}
.tier-low{background:var(--low-bg);color:var(--low);border:1px solid #BBF7D0;}
.score-row{display:flex;gap:12px;margin:10px 0 6px;flex-wrap:wrap;}
.score-item{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:5px 11px;font-family:'DM Mono',monospace;font-size:.74rem;color:var(--muted);}
.score-item b{color:var(--text);margin-left:4px;}
.section-label{font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted2);padding-bottom:6px;border-bottom:1px solid var(--border);margin:22px 0 10px;}
.rec-box{background:#ECFDF5;border:1px solid #6EE7B7;border-left:3px solid var(--accent);border-radius:6px;padding:10px 14px;font-size:.83rem;line-height:1.6;margin-top:8px;color:var(--accent);}
.conflict-card{background:var(--warn-bg);border:1px solid #FECACA;border-left:4px solid var(--warn);border-radius:8px;padding:14px 18px;margin:8px 0;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px 22px;margin:4px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.metric-card h2{font-family:'DM Serif Display',serif;font-size:2.2rem;margin:0;line-height:1;}
.metric-card p{color:var(--muted);font-size:.82rem;margin:6px 0 0;font-family:'DM Mono',monospace;}
.meta-chip{display:inline-block;background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:'DM Mono',monospace;font-size:.72rem;color:var(--accent);margin:2px;}
.page-title{font-family:'DM Serif Display',serif;font-size:2rem;letter-spacing:-.3px;color:var(--text);margin-bottom:2px;}
div[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
div[data-testid="stRadio"] label{color:var(--text)!important;}
div[data-testid="stSidebar"] div[data-testid="stMetric"] label,
div[data-testid="stSidebar"] div[data-testid="stMetric"] div{color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)


# ── Inline Agent 5 logic ─────────────────────────────────────────────────────
HEDGING = [
    (r"\b(may|might|could|possibly|perhaps|probably|likely|seemingly|apparently)\b",0.7,"epistemic_hedge"),
    (r"\b(suggest|indicate|appear|seem|imply)\b",0.65,"soft_claim"),
    (r"\b(limited|preliminary|pilot|exploratory|initial)\b",0.6,"scope_limit"),
    (r"\b(further (work|study|research|investigation) (is|are)? (needed|required|warranted))\b",0.8,"gap_signal"),
    (r"\b(unclear|unknown|uncertain|ambiguous|inconclusive)\b",0.85,"explicit_uncertainty"),
    (r"\b(to the best of our knowledge|as far as we know)\b",0.75,"knowledge_boundary"),
    (r"\b(we assume|we hypothesize|we conjecture|we speculate)\b",0.7,"assumption"),
    (r"\b(small (sample|dataset|cohort|set))\b",0.65,"data_weakness"),
    (r"\b(not (yet|fully|completely) (understood|explored|validated|tested))\b",0.8,"incomplete_knowledge"),
    (r"\b(more (data|evidence|experiments|studies) (is|are)? needed)\b",0.85,"evidence_gap"),
    (r"\b(future work|future research|future studies)\b",0.75,"future_work"),
    (r"\b(cannot (be|fully) (generalized|applied|extended))\b",0.75,"generalizability_limit"),
    (r"\b(limitation|drawback|weakness|shortcoming|caveat)\b",0.8,"limitation"),
]
METH_WEAK = [
    (r"\b(n\s*=\s*\d{1,2})\b",0.7,"small_n"),
    (r"\b(\d{1,2}\s+subjects?|\d{1,2}\s+participants?)\b",0.65,"small_sample"),
    (r"\b(no (baseline|comparison|control|ablation))\b",0.7,"missing_baseline"),
    (r"\b(synthetic|simulated|toy|artificial) (data|dataset)\b",0.65,"synthetic_data"),
]
GAP_TRIGGERS = [
    (r"\b(future work|future research|future studies)\b","future_direction"),
    (r"\b(open (question|problem|challenge|issue))\b","open_question"),
    (r"\b(remains? (unknown|unexplored|understudied|open))\b","unexplored"),
    (r"\b(has not been (studied|explored|investigated|addressed))\b","unstudied"),
    (r"\b(lack (of|ing) (data|evidence|benchmarks?|datasets?))\b","data_gap"),
    (r"\b(further (investigation|analysis|study) (is|are)? (needed|required))\b","investigation_needed"),
    (r"\b(promising direction|potential (avenue|direction|extension))\b","promising_direction"),
    (r"\b(underexplored|understudied|overlooked|neglected)\b","underexplored"),
]
GAP_CATS = {
    "future_direction":("Future Research Direction",0.9,0.8),
    "open_question":("Open Scientific Question",0.95,0.6),
    "unexplored":("Unexplored Area",0.85,0.7),
    "unstudied":("Unstudied Problem",0.9,0.65),
    "data_gap":("Missing Data / Benchmark",0.8,0.75),
    "investigation_needed":("Investigation Required",0.85,0.8),
    "promising_direction":("Promising Direction",0.7,0.9),
    "underexplored":("Underexplored Domain",0.8,0.7),
    "epistemic_hedge":("Epistemic Uncertainty",0.6,0.6),
    "data_weakness":("Weak Empirical Evidence",0.75,0.7),
    "small_n":("Insufficient Sample Size",0.85,0.9),
    "missing_baseline":("Missing Baseline Comparison",0.8,0.85),
    "generalizability_limit":("Generalizability Unknown",0.9,0.65),
    "limitation":("Acknowledged Limitation",0.7,0.75),
    "gap_signal":("Evidence Gap",0.8,0.75),
    "future_work":("Future Work Stated",0.75,0.85),
}
RECS = {
    "future_direction":"Design a follow-up study extending stated future directions.",
    "open_question":"Formulate this as a falsifiable hypothesis and design controlled experiments.",
    "unexplored":"Conduct a systematic survey + pilot study of this unexplored area.",
    "unstudied":"A first-of-its-kind empirical study here would be high-impact.",
    "data_gap":"Curate a dedicated benchmark dataset. Establish evaluation protocols.",
    "investigation_needed":"Design controlled experiments with proper baselines.",
    "promising_direction":"Develop a prototype exploring this direction, evaluate against baselines.",
    "underexplored":"A systematic study would provide foundational knowledge for the field.",
    "epistemic_hedge":"Re-examine hedged claims with stronger experimental designs.",
    "data_weakness":"Replicate with larger, more diverse datasets to validate findings.",
    "small_n":"Reproduce at scale — power analysis to determine minimum N.",
    "missing_baseline":"Add missing baselines; compare against established state-of-the-art.",
    "generalizability_limit":"Test across diverse domains, languages, datasets.",
    "limitation":"Address acknowledged limitations directly in follow-up work.",
    "gap_signal":"Gather more evidence; design experiments to fill this gap.",
    "future_work":"This deferred work represents a natural, high-value extension.",
}

def score_uncertainty(sent):
    s=sent.lower(); hits=[]; spans=[]; total=0; count=0
    for pat,w,label in HEDGING:
        m=re.search(pat,s)
        if m: hits.append(label); spans.append(m.group(0)); total+=w; count+=1
    for pat,w,label in METH_WEAK:
        m=re.search(pat,s)
        if m: hits.append(label); spans.append(m.group(0)); total+=w; count+=1
    strong=sum(1 for p in [r"\b(definitively|conclusively|clearly|undoubtedly)\b",r"\b(always|never|all|none|every)\b",r"\b(proves?|establishes?|confirms?)\b",r"\b(state-of-the-art|best|optimal)\b"] if re.search(p,s))
    total=max(0,total-strong*0.15)
    if count==0: return 0.0,[],[]
    return round(min(total/(count+1),1.0),3),hits,spans

def detect_gaps(sent):
    s=sent.lower(); gaps=[]
    for pat,gtype in GAP_TRIGGERS:
        m=re.search(pat,s)
        if m: gaps.append({"type":gtype,"trigger":m.group(0),"sentence":sent.strip()})
    return gaps

def analyze_paper(pid, text):
    sents=[s.strip() for s in re.split(r'(?<=[.!?])\s+',text) if len(s.strip())>25]
    u_sents=[]; gap_sigs=[]; u_types=defaultdict(int); total_u=0
    for sent in sents:
        u,types,spans=score_uncertainty(sent)
        if u>0.1:
            u_sents.append({"sentence":sent,"uncertainty_score":u,"uncertainty_types":types,"uncertainty_spans":spans})
            total_u+=u
            for t in types: u_types[t]+=1
        gap_sigs.extend(detect_gaps(sent))
    seen=set(); unique_gaps=[]
    for g in gap_sigs:
        k=(g["type"],g["trigger"])
        if k not in seen: seen.add(k); unique_gaps.append(g)
    avg_u=total_u/max(len(sents),1)
    return {"paper_id":pid,"total_sentences":len(sents),"uncertain_sentence_count":len(u_sents),"avg_uncertainty":round(avg_u,4),"uncertainty_ratio":round(len(u_sents)/max(len(sents),1),3),"dominant_uncertainty_types":dict(sorted(u_types.items(),key=lambda x:-x[1])[:5]),"uncertain_sentences":sorted(u_sents,key=lambda x:-x["uncertainty_score"])[:8],"gap_signals":unique_gaps[:10]}

def generate_gaps(paper_analyses):
    gaps=[]; seen=set()
    for pid,a in paper_analyses.items():
        for gs in a.get("gap_signals",[]):
            gtype=gs["type"]; sent=gs["sentence"]; k=(gtype,sent[:60])
            if k in seen: continue
            seen.add(k)
            cat,impact,feas=GAP_CATS.get(gtype,("Research Gap",0.7,0.6))
            nov=0.8; comp=round(impact*0.45+feas*0.3+nov*0.25,3)
            gaps.append({"gap_id":f"G{len(gaps)+1:03d}","paper_source":pid,"gap_type":gtype,"category":cat,"description":sent,"trigger_phrase":gs["trigger"],"impact_score":impact,"feasibility_score":feas,"novelty_score":nov,"composite_score":comp,"recommendation":RECS.get(gtype,f"Investigate the gap from {pid}."),"priority":"HIGH" if comp>0.75 else "MEDIUM" if comp>0.55 else "LOW"})
        for us in a.get("uncertain_sentences",[])[:3]:
            for utype in us.get("uncertainty_types",[]):
                k=(utype,us["sentence"][:60])
                if k in seen: continue
                seen.add(k)
                cat,impact,feas=GAP_CATS.get(utype,("Research Gap",0.6,0.6))
                nov=0.65; comp=round(impact*0.45+feas*0.3+nov*0.25,3)
                gaps.append({"gap_id":f"G{len(gaps)+1:03d}","paper_source":pid,"gap_type":utype,"category":cat,"description":us["sentence"],"trigger_phrase":", ".join(us.get("uncertainty_spans",[])[:2]),"impact_score":impact,"feasibility_score":feas,"novelty_score":nov,"composite_score":comp,"recommendation":RECS.get(utype,f"Investigate gap from {pid}."),"priority":"HIGH" if comp>0.75 else "MEDIUM" if comp>0.55 else "LOW"})
    gaps.sort(key=lambda x:-x["composite_score"])
    for i,g in enumerate(gaps):
        g["rank"]=i+1
        sc=g["composite_score"]
        g["tier"]="🔴 Critical" if sc>0.80 else "🟠 High" if sc>0.70 else "🟡 Medium" if sc>0.55 else "🟢 Low"
    return gaps

def detect_conflicts(paper_analyses):
    conflicts=[]; pids=list(paper_analyses.keys())
    for i in range(len(pids)):
        for j in range(i+1,len(pids)):
            p1,p2=pids[i],pids[j]; u1=paper_analyses[p1].get("avg_uncertainty",0); u2=paper_analyses[p2].get("avg_uncertainty",0); diff=abs(u1-u2)
            if diff>0.08:
                more=p1 if u1>u2 else p2; less=p2 if u1>u2 else p1
                conflicts.append({"paper_1":p1,"paper_2":p2,"paper_1_uncertainty":u1,"paper_2_uncertainty":u2,"uncertainty_gap":round(diff,4),"interpretation":f"{more} expresses significantly higher uncertainty than {less}. May indicate methodological differences, data limitations, or genuine scientific disagreement about confidence levels.","severity":"HIGH" if diff>0.2 else "MEDIUM"})
    return sorted(conflicts,key=lambda x:-x["uncertainty_gap"])

def run_local(papers):
    analyses={pid:analyze_paper(pid,txt) for pid,txt in papers.items()}
    gaps=generate_gaps(analyses); conflicts=detect_conflicts(analyses)
    from collections import Counter
    pc=Counter(g["priority"] for g in gaps)
    return {"agent":"Agent_5","paper_analyses":analyses,"research_gaps":gaps,"cross_paper_conflicts":conflicts,"summary":{"total_papers":len(papers),"total_gaps":len(gaps),"high_priority_gaps":pc["HIGH"],"medium_priority_gaps":pc["MEDIUM"],"low_priority_gaps":pc["LOW"],"avg_uncertainty_overall":round(sum(a["avg_uncertainty"] for a in analyses.values())/max(len(analyses),1),4),"cross_paper_conflicts":len(conflicts)}}

def call_mistral(papers, api_key):
    import urllib.request
    ctx="\n\n".join(f"=== PAPER: {pid} ===\n{txt[:2500]}" for pid,txt in papers.items())
    sys_p=('You are a senior researcher specializing in uncertainty quantification and research gap analysis. '
           'Identify hedging language, methodological weaknesses, open questions. Rank gaps by Impact × Feasibility × Novelty. '
           'Respond ONLY valid JSON: {"research_gaps":[{"gap_id":"G001","paper_source":"...","gap_type":"future_direction|open_question|data_gap|...","category":"...","description":"...","trigger_phrase":"...","impact_score":0.85,"feasibility_score":0.75,"novelty_score":0.8,"composite_score":0.82,"recommendation":"...","priority":"HIGH|MEDIUM|LOW","tier":"🟠 High","rank":1}],"key_findings":{"most_uncertain_paper":"...","top_gap":"...","critical_missing":"..."}}')
    payload=json.dumps({"model":"mistral-large-latest","messages":[{"role":"system","content":sys_p},{"role":"user","content":f"Analyze:\n\n{ctx}"}],"temperature":0.2,"max_tokens":3000}).encode()
    req=urllib.request.Request("https://api.mistral.ai/v1/chat/completions",data=payload,headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r:
        data=json.loads(r.read())
    content=data["choices"][0]["message"]["content"]
    content=re.sub(r"```(?:json)?","",content).strip().rstrip("`").strip()
    return json.loads(content)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Agent 5")
    st.markdown("**Uncertainty & Research Gap Analyst**")
    st.markdown("<small style='color:#9C9890;font-family:DM Mono'>Research Intelligence</small>",unsafe_allow_html=True)
    st.markdown("---")
    use_mistral=bool(MISTRAL_API_KEY and MISTRAL_API_KEY!="YOUR_MISTRAL_API_KEY_HERE")
    if use_mistral: st.markdown("🟢 **Mistral AI** — active")
    else: st.markdown("⚠️ Set `MISTRAL_API_KEY` in dashboard.py")
    st.markdown("---")
    if "results5" in st.session_state and st.session_state.results5:
        s0=st.session_state.results5.get("summary",{})
        st.metric("Total Gaps",s0.get("total_gaps",0))
        st.metric("High Priority",s0.get("high_priority_gaps",0))
        st.metric("Avg Uncertainty",f"{s0.get('avg_uncertainty_overall',0):.1%}")
        st.metric("Conflicts",s0.get("cross_paper_conflicts",0))
    st.markdown("---")
    view=st.radio("View",["🏠 Overview","❓ Research Gaps","📊 Uncertainty Heatmap","⚡ Conflict Detector","🧩 Gap Clusters","📋 Full Table"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='page-title'>🧭 Uncertainty & Research Gap Analyst</h1>",unsafe_allow_html=True)
st.markdown("<p style='color:#6B6560;font-size:.92rem;margin-bottom:1.5rem'>Upload scientific papers — Agent 5 detects uncertainty, identifies open research questions, and ranks gaps by Impact × Feasibility × Novelty.</p>",unsafe_allow_html=True)

st.markdown("<div class='section-label'>Upload Scientific Papers</div>",unsafe_allow_html=True)
uploaded_files=st.file_uploader("Upload papers (.txt · .md · .tex)",type=["txt","md","tex"],accept_multiple_files=True,label_visibility="collapsed")

OUTPUT_DIR=os.path.join(os.path.dirname(__file__),"..","data","output")

def load_data_papers():
    d=os.path.join(os.path.dirname(__file__),"..","data"); res={}
    for fp in glob.glob(os.path.join(d,"*.txt")):
        pid=os.path.splitext(os.path.basename(fp))[0]
        with open(fp,encoding="utf-8") as f: res[pid]=f.read()
    return res

papers=load_data_papers()
if uploaded_files:
    for uf in uploaded_files:
        papers[os.path.splitext(uf.name)[0]]=uf.read().decode("utf-8","ignore")
    st.success(f"✓ {len(uploaded_files)} file(s) uploaded · {len(papers)} total papers ready")

if papers:
    with st.expander(f"📄 {len(papers)} paper(s) loaded"):
        for pid,txt in papers.items():
            st.markdown(f"**{pid}** — {len(txt):,} chars")

col_btn,col_info=st.columns([2,5])
with col_btn: run_clicked=st.button("▶ Run Analysis",use_container_width=True)
with col_info:
    if not papers: st.warning("Upload papers or add .txt files to `data/`.")
    elif not use_mistral: st.info("Local analysis engine active. Set MISTRAL_API_KEY for LLM-enhanced gaps.")

if run_clicked:
    if not papers: st.error("No papers to analyze."); st.stop()
    with st.spinner("Detecting uncertainty and mapping research gaps…"):
        prog=st.progress(0,text="Analyzing uncertainty…")
        result=run_local(papers)
        prog.progress(55,text="Ranking gaps…")
        if use_mistral:
            try:
                prog.progress(70,text="Mistral AI enhancing…")
                mi=call_mistral(papers,MISTRAL_API_KEY)
                if mi.get("research_gaps"):
                    for g in mi["research_gaps"]:
                        g.setdefault("paper_source",list(papers.keys())[0])
                        g.setdefault("gap_type","future_direction"); g.setdefault("novelty_score",0.8)
                    result["research_gaps"]=mi["research_gaps"]+result["research_gaps"][:5]
                    for i,g in enumerate(result["research_gaps"]): g["rank"]=i+1
                result["mistral_findings"]=mi.get("key_findings",{})
                from collections import Counter
                pc2=Counter(g.get("priority","LOW") for g in result["research_gaps"])
                result["summary"].update({"total_gaps":len(result["research_gaps"]),"high_priority_gaps":pc2["HIGH"],"medium_priority_gaps":pc2["MEDIUM"],"low_priority_gaps":pc2["LOW"]})
            except Exception as e: st.warning(f"Mistral call failed: {e}")
        prog.progress(90,text="Saving…")
        os.makedirs(OUTPUT_DIR,exist_ok=True)
        with open(os.path.join(OUTPUT_DIR,"agent5_results.json"),"w") as f: json.dump(result,f,indent=2)
        st.session_state.results5=result; prog.progress(100,text="Done.")
    st.success(f"✅ {result['summary']['total_gaps']} gaps identified."); st.rerun()

res_file=os.path.join(OUTPUT_DIR,"agent5_results.json")
if "results5" not in st.session_state:
    if os.path.exists(res_file):
        with open(res_file) as f: st.session_state.results5=json.load(f)
    else: st.session_state.results5=None

results=st.session_state.results5
if not results: st.markdown("---"); st.info("Upload papers and click **Run Analysis** to begin."); st.stop()

s=results.get("summary",{}); gaps=results.get("research_gaps",[]); analyses=results.get("paper_analyses",{}); conflicts=results.get("cross_paper_conflicts",[])

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if view=="🏠 Overview":
    st.markdown("## 🏠 Overview")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h2 style='color:#9F1239'>{s.get('high_priority_gaps',0)}</h2><p>🔴 High Priority Gaps</p></div>",unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h2 style='color:#A16207'>{s.get('medium_priority_gaps',0)}</h2><p>🟡 Medium Priority</p></div>",unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h2 style='color:#1B4332'>{s.get('total_gaps',0)}</h2><p>📌 Total Gaps</p></div>",unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h2 style='color:#7C2D12'>{s.get('cross_paper_conflicts',0)}</h2><p>⚡ Conflicts</p></div>",unsafe_allow_html=True)
    if results.get("mistral_findings"):
        mf=results["mistral_findings"]; st.markdown("<div class='section-label'>🧠 Mistral AI Key Findings</div>",unsafe_allow_html=True)
        ma,mb,mc=st.columns(3)
        with ma: st.error(f"**Most Uncertain**\n\n{mf.get('most_uncertain_paper','—')}")
        with mb: st.warning(f"**Top Gap**\n\n{mf.get('top_gap','—')}")
        with mc: st.info(f"**Critical Missing**\n\n{mf.get('critical_missing','—')}")
    st.markdown("<div class='section-label'>Per-Paper Uncertainty</div>",unsafe_allow_html=True)
    for pid,a in analyses.items():
        u=a.get("avg_uncertainty",0)
        color="#9F1239" if u>0.25 else "#C2410C" if u>0.15 else "#166534"
        st.markdown(f"<div style='margin:8px 0'><div style='font-size:.8rem;color:#6B6560;font-family:DM Mono;margin-bottom:3px'>{pid} — {u:.1%} avg uncertainty · {a.get('uncertain_sentence_count',0)} uncertain sentences</div><div style='background:#F0EDE7;border-radius:4px;height:8px'><div style='background:{color};width:{min(u*300,100)}%;height:8px;border-radius:4px'></div></div></div>",unsafe_allow_html=True)
    if gaps:
        try:
            import plotly.graph_objects as go
            from collections import Counter
            tc=Counter(g.get("gap_type","other") for g in gaps).most_common(8)
            fig=go.Figure(go.Bar(x=[t[1] for t in tc],y=[t[0].replace("_"," ").title() for t in tc],orientation="h",marker=dict(color="#1B4332",opacity=0.8),text=[t[1] for t in tc],textposition="auto",textfont=dict(family="DM Mono",size=11)))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#1A1714",family="Instrument Sans"),height=300,margin=dict(l=10,r=10,t=30,b=10),title="Top Gap Types",xaxis=dict(showgrid=True,gridcolor="#DDD9D1"),yaxis=dict(showgrid=False))
            st.plotly_chart(fig,use_container_width=True)
        except ImportError: st.info("pip install plotly for charts")

# ── RESEARCH GAPS ─────────────────────────────────────────────────────────────
elif view=="❓ Research Gaps":
    st.markdown("## ❓ Research Gap Dashboard")
    st.caption("Ranked by composite score: Impact (45%) × Feasibility (30%) × Novelty (25%)")
    if not gaps: st.warning("No gaps found."); st.stop()
    cf1,cf2,cf3=st.columns(3)
    with cf1: pf=st.multiselect("Priority",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM","LOW"])
    with cf2:
        paper_opts=list(set(g.get("paper_source","") for g in gaps))
        papf=st.multiselect("Source Paper",paper_opts,default=paper_opts)
    with cf3: min_sc=st.slider("Min Composite Score",0.0,1.0,0.0,0.05)
    filtered=[g for g in gaps if g.get("priority","LOW") in pf and g.get("paper_source","") in papf and g.get("composite_score",0)>=min_sc]
    st.markdown(f"**{len(filtered)} gaps** matching filters")
    for gap in filtered:
        tier=gap.get("tier","🟢 Low"); tc2="critical" if "Critical" in tier else "high" if "High" in tier else "medium" if "Medium" in tier else "low"
        imp=gap.get("impact_score",0); feas=gap.get("feasibility_score",0); nov=gap.get("novelty_score",0); comp=gap.get("composite_score",0)
        st.markdown(f"""<div class='gap-card'>
        <div style='display:flex;align-items:flex-start;gap:16px'>
          <div class='gap-rank'>{gap.get('rank','—')}</div>
          <div style='flex:1'>
            <div style='margin-bottom:6px'>
              <span class='tier-badge tier-{tc2}'>{tier}</span>
              <span style='font-size:.78rem;color:#9C9890;font-family:DM Mono;margin-left:8px'>{gap.get('paper_source','')} · {gap.get('gap_type','').replace('_',' ')}</span>
            </div>
            <div style='font-size:.9rem;line-height:1.6;margin-bottom:8px'>{gap.get('description','')}</div>
            <div class='score-row'>
              <span class='score-item'>IMP <b>{imp:.2f}</b></span>
              <span class='score-item'>FEAS <b>{feas:.2f}</b></span>
              <span class='score-item'>NOV <b>{nov:.2f}</b></span>
              <span class='score-item' style='font-weight:600'>COMP <b>{comp:.3f}</b></span>
            </div>
            <div class='rec-box'>💡 {gap.get('recommendation','')}</div>
          </div>
        </div></div>""",unsafe_allow_html=True)

# ── UNCERTAINTY HEATMAP ───────────────────────────────────────────────────────
elif view=="📊 Uncertainty Heatmap":
    st.markdown("## 📊 Uncertainty Analysis Heatmap")
    if not analyses: st.warning("No paper analyses."); st.stop()
    try:
        import plotly.graph_objects as go
        paper_list=list(analyses.keys())
        all_types=sorted(set(t for a in analyses.values() for t in a.get("dominant_uncertainty_types",{}).keys()))
        z=[[analyses[p].get("dominant_uncertainty_types",{}).get(t,0) for t in all_types] for p in paper_list]
        if z and all_types:
            fig=go.Figure(data=go.Heatmap(z=z,x=[t.replace("_"," ") for t in all_types],y=paper_list,colorscale=[[0,"#F7F4EF"],[0.4,"#FDE68A"],[0.7,"#F59E0B"],[1,"#92400E"]],text=z,texttemplate="%{text}",textfont=dict(size=11,family="DM Mono")))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#1A1714",family="Instrument Sans"),height=380,margin=dict(l=10,r=10,t=40,b=80),title="Uncertainty Type Frequency per Paper",xaxis=dict(tickangle=-30,tickfont=dict(size=10)),yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig,use_container_width=True)
    except ImportError: st.info("pip install plotly")
    st.markdown("<div class='section-label'>Uncertain Sentences by Paper</div>",unsafe_allow_html=True)
    for pid,a in analyses.items():
        with st.expander(f"📄 {pid} — {a.get('uncertain_sentence_count',0)} uncertain sentences | avg: {a.get('avg_uncertainty',0):.1%}"):
            for us in a.get("uncertain_sentences",[]):
                score=us.get("uncertainty_score",0); color="#9F1239" if score>0.5 else "#C2410C" if score>0.3 else "#A16207"
                st.markdown(f"<div style='background:#F7F4EF;border-left:3px solid {color};border-radius:4px;padding:10px 14px;margin:6px 0;font-size:.82rem;line-height:1.6'><div style='color:#9C9890;font-family:DM Mono;font-size:.7rem;margin-bottom:4px'>uncertainty: {score:.1%} · types: {', '.join(us.get('uncertainty_types',[])[:3])}</div>{us.get('sentence','')}</div>",unsafe_allow_html=True)

# ── CONFLICT DETECTOR ─────────────────────────────────────────────────────────
elif view=="⚡ Conflict Detector":
    st.markdown("## ⚡ Cross-Paper Uncertainty Conflicts")
    st.caption("Papers expressing significantly different levels of certainty on related topics.")
    if not conflicts: st.success("✅ No significant cross-paper uncertainty conflicts detected.")
    else:
        for cf in conflicts:
            sev_color="#9F1239" if cf.get("severity")=="HIGH" else "#C2410C"
            st.markdown(f"""<div class='conflict-card'>
            <div style='font-family:DM Mono;font-size:.75rem;color:{sev_color};margin-bottom:8px;font-weight:600'>⚡ CONFLICT — {cf.get('severity','MEDIUM')}</div>
            <div style='font-size:.95rem;font-weight:600;margin-bottom:8px'>{cf['paper_1']} ↔ {cf['paper_2']}</div>
            <div style='display:flex;gap:20px;margin-bottom:10px;font-family:DM Mono;font-size:.8rem;color:#6B6560'>
              <span>{cf['paper_1']}: <b style='color:#1A1714'>{cf['paper_1_uncertainty']:.1%}</b></span>
              <span>{cf['paper_2']}: <b style='color:#1A1714'>{cf['paper_2_uncertainty']:.1%}</b></span>
              <span>Δ gap: <b style='color:{sev_color}'>{cf['uncertainty_gap']:.1%}</b></span>
            </div>
            <div style='font-size:.83rem;line-height:1.6;color:#6B6560'>{cf.get('interpretation','')}</div>
            </div>""",unsafe_allow_html=True)

# ── GAP CLUSTERS ──────────────────────────────────────────────────────────────
elif view=="🧩 Gap Clusters":
    st.markdown("## 🧩 Research Gap Clusters by Theme")
    if not gaps: st.warning("No gaps."); st.stop()
    THEME_PATTERNS={"Data & Benchmarks":["data","dataset","benchmark","corpus","sample","annotation"],"Methodology":["method","model","architecture","algorithm","approach","framework"],"Evaluation":["evaluat","metric","measure","baseline","comparison","validation"],"Generalizability":["general","domain","transfer","robust","universal","diverse"],"Scalability":["scale","large","production","efficiency","deploy","real-world"],"Interpretability":["interpret","explain","transparent","understand","visuali"],"Ethics & Safety":["bias","fair","safe","harm","privacy","ethic"],"Theoretical":["theory","proof","bound","formal","theorem"],"Applications":["appli","use case","downstream"]}
    clusters=defaultdict(list)
    for g in gaps:
        txt=(g.get("description","")+g.get("gap_type","")+g.get("category","")).lower(); assigned=False
        for theme,kws in THEME_PATTERNS.items():
            if any(kw in txt for kw in kws): clusters[theme].append(g); assigned=True; break
        if not assigned: clusters["Other"].append(g)
    COLORS={"Data & Benchmarks":"#1B4332","Methodology":"#3730A3","Evaluation":"#C2410C","Generalizability":"#166534","Scalability":"#9F1239","Interpretability":"#A16207","Ethics & Safety":"#7C3AED","Theoretical":"#1E40AF","Applications":"#0F766E","Other":"#6B6560"}
    try:
        import plotly.graph_objects as go
        tl=list(clusters.keys()); ts=[len(v) for v in clusters.values()]; tc3=[COLORS.get(t,"#6B6560") for t in tl]
        fig=go.Figure(data=[go.Bar(x=tl,y=ts,marker=dict(color=tc3,opacity=0.85),text=ts,textposition="auto",textfont=dict(family="DM Mono",size=12))])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#1A1714",family="Instrument Sans"),height=300,margin=dict(l=10,r=10,t=30,b=60),title="Gaps per Research Theme",xaxis=dict(tickangle=-25),yaxis=dict(showgrid=True,gridcolor="#DDD9D1"))
        st.plotly_chart(fig,use_container_width=True)
    except ImportError: pass
    for theme,tg in clusters.items():
        color=COLORS.get(theme,"#6B6560")
        with st.expander(f"**{theme}** — {len(tg)} gaps"):
            for g in tg[:6]:
                st.markdown(f"<div style='border-left:3px solid {color};background:#F7F4EF;border-radius:4px;padding:10px 14px;margin:6px 0;font-size:.82rem;line-height:1.6'><span style='font-family:DM Mono;color:{color};font-size:.7rem'>[{g.get('priority','?')}] score={g.get('composite_score',0):.3f}</span><br/>{g.get('description','')}</div>",unsafe_allow_html=True)

# ── FULL TABLE ────────────────────────────────────────────────────────────────
elif view=="📋 Full Table":
    st.markdown("## 📋 Complete Research Gap Table")
    if not gaps: st.warning("No gaps."); st.stop()
    try:
        import pandas as pd
        df=pd.DataFrame([{"Rank":g.get("rank",""),"Tier":g.get("tier",""),"Category":g.get("category",""),"Paper":g.get("paper_source",""),"Type":g.get("gap_type","").replace("_"," "),"Impact":f"{g.get('impact_score',0):.2f}","Feasibility":f"{g.get('feasibility_score',0):.2f}","Novelty":f"{g.get('novelty_score',0):.2f}","Composite":f"{g.get('composite_score',0):.3f}","Priority":g.get("priority",""),"Description":g.get("description","")[:100]+"..."} for g in gaps])
        st.dataframe(df,use_container_width=True,height=600)
        st.download_button("⬇️ Download CSV",df.to_csv(index=False),"research_gaps.csv","text/csv")
    except ImportError:
        for g in gaps: st.write(f"**#{g.get('rank')}** {g.get('category')} | {g.get('composite_score',0):.3f} | {g.get('description','')[:100]}")
