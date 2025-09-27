# ============================================
# app.py — ShopUNow AI Assistant (Streamlit Cloud, Stable)
# ============================================

import os
import json
import faiss
import random
import threading
import traceback
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz

# LangChain + LangGraph
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# =========================================================
# 1️⃣ Build FAQ Vector Store
# =========================================================
_FAQ_VECTOR_STORE = None
_FAQ_DOCS: List[Document] = []
_FAQ_LOCK = threading.Lock()

def resolve_faq_path() -> str:
    """Look for FAQ file in project root or /data."""
    candidates = ["shopunow_faqs.jsonl", os.path.join("data", "shopunow_faqs.jsonl")]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("❌ shopunow_faqs.jsonl not found — please commit it next to app.py or upload via sidebar.")

def load_faq_documents(path: str) -> List[Document]:
    """Read JSONL FAQs into LangChain documents."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                q, a = rec.get("question", "").strip(), rec.get("answer", "").strip()
                dept = rec.get("department", "unknown").strip()
                if not q or not a:
                    continue
                docs.append(Document(
                    page_content=f"Q: {q}\nA: {a}",
                    metadata={"question": q, "answer": a, "department": dept}
                ))
            except json.JSONDecodeError:
                continue
    return docs

def build_faq_store(docs: List[Document]) -> FAISS:
    """Build FAISS store with OpenAI embeddings."""
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")
    dim = len(emb.embed_query("hello"))
    index = faiss.IndexFlatIP(dim)
    store = FAISS(embedding_function=emb, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    store.add_documents(docs, ids=[f"faq_{i}" for i in range(len(docs))])
    return store

def get_faq_vector_store() -> Tuple[FAISS, List[Document]]:
    """Lazy initializer for FAQ store."""
    global _FAQ_VECTOR_STORE, _FAQ_DOCS
    if _FAQ_VECTOR_STORE:
        return _FAQ_VECTOR_STORE, _FAQ_DOCS
    with _FAQ_LOCK:
        if not _FAQ_VECTOR_STORE:
            path = resolve_faq_path()
            _FAQ_DOCS = load_faq_documents(path)
            _FAQ_VECTOR_STORE = build_faq_store(_FAQ_DOCS)
            print(f"✅ Loaded {len(_FAQ_DOCS)} FAQs", flush=True)
    return _FAQ_VECTOR_STORE, _FAQ_DOCS

# =========================================================
# 2️⃣ Agent Logic
# =========================================================
random.seed(42)
np.random.seed(42)

_SENTIMENT = None
def detect_sentiment(text: str) -> Literal["negative","neutral","positive"]:
    global _SENTIMENT
    if _SENTIMENT is None:
        _SENTIMENT = SentimentIntensityAnalyzer()
    score = _SENTIMENT.polarity_scores(text or "").get("compound", 0.0)
    return "negative" if score <= -0.3 else "positive" if score >= 0.3 else "neutral"

DEPT_KEYWORDS = {
    "Orders & Returns": ["order","return","refund","delivery","package","tracking","exchange","cancel"],
    "Payments & Billing": ["payment","card","upi","wallet","invoice","coupon","billing","price"],
    "Customer Support": ["support","complaint","help","agent","contact","phone","email"],
    "HR & IT Helpdesk": ["salary","payroll","policy","leave","password","vpn","hrms","hardware"],
}

def classify_department_with_confidence(query: str) -> Tuple[Optional[str], float, Dict[str,int]]:
    text = query.lower()
    scores = {d: sum(1 for k in kws if k in text) for d,kws in DEPT_KEYWORDS.items()}
    top, val = max(scores.items(), key=lambda x:x[1])
    if val == 0:
        return None, 0.0, scores
    conf = 0.6 if val == 1 else 0.8
    return top, conf, scores

DEPT_SIM_THRESHOLDS = {"Orders & Returns":0.8,"Payments & Billing":0.78,"Customer Support":0.75,"HR & IT Helpdesk":0.8,None:0.8}

class AgentState(BaseModel):
    user_input: str
    department: Optional[str]=None
    dept_confidence: float=0.0
    sentiment: Optional[str]=None
    intent: Optional[str]=None
    tools_used: List[str]=Field(default_factory=list)
    answer: Optional[str]=None
    confidence: float=0.0

def extract_answer_text(page_content: str)->str:
    if not page_content: return ""
    lc=page_content.lower()
    if "a:" in lc:
        return page_content[lc.find("a:")+2:].strip()
    return page_content

def route_intent(state: AgentState)->Dict[str,Any]:
    q=state.user_input.lower()
    sent=detect_sentiment(q)
    dept,conf,_=classify_department_with_confidence(q)
    if sent=="negative": intent="human_escalation"
    elif any(k in q for k in ["track","where is my order"]): intent="order_status"
    elif "return" in q: intent="return_create"
    else: intent="rag"
    return {"intent":intent,"department":dept,"dept_confidence":conf,"sentiment":sent}

def tool_node(state: AgentState)->Dict[str,Any]:
    q=state.user_input.strip()
    dept,conf,intent=state.department,state.dept_confidence,state.intent
    if intent=="order_status":
        return {"answer":"Your order is being processed.","tools_used":["order_status"],"confidence":float(0.9)}
    if intent=="return_create":
        return {"answer":"Return initiated. You'll get pickup details by email.","tools_used":["return"],"confidence":float(0.9)}
    if intent=="human_escalation":
        return {"answer":"Escalating to human support.","tools_used":["escalation"],"confidence":float(0.3)}
    if intent=="rag":
        if not dept or conf<0.6:
            return {"answer":"This may relate to multiple areas. Escalating.","tools_used":["escalation"],"confidence":float(0.3)}
        try:
            store,docs=get_faq_vector_store()
            results=store.similarity_search_with_score(q,k=5)
            filtered=[(d,float(s)) for d,s in results if d.metadata.get("department")==dept] or [(d,float(s)) for d,s in results]
            if not filtered:
                return {"answer":"No relevant info found.","tools_used":["escalation"],"confidence":float(0.2)}
            doc,sim=filtered[0]
            sim=float(sim)
            th=float(DEPT_SIM_THRESHOLDS.get(dept,0.8))
            if sim<th:
                best=None;bf=0.0
                for d in docs:
                    fs=fuzz.partial_ratio(q,d.metadata.get("question",""))/100.0
                    if fs>bf: best,bf=d,fs
                if best and bf>=0.92:
                    return {"answer":extract_answer_text(best.page_content),"tools_used":["fuzzy"],"confidence":float(bf)}
                return {"answer":"I'm not confident. Escalating.","tools_used":["escalation"],"confidence":sim}
            return {"answer":extract_answer_text(doc.page_content),"tools_used":["rag"],"confidence":sim}
        except Exception as e:
            traceback.print_exc()
            return {"answer":f"Search error: {e}","tools_used":["error"],"confidence":float(0.0)}
    return {"answer":"Please rephrase your question.","tools_used":["fallback"],"confidence":float(0.3)}

graph=StateGraph(AgentState)
graph.add_node("route",route_intent)
graph.add_node("tool",tool_node)
graph.add_edge(START,"route")
graph.add_edge("route","tool")
graph.add_edge("tool",END)
app=graph.compile(checkpointer=MemorySaver())

def ask(query:str)->str:
    out=app.invoke({"user_input":query},config={"configurable":{"thread_id":"streamlit"}})
    return out.get("answer","No answer.")

# =========================================================
# 3️⃣ Streamlit UI
# =========================================================
st.set_page_config(page_title="ShopUNow Assistant", layout="centered")
st.title("🛍️ ShopUNow AI Assistant")

# Sidebar setup
st.sidebar.header("⚙️ Setup & Testing")

faq_path = "shopunow_faqs.jsonl"
if os.path.exists(faq_path):
    st.sidebar.success("✅ FAQ file found and loaded.")
else:
    st.sidebar.warning("⚠️ No FAQ file found. Please upload one.")
    uploaded = st.sidebar.file_uploader("Upload shopunow_faqs.jsonl", type="jsonl")
    if uploaded:
        with open(faq_path,"wb") as f: f.write(uploaded.read())
        st.sidebar.success("✅ Uploaded! Refresh the page.")
        st.stop()

# Run built-in test queries
if st.sidebar.button("🧪 Run Built-in Tests"):
    st.sidebar.info("Running test queries...")
    test_queries = [
        "What are your support hours?",
        "Tell me order status for order id ORD-1234",
        "I want a return because the product is wrong",
        "My password reset isn't working, this is frustrating",
        "How do I pay with UPI?",
        "How to apply for leaves?",
        "What is the leave policy?",
    ]
    for tq in test_queries:
        st.sidebar.write(f"**🧑 {tq}**")
        try:
            ans = ask(tq)
            st.sidebar.success(f"🤖 {ans}")
        except Exception as e:
            st.sidebar.error(f"❌ {e}")

st.divider()
st.subheader("💬 Chat with Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("Ask something:")

if st.button("Ask") and query.strip():
    st.session_state.chat.append(("🧑 You", query))
    try:
        ans = ask(query)
    except Exception as e:
        ans=f"❌ Error: {e}"
    st.session_state.chat.append(("🤖 Agent", ans))

for s,m in st.session_state.chat:
    st.markdown(f"**{s}:** {m}")

if st.button("Clear Chat"):
    st.session_state.chat=[]
    st.rerun()
