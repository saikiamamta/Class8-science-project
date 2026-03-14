"""
╔══════════════════════════════════════════════════════════════════════╗
║          LearnIQ — AI Tutor for CBSE Grade 8 Science                ║
║          Built with LangChain + ChromaDB + GPT-4o-mini              ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. pip install streamlit langchain langchain-openai langchain-community
               chromadb pypdf tiktoken
  2. Set your OpenAI key (see Section 0 below for OS-specific notes)
  3. Place your NCERT_Class8_Science.pdf in the same folder as this file
  4. streamlit run learniq_app.py

🔁  CHECKPOINT — SWAP SUBJECT/GRADE (1 variable):
     Line ~60:  PDF_PATH = "NCERT_Class8_Science.pdf"
     # REMOVE this line:
PDF_PATH = "NCERT_Class8_Science.pdf"

# REPLACE with this:
PDF_PATHS = [
    "hecu1cc.pdf", "hecu1ps.pdf", "hecu101.pdf", "hecu102.pdf",
    "hecu103.pdf", "hecu104.pdf", "hecu105.pdf", "hecu106.pdf",
    "hecu107.pdf", "hecu108.pdf", "hecu109.pdf", "hecu110.pdf",
    "hecu111.pdf", "hecu112.pdf", "hecu113.pdf",
]
     Change to any other PDF (e.g., "NCERT_Class9_Physics.pdf")
     and update APP_TITLE if you like.
"""

# ─────────────────────────────────────────────────────────────────────
# SECTION 0 — API KEY SETUP
# ─────────────────────────────────────────────────────────────────────
# ⚑ CHECKPOINT — OS-SPECIFIC API KEY DIFFERENCES:
#
#   macOS / Linux (terminal):
#       export OPENAI_API_KEY="sk-..."
#
#   Windows CMD:
#       set OPENAI_API_KEY=sk-...
#
#   Windows PowerShell:
#       $env:OPENAI_API_KEY="sk-..."
#
#   Alternative (NOT for production — hardcoded keys leak easily):
#       os.environ["OPENAI_API_KEY"] = "sk-..."   ← uncomment line below
#
#   Best practice for Streamlit Cloud:
#       Add key to .streamlit/secrets.toml as:
#           OPENAI_API_KEY = "sk-..."
#       Then read with: os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# ─────────────────────────────────────────────────────────────────────

import os
import streamlit as st
from pathlib import Path

# Uncomment ONLY if you cannot set env var externally (not recommended):
# os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"

# If deploying to Streamlit Cloud, uncomment the two lines below:
# if "OPENAI_API_KEY" in st.secrets:
#     os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

# 🔁 CHECKPOINT — SWAP SUBJECT/GRADE: Change this ONE variable
PDF_PATH = "NCERT_Class8_Science.pdf"          # ← change for different subject

APP_TITLE       = "LearnIQ — CBSE Grade 8 Science Tutor"
CHROMA_DIR      = "./chroma_db"                # persisted vector store on disk
CHUNK_SIZE      = 500                          # tokens per chunk
CHUNK_OVERLAP   = 50                           # overlap between chunks
NUM_CHUNKS      = 4                            # source chunks fetched per query
EMBED_MODEL     = "text-embedding-3-small"     # cheapest OpenAI embedding
LLM_MODEL       = "gpt-4o-mini"               # cost-controlled LLM


# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — SYSTEM PROMPT (Bloom's Taxonomy Scaffolding)
# ─────────────────────────────────────────────────────────────────────
# This prompt enforces:
#   (a) Answers ONLY from retrieved context (hallucination guard)
#   (b) Bloom's competency ladder: Remember → Understand → Apply →
#       Analyse → Evaluate → Create
#   (c) Chapter source citation in every reply

SYSTEM_PROMPT_TEMPLATE = """
You are LearnIQ, a friendly and encouraging AI tutor for CBSE Grade 8 Science.

STRICT RULE: Only answer from the retrieved context below.
If the answer is not found in the context, say:
"I could not find this in the textbook. Please check Chapter [X] or ask your teacher."
Never make up facts or use outside knowledge.

PEDAGOGY — COMPETENCY LADDER (Bloom's Taxonomy):
Always begin your answer at the student's current level and scaffold upward:
  Step 1 REMEMBER    — State the core fact simply (1–2 sentences).
  Step 2 UNDERSTAND  — Explain WHY or HOW in plain language with an analogy.
  Step 3 APPLY       — Give a real-life example or a solved mini-problem.
  Step 4 ANALYSE     — Break the concept into parts; compare/contrast where useful.
  Step 5 EVALUATE    — Pose a reflective question: "Which is better, and why?"
  Step 6 CREATE      — End with a small challenge: "Can you design / predict / invent…?"

Adapt depth to the question — a simple recall question needs Steps 1–2;
a complex question deserves all 6 steps across short paragraphs.
Keep language warm, age-appropriate (Grade 8), and avoid jargon.

CITATION RULE: End every answer with a line:
  📖 Source: [Chapter name, e.g., Chapter 3 — Synthetic Fibres and Plastics]

Retrieved context:
{context}

Student question: {question}

Your scaffolded answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — PDF LOADING AND CHUNKING
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="📚 Loading and indexing textbook — one-time setup...")
def build_retriever():
    """
    Loads the NCERT PDF, splits it into overlapping chunks,
    embeds them with text-embedding-3-small, and persists to ChromaDB.
    On subsequent runs the persisted DB is reloaded (no re-embedding).
    """

    # --- Check if vector store already persisted ---
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        # Reload from disk — no PDF re-processing, no embedding cost
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})

    # --- First run: load PDF ---
    raw_pages = []
for pdf in PDF_PATHS:
    if not Path(pdf).exists():
        st.warning(f"Skipping missing file: {pdf}")
        continue
    loader = PyPDFLoader(pdf)
    raw_pages.extend(loader.load())

if not raw_pages:
    st.error("No PDFs loaded. Check all files are in the project folder.")
    st.stop()                  # list of Document objects, one per page






    # --- Split into chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],   # prefer natural sentence breaks
    )
    chunks = splitter.split_documents(raw_pages)

    # --- Embed with text-embedding-3-small ---
    # Cost: ~$0.00002 per 1K tokens — a 280-page book ≈ ~70K tokens ≈ $0.0014 total
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # --- Store in ChromaDB, persisted to disk ---
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,          # survives app restarts
    )

    return vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — RetrievalQA CHAIN
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_chain(_retriever):
    """
    Wires the retriever to GPT-4o-mini via LangChain RetrievalQA.
    return_source_documents=True lets us display chapter badges.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,       # low temp = factual, less creative hallucination
        max_tokens=700,        # cap per reply for cost control
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                    # concatenates all chunks into one prompt
        retriever=_retriever,
        return_source_documents=True,          # we'll extract chapter info from these
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — CHAPTER BADGE HELPER
# ─────────────────────────────────────────────────────────────────────

def extract_chapter_badges(source_docs):
    """
    Parses source Document metadata to produce deduplicated chapter badges.
    PyPDFLoader stores page number in doc.metadata["page"] (0-indexed).
    We map rough page ranges to NCERT Class 8 chapter names.
    Adjust the page_ranges dict if your PDF edition differs.
    """
    # Approximate page ranges for NCERT Class 8 Science (adjust as needed)
    page_ranges = {
        (1,   18):  "Ch 1 — Crop Production and Management",
        (19,  38):  "Ch 2 — Microorganisms: Friend and Foe",
        (39,  54):  "Ch 3 — Synthetic Fibres and Plastics",
        (55,  70):  "Ch 4 — Materials: Metals and Non-Metals",
        (71,  90):  "Ch 5 — Coal and Petroleum",
        (91, 108):  "Ch 6 — Combustion and Flame",
        (109, 128): "Ch 7 — Conservation of Plants and Animals",
        (129, 148): "Ch 8 — Cell — Structure and Functions",
        (149, 170): "Ch 9 — Reproduction in Animals",
        (171, 190): "Ch 10 — Reaching the Age of Adolescence",
        (191, 210): "Ch 11 — Force and Pressure",
        (211, 230): "Ch 12 — Friction",
        (231, 248): "Ch 13 — Sound",
        (249, 268): "Ch 14 — Chemical Effects of Electric Current",
        (269, 285): "Ch 15 — Some Natural Phenomena",
    }

    badges = set()
    for doc in source_docs:
        page = doc.metadata.get("page", 0) + 1   # convert 0-index → 1-index
        for (start, end), chapter in page_ranges.items():
            if start <= page <= end:
                badges.add(chapter)
                break
        else:
            badges.add(f"Page {page}")            # fallback if page not mapped

    return sorted(badges)


# ─────────────────────────────────────────────────────────────────────
# SECTION 6 — STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="LearnIQ",
        page_icon="🧪",
        layout="centered",
    )

    # ── Custom CSS for clean, student-friendly design ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&display=swap');

        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

        .learniq-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .learniq-header h1 { color: #e94560; margin: 0; font-size: 2rem; font-weight: 900; }
        .learniq-header p  { color: #a8b2d8; margin: 0.3rem 0 0; font-size: 0.95rem; }

        .badge {
            display: inline-block;
            background: #e94560;
            color: white;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            margin: 3px 3px 0 0;
        }

        .source-row { margin-top: 0.5rem; }

        .bloom-tip {
            background: #0f3460;
            border-left: 4px solid #e94560;
            color: #a8b2d8;
            padding: 0.6rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.82rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="learniq-header">
        <h1>🧪 LearnIQ</h1>
        <p>Your CBSE Grade 8 Science Tutor · Powered by your textbook</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bloom-tip">
        💡 <strong>How I teach:</strong> I start with the basic fact, then explain why it works,
        give you a real-life example, and finally challenge you to think deeper —
        following Bloom's Taxonomy ladder!
    </div>
    """, unsafe_allow_html=True)

    # ── Build retriever & chain (cached after first load) ──
    retriever = build_retriever()
    chain     = build_chain(retriever)

    # ── Session state for chat history ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Display existing chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("badges"):
                badge_html = "".join(
                    f'<span class="badge">📖 {b}</span>' for b in msg["badges"]
                )
                st.markdown(
                    f'<div class="source-row">{badge_html}</div>',
                    unsafe_allow_html=True,
                )

    # ── Chat input ──
    user_input = st.chat_input("Ask me anything from your Science textbook…")

    if user_input:
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking through Bloom's ladder…"):
                result = chain.invoke({"query": user_input})

            answer       = result["result"]
            source_docs  = result.get("source_documents", [])
            badges       = extract_chapter_badges(source_docs)

            st.markdown(answer)

            # Show chapter source badges
            if badges:
                badge_html = "".join(
                    f'<span class="badge">📖 {b}</span>' for b in badges
                )
                st.markdown(
                    f'<div class="source-row">{badge_html}</div>',
                    unsafe_allow_html=True,
                )

        # Persist to session state
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "badges":  badges,
        })

    # ── Sidebar: debug + cost awareness ──
    with st.sidebar:
        st.markdown("### ⚙️ Session Info")
        st.caption(f"Model: `{LLM_MODEL}`")
        st.caption(f"Embeddings: `{EMBED_MODEL}`")
        st.caption(f"Chunks retrieved: `{NUM_CHUNKS}` per query")
        st.caption(f"Vector DB: ChromaDB at `{CHROMA_DIR}`")
        st.markdown("---")
        st.markdown("### 💰 Budget Tips")
        st.info(
            "Each question costs ≈ $0.0003–$0.0008.\n\n"
            "Embedding the PDF (one-time) costs ≈ $0.0014.\n\n"
            "~$5 budget supports **6,000–16,000 questions**."
        )
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════
# ⚑ CHECKPOINT — SWAP ChromaDB FOR PINECONE (2 lines to change)
#
# 1. Replace the Chroma import:
#    FROM:  from langchain_community.vectorstores import Chroma
#    TO:    from langchain_pinecone import PineconeVectorStore
#           import pinecone
#           pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-east-1")
#
# 2. Replace Chroma.from_documents(...) with:
#    vectorstore = PineconeVectorStore.from_documents(
#        documents=chunks,
#        embedding=embeddings,
#        index_name="learniq-cbse-grade8",   # create this index in Pinecone dashboard first
#    )
#
# Everything else (retriever, chain, UI) stays identical.
# ══════════════════════════════════════════════════════════════════════
