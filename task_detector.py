"""
task_detector.py
-----------------
Detects tasks, commitments, and scheduled events in WhatsApp chat messages.
Drop this file into your project root and import it in app.py.
"""

import re
import pandas as pd
from datetime import datetime


# ─── Keyword Rule Patterns ────────────────────────────────────────────────────

COMMITMENT_PATTERNS = [
    # Future promises / will-do
    r"\bI['']?ll\b.{0,60}",
    r"\bI will\b.{0,60}",
    r"\bI am going to\b.{0,60}",
    r"\bI'm going to\b.{0,60}",
    r"\bI would\b.{0,60}",
    r"\bwill send\b.{0,60}",
    r"\bwill share\b.{0,60}",
    r"\bwill call\b.{0,60}",
    r"\bwill check\b.{0,60}",
    r"\bwill do\b.{0,60}",
    r"\bwill get back\b.{0,60}",
    r"\bwill try\b.{0,60}",
    # Meeting / scheduling
    r"\blet['']?s meet\b.{0,60}",
    r"\bmeet (at|on|tomorrow|today)\b.{0,60}",
    r"\bcall (at|on|tomorrow|today)\b.{0,60}",
    r"\bscheduled? (for|at|on)\b.{0,60}",
    r"\breminder\b.{0,60}",
    r"\bdon['']?t forget\b.{0,60}",
    # Deadlines
    r"\bby (tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|eow)\b.{0,60}",
    r"\bdeadline\b.{0,60}",
    r"\bdue (by|on|tomorrow)\b.{0,60}",
    r"\bsubmit\b.{0,60}",
    # Sending / sharing
    r"\bsend(ing)? (you|the|it|this|them)\b.{0,60}",
    r"\bshare (the|it|this|link|file)\b.{0,60}",
    r"\bforward (the|it|this)\b.{0,60}",
]

TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}\s?(am|pm|AM|PM)?\b",
    r"\b(tomorrow|tonight|today|morning|evening|afternoon|night)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b\d{1,2}(st|nd|rd|th)?\s(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\bnext (week|month|monday|tuesday|wednesday|thursday|friday)\b",
]

COMPILED_COMMITMENTS = [re.compile(p, re.IGNORECASE) for p in COMMITMENT_PATTERNS]
COMPILED_TIME = [re.compile(p, re.IGNORECASE) for p in TIME_PATTERNS]


def _has_time_mention(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_TIME)


def _classify_task(text: str) -> str:
    """Rough category for a detected commitment."""
    t = text.lower()
    if any(k in t for k in ["meet", "call", "zoom", "sync", "catch up"]):
        return "📅 Meeting / Call"
    if any(k in t for k in ["send", "share", "forward", "upload", "attach"]):
        return "📤 Send / Share"
    if any(k in t for k in ["submit", "deadline", "due", "deliver"]):
        return "⏰ Deadline"
    if any(k in t for k in ["reminder", "don't forget", "dont forget", "remember"]):
        return "🔔 Reminder"
    return "✅ Task / Commitment"


def detect_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns  ['date', 'user', 'message']
         (typical output from your preprocessor)

    Returns
    -------
    DataFrame of detected tasks with columns:
        date, user, message, category, has_time
    """
    results = []
    for _, row in df.iterrows():
        msg = str(row.get("message", ""))
        if msg in ("<Media omitted>", "This message was deleted", ""):
            continue

        matched = any(p.search(msg) for p in COMPILED_COMMITMENTS)
        if matched:
            results.append({
                "date": row.get("date", ""),
                "user": row.get("user", ""),
                "message": msg,
                "category": _classify_task(msg),
                "has_time": _has_time_mention(msg),
            })

    return pd.DataFrame(results)


# ─── Streamlit Section ────────────────────────────────────────────────────────

def render_task_section(df: pd.DataFrame):
    """Call this from app.py to render the Tasks UI block."""
    import streamlit as st

    st.markdown("## 📋 Pending Tasks & Commitments")
    st.caption("Automatically detected from your chat using keyword rules.")

    tasks_df = detect_tasks(df)

    if tasks_df.empty:
        st.info("No tasks or commitments detected in this chat.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks Found", len(tasks_df))
    col2.metric("Tasks with Time Mention", tasks_df["has_time"].sum())
    col3.metric("Contributors", tasks_df["user"].nunique())

    # Category filter
    categories = ["All"] + sorted(tasks_df["category"].unique().tolist())
    selected = st.selectbox("Filter by Category", categories)

    filtered = tasks_df if selected == "All" else tasks_df[tasks_df["category"] == selected]

    # Per-user breakdown
    with st.expander("👤 Tasks by User", expanded=False):
        user_counts = tasks_df.groupby("user").size().reset_index(name="Tasks")
        st.dataframe(user_counts, use_container_width=True)

    # Task list
    st.markdown(f"### {selected} ({len(filtered)} items)")
    for _, row in filtered.iterrows():
        badge = "🕐" if row["has_time"] else ""
        with st.container():
            st.markdown(
                f"""<div style="background:#f0f9ff;border-left:4px solid #0ea5e9;
                padding:10px 14px;border-radius:6px;margin-bottom:8px;">
                <b>{row['category']}</b> {badge}<br>
                <small style="color:#64748b;">{row['user']} &nbsp;·&nbsp; {str(row['date'])[:10]}</small><br>
                {row['message']}
                </div>""",
                unsafe_allow_html=True,
            )
