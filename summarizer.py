"""
summarizer.py
--------------
Generates a conversation summary from WhatsApp chat data.

Two modes:
  1. Frequency-based (offline, no API key needed)  ← default
  2. LLM-based via Anthropic API (set USE_LLM = True and provide ANTHROPIC_API_KEY in secrets)

Drop this file into your project root and import it in app.py.
"""

import re
import math
import os
from collections import Counter, defaultdict

import pandas as pd

# Toggle this to True once you have an Anthropic API key in Streamlit secrets
USE_LLM = False  # Set to True to enable LLM summarization


# ─── Stop Words ──────────────────────────────────────────────────────────────

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","it","i","you","we","they","he","she","that","this","was","are",
    "be","been","have","has","had","do","does","did","not","so","if","as",
    "my","your","our","its","me","him","her","them","us","will","can","just",
    "by","from","up","out","into","about","than","more","also","some","what",
    "ok","okay","yes","no","hi","hey","lol","haha","😂","👍","🙏","na","hai",
    "bhi","aur","nahi","hoga","kar","karo","kya","toh","ko","se","ka","ki",
    "ke","ho","ye","vo","ab","ne","tha","thi","the","hain","hoon","woh","yeh",
    "mujhe","tumhe","aap","main","hum","kuch","ek","nhi","kr","h","k","m",
}


# ─── Frequency-based Summarization ───────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower()


def _score_sentences(messages: list[str], top_n_words: int = 15) -> list[tuple[str, float]]:
    """Score each message by the frequency of its non-stop words."""
    word_freq: Counter = Counter()
    for msg in messages:
        for word in _clean(msg).split():
            if word not in STOP_WORDS and len(word) > 2:
                word_freq[word] += 1

    total = sum(word_freq.values()) or 1
    top_words = {w: c / total for w, c in word_freq.most_common(top_n_words)}

    scored = []
    for msg in messages:
        if len(msg) < 10:
            continue
        words = _clean(msg).split()
        score = sum(top_words.get(w, 0) for w in words)
        if score > 0:
            scored.append((msg, score))

    return scored


def frequency_summary(df: pd.DataFrame, top_sentences: int = 8) -> dict:
    """
    Returns a dict with:
      summary_sentences, top_keywords, active_hours, peak_date, user_stats
    """
    messages = [
        m for m in df["message"].tolist()
        if m not in ("<Media omitted>", "This message was deleted", "") and len(str(m)) > 5
    ]

    scored = _score_sentences(messages)
    scored.sort(key=lambda x: x[1], reverse=True)
    summary_sentences = [s for s, _ in scored[:top_sentences]]

    # Top keywords
    word_freq: Counter = Counter()
    for msg in messages:
        for w in _clean(msg).split():
            if w not in STOP_WORDS and len(w) > 3:
                word_freq[w] += 1
    top_keywords = [w for w, _ in word_freq.most_common(15)]

    # Time analysis
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"])

    active_hours = {}
    if not df2.empty and hasattr(df2["date"].dt, "hour"):
        active_hours = df2.groupby(df2["date"].dt.hour).size().to_dict()

    peak_date = None
    if not df2.empty:
        daily = df2.groupby(df2["date"].dt.date).size()
        if not daily.empty:
            peak_date = str(daily.idxmax())

    # Per-user stats
    user_stats = (
        df.groupby("user")["message"]
        .count()
        .reset_index(name="messages")
        .sort_values("messages", ascending=False)
        .head(5)
        .to_dict("records")
    )

    return {
        "summary_sentences": summary_sentences,
        "top_keywords":      top_keywords,
        "active_hours":      active_hours,
        "peak_date":         peak_date,
        "user_stats":        user_stats,
        "total_messages":    len(messages),
    }


# ─── LLM-based Summarization ─────────────────────────────────────────────────

def llm_summary(df: pd.DataFrame, max_messages: int = 200) -> str:
    """
    Uses Anthropic claude-sonnet-4-20250514 to generate a natural language summary.
    Requires:
      - pip install anthropic
      - ANTHROPIC_API_KEY in Streamlit secrets OR environment variable
    """
    try:
        import anthropic
        import streamlit as st

        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "⚠️ API key not found. Add ANTHROPIC_API_KEY to Streamlit secrets."

        sample = df[~df["message"].isin(["<Media omitted>", "This message was deleted"])]
        sample = sample.tail(max_messages)

        chat_text = "\n".join(
            f"{row['user']}: {row['message']}"
            for _, row in sample.iterrows()
        )

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the following WhatsApp chat in 5-8 bullet points. "
                        "Highlight the main topics discussed, any decisions made, "
                        "and any pending tasks or commitments. Be concise.\n\n"
                        f"{chat_text}"
                    ),
                }
            ],
        )
        return message.content[0].text

    except Exception as e:
        return f"⚠️ LLM summarization failed: {e}"


# ─── Streamlit Section ────────────────────────────────────────────────────────

def render_summary_section(df: pd.DataFrame):
    """Call this from app.py to render the Summary UI block."""
    import streamlit as st

    st.markdown("## Conversation Summary")
    st.caption("Auto-generated overview of your chat.")

    mode = "AI Summary (LLM)" if USE_LLM else "📊 Keyword-based Summary"
    st.markdown(f"**Mode:** {mode}")

    with st.spinner("Analyzing chat…"):
        stats = frequency_summary(df)

    # Top-level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", stats["total_messages"])
    col2.metric("Peak Day", stats["peak_date"] or "—")
    col3.metric("Top Contributors", len(stats["user_stats"]))

    # Most active hours
    if stats["active_hours"]:
        hours_df = pd.DataFrame(
            list(stats["active_hours"].items()), columns=["Hour", "Messages"]
        ).sort_values("Hour")
        with st.expander("Most Active Hours", expanded=False):
            st.bar_chart(hours_df.set_index("Hour"))

    # Top keywords
    if stats["top_keywords"]:
        st.markdown("#### Hot Topics & Keywords")
        kw_html = " ".join(
            f'<span style="background:#1C2333;color:#25D366;border:1px solid #25D366;'
            f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;'
            f'font-size:0.85rem;font-weight:600;">'
            f'{w}</span>'
            for w in stats["top_keywords"]
        )
        st.markdown(kw_html, unsafe_allow_html=True)

    # Summary sentences
    st.markdown("#### Key Highlights")
    if USE_LLM:
        with st.spinner("Generating AI summary…"):
            llm_text = llm_summary(df)
        st.markdown(llm_text)
    else:
        if stats["summary_sentences"]:
            for i, sentence in enumerate(stats["summary_sentences"], 1):
                st.markdown(
                    f"""<div style="background:#1C2333;border-left:4px solid #6366f1;
                    padding:10px 16px;border-radius:8px;margin-bottom:8px;">
                    <span style="color:#818CF8;font-weight:700;">#{i}</span>
                    <span style="color:#E6EDF3;"> {sentence}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Not enough text content to extract key highlights.")

    # Per-user leaderboard
    if stats["user_stats"]:
        st.markdown("#### 🏆 Top Contributors")
        for rank, u in enumerate(stats["user_stats"], 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank - 1]
            st.markdown(f"{medal} **{u['user']}** — {u['messages']} messages")
