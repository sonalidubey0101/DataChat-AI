"""
response_intelligence.py
-------------------------
Analyzes reply speed, response patterns, and ghosting behavior in WhatsApp chats.

Works on group & personal chats.
No external dependencies beyond pandas & streamlit.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Constants ────────────────────────────────────────────────────────────────

# A message pair where the gap exceeds this is NOT considered a reply
# (conversation probably ended and restarted)
MAX_REPLY_GAP_HOURS = 12

# Gap > this = ghosted (no reply to a conversation starter)
GHOST_THRESHOLD_HOURS = 24

# Conversation break: silence longer than this starts a new "conversation thread"
CONVO_BREAK_MINUTES = 60


# ─── Core Analysis ────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and parse the dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[~df["message"].isin(["<Media omitted>", "This message was deleted"])]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _build_reply_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each message, find the next message from a DIFFERENT user within
    MAX_REPLY_GAP_HOURS. That pair = a reply.
    Returns DataFrame with: sender, replier, delay_seconds, delay_minutes
    """
    pairs = []
    for i in range(len(df) - 1):
        curr = df.iloc[i]
        nxt  = df.iloc[i + 1]

        if curr["user"] == nxt["user"]:
            continue  # same person, not a reply

        gap = (nxt["date"] - curr["date"]).total_seconds()
        if gap <= 0 or gap > MAX_REPLY_GAP_HOURS * 3600:
            continue

        pairs.append({
            "sender":        curr["user"],
            "replier":       nxt["user"],
            "sent_at":       curr["date"],
            "replied_at":    nxt["date"],
            "delay_seconds": gap,
            "delay_minutes": gap / 60,
        })

    return pd.DataFrame(pairs)


def _build_conversations(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split chat into conversation threads by silence gaps."""
    convos = []
    start = 0
    for i in range(1, len(df)):
        gap = (df.iloc[i]["date"] - df.iloc[i - 1]["date"]).total_seconds() / 60
        if gap > CONVO_BREAK_MINUTES:
            convos.append(df.iloc[start:i])
            start = i
    convos.append(df.iloc[start:])
    return convos


def _detect_ghosting(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Ghosting = a user sends a message that gets NO reply for > GHOST_THRESHOLD_HOURS,
    and it's the last message in that conversation thread.
    """
    convos = _build_conversations(df)
    ghosts = []

    for convo in convos:
        if len(convo) < 2:
            continue
        last_msg = convo.iloc[-1]
        # Check if anyone replied to the last message
        elapsed = (df["date"].max() - last_msg["date"]).total_seconds() / 3600
        if elapsed < GHOST_THRESHOLD_HOURS:
            continue  # too recent, might still get a reply

        # Was last message left on read?
        last_sender = last_msg["user"]
        last_time   = last_msg["date"]

        # Check if there's NO reply from someone else after this
        after = df[(df["date"] > last_time)]
        if after.empty or (after["user"] == last_sender).all():
            ghosts.append({
                "ghosted_by":    last_sender,          # who was left hanging
                "last_message":  last_msg["message"][:100],
                "sent_at":       last_time,
                "hours_waiting": round(elapsed, 1),
            })

    return pd.DataFrame(ghosts)


def analyze_responses(df: pd.DataFrame) -> dict:
    """
    Master analysis function.

    Returns dict with:
      reply_pairs, avg_delay, fastest_replier, slowest_replier,
      user_stats, ghosting_df, hourly_activity
    """
    df = _prepare(df)
    pairs = _build_reply_pairs(df)
    ghosts = _detect_ghosting(df, pairs)

    user_stats = {}
    if not pairs.empty:
        for user in df["user"].unique():
            user_replies = pairs[pairs["replier"] == user]
            user_stats[user] = {
                "avg_reply_min":    round(user_replies["delay_minutes"].mean(), 1) if not user_replies.empty else None,
                "median_reply_min": round(user_replies["delay_minutes"].median(), 1) if not user_replies.empty else None,
                "fastest_reply_s":  round(user_replies["delay_seconds"].min(), 0) if not user_replies.empty else None,
                "reply_count":      len(user_replies),
                "messages_sent":    len(df[df["user"] == user]),
            }

        avg_by_user = {
            u: s["avg_reply_min"]
            for u, s in user_stats.items()
            if s["avg_reply_min"] is not None
        }
        fastest_replier = min(avg_by_user, key=avg_by_user.get) if avg_by_user else None
        slowest_replier = max(avg_by_user, key=avg_by_user.get) if avg_by_user else None
    else:
        fastest_replier = slowest_replier = None

    # Hourly activity
    df["hour"] = df["date"].dt.hour
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

    return {
        "reply_pairs":     pairs,
        "user_stats":      user_stats,
        "fastest_replier": fastest_replier,
        "slowest_replier": slowest_replier,
        "ghosting_df":     ghosts,
        "hourly_activity": hourly,
        "total_messages":  len(df),
        "total_users":     df["user"].nunique(),
    }


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def _fmt_time(minutes) -> str:
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return "—"
    minutes = float(minutes)
    if minutes < 1:
        return f"{int(minutes * 60)}s"
    if minutes < 60:
        return f"{int(minutes)}m"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m"


def _ghost_label(hours: float) -> str:
    if hours < 48:   return f"👻 {hours:.0f}h silence"
    if hours < 168:  return f"💀 {hours/24:.0f} days"
    return f"☠️ {hours/168:.1f} weeks"


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def render_response_section(df: pd.DataFrame):
    st.markdown("## ⏱️ Response Intelligence")
    st.caption("Who replies fastest? Who ghosts? Deep reply-pattern analysis.")

    if df["user"].nunique() < 2:
        st.warning("Response Intelligence needs at least 2 participants.")
        return

    with st.spinner("Analyzing reply patterns…"):
        data = analyze_responses(df)

    pairs      = data["reply_pairs"]
    user_stats = data["user_stats"]
    ghosts     = data["ghosting_df"]

    if pairs.empty:
        st.warning("Not enough back-and-forth messages found to analyze response times.")
        return

    # ── Hero metrics ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reply Pairs", len(pairs))
    col2.metric("⚡ Fastest Replier", data["fastest_replier"] or "—")
    col3.metric("🐢 Slowest Replier", data["slowest_replier"] or "—")
    col4.metric("👻 Ghost Instances", len(ghosts))

    # ── Per-user reply card ──────────────────────────────────────────────────
    st.markdown("### 👤 Reply Speed by User")

    sorted_users = sorted(
        user_stats.items(),
        key=lambda x: (x[1]["avg_reply_min"] or 9999),
    )

    cols = st.columns(min(len(sorted_users), 3))
    for i, (user, stats) in enumerate(sorted_users):
        with cols[i % 3]:
            avg = stats["avg_reply_min"]
            med = stats["median_reply_min"]
            fastest = stats["fastest_reply_s"]
            replies = stats["reply_count"]

            # Emoji rank
            if i == 0:   rank = "⚡ Fastest"
            elif i == len(sorted_users) - 1: rank = "🐢 Slowest"
            else:         rank = f"#{i+1}"

            st.markdown(
                f"""<div style="background:#161B22;border:1px solid #30363D;
                border-radius:12px;padding:14px 16px;margin-bottom:10px;text-align:center;">
                <div style="font-size:0.75rem;color:#25D366;font-weight:700;
                letter-spacing:1px;">{rank}</div>
                <div style="font-size:1.1rem;font-weight:600;margin:4px 0;">{user}</div>
                <div style="font-size:1.6rem;font-weight:700;color:#E6EDF3;">
                {_fmt_time(avg)}</div>
                <div style="font-size:0.72rem;color:#6E7681;">avg reply time</div>
                <hr style="border-color:#30363D;margin:8px 0;">
                <div style="font-size:0.78rem;color:#8B949E;">
                Median: {_fmt_time(med)}<br>
                Fastest: {_fmt_time((fastest or 0)/60)}<br>
                Replies sent: {replies}
                </div></div>""",
                unsafe_allow_html=True,
            )

    # ── Reply delay distribution ─────────────────────────────────────────────
    with st.expander("📊 Reply Delay Distribution", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0E1117")
        ax.set_facecolor("#161B22")
        capped = pairs["delay_minutes"].clip(upper=120)
        ax.hist(capped, bins=40, color="#25D366", alpha=0.8, edgecolor="#0E1117")
        ax.set_xlabel("Reply time (minutes, capped at 2h)", color="#8B949E")
        ax.set_ylabel("Count", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        st.pyplot(fig)

    # ── Who replies to whom ──────────────────────────────────────────────────
    with st.expander("🔄 Who Replies to Whom?", expanded=False):
        matrix = pairs.groupby(["replier", "sender"])["delay_minutes"].mean().unstack(fill_value=0)
        matrix = matrix.round(1)
        st.markdown("**Average reply time (minutes)** — rows = replier, cols = who they replied to")
        st.dataframe(matrix.style.background_gradient(cmap="Greens"), use_container_width=True)

    # ── Ghosting section ─────────────────────────────────────────────────────
    st.markdown("### 👻 Ghosting Detector")
    st.caption(f"Messages left unanswered for >{GHOST_THRESHOLD_HOURS}h in a dead conversation.")

    if ghosts.empty:
        st.success("No ghosting detected! Everyone replies. 🎉")
    else:
        ghost_counts = ghosts["ghosted_by"].value_counts().reset_index()
        ghost_counts.columns = ["User", "Times Ghosted"]

        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            st.markdown("**Ghost Leaderboard 💀**")
            for _, r in ghost_counts.iterrows():
                st.markdown(
                    f"<b>{r['User']}</b> — "
                    f"<span style='color:#F0A500;'>{r['Times Ghosted']}x left on read</span>",
                    unsafe_allow_html=True,
                )
        with col_g2:
            st.markdown("**Ghost Instances**")
            for _, r in ghosts.iterrows():
                st.markdown(
                    f"""<div style="background:#1A1A2E;border-left:4px solid #F0A500;
                    padding:8px 14px;border-radius:6px;margin-bottom:6px;font-size:0.85rem;">
                    <b>{r['ghosted_by']}</b> {_ghost_label(r['hours_waiting'])}<br>
                    <span style="color:#8B949E;">{str(r['sent_at'])[:16]}</span><br>
                    "{r['last_message']}"
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Hourly heatmap ───────────────────────────────────────────────────────
    with st.expander("🕐 When Does Everyone Reply?", expanded=False):
        hourly = data["hourly_activity"]
        fig, ax = plt.subplots(figsize=(10, 2), facecolor="#0E1117")
        ax.set_facecolor("#161B22")
        hours = list(range(24))
        values = [hourly.get(h, 0) for h in hours]
        bars = ax.bar(hours, values, color="#25D366", alpha=0.8)
        ax.set_xticks(hours)
        ax.set_xticklabels(
            [f"{h:02d}:00" if h % 3 == 0 else "" for h in hours],
            color="#8B949E", fontsize=7,
        )
        ax.set_ylabel("Messages", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        st.pyplot(fig)
