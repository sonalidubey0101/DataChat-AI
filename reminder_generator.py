"""
reminder_generator.py
----------------------
Detects dates, times, and event mentions in WhatsApp messages.
Converts them into a structured "Upcoming / Detected Events" list.

No external API needed — pure regex + dateutil parsing.
pip install python-dateutil  (add to requirements.txt)
"""

import re
import pandas as pd
from datetime import datetime, date, timedelta

try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


# ─── Patterns ────────────────────────────────────────────────────────────────

# Relative day words
RELATIVE_DAY = {
    "today":     0,
    "tonight":   0,
    "tomorrow":  1,
    "day after tomorrow": 2,
    "parso":     2,   # Hindi slang
    "kal":       1,
}

WEEKDAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

# Time pattern  e.g.  3pm  |  15:30  |  3:00 pm
TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)\b"
    r"|\b([01]?\d|2[0-3]):([0-5]\d)\b",
    re.IGNORECASE,
)

# Explicit date  e.g.  15 Jan  |  Jan 15  |  15/01  |  15-01-2025
DATE_RE = re.compile(
    r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b"
    r"|\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})\b",
    re.IGNORECASE,
)

# Event keywords that signal something scheduled
EVENT_KEYWORDS_RE = re.compile(
    r"\b(meet(ing)?|call|interview|appointment|class|lecture|exam|test|"
    r"deadline|submission|party|event|ceremony|wedding|birthday|anniversary|"
    r"flight|train|trip|travel|visit|presentation|demo|sync|standup|"
    r"seminar|workshop|conference|webinar|session|hackathon)\b",
    re.IGNORECASE,
)

# Commitment trigger words
TRIGGER_RE = re.compile(
    r"\b(remind|reminder|don'?t forget|remember|scheduled?|fixed|confirmed|"
    r"let'?s meet|meeting at|call at|join at|catch up|be there|come at|"
    r"we have|we need to|need to be|make sure)\b",
    re.IGNORECASE,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_time_str(text: str) -> str | None:
    m = TIME_RE.search(text)
    if not m:
        return None
    if m.group(1):  # 12-hr format
        h, mi, meridiem = m.group(1), m.group(2) or "00", m.group(3)
        return f"{h}:{mi} {meridiem.upper()}"
    else:           # 24-hr format
        return f"{m.group(4)}:{m.group(5)}"


def _extract_date(text: str, msg_date: date) -> date | None:
    """Try to find an absolute or relative date in the message."""
    lower = text.lower().strip()

    # Relative keywords
    for phrase, delta in RELATIVE_DAY.items():
        if phrase in lower:
            return msg_date + timedelta(days=delta)

    # Named weekday  →  next occurrence
    for i, day in enumerate(WEEKDAYS):
        if re.search(r"\b" + day + r"\b", lower):
            today_wd = msg_date.weekday()
            diff = (i - today_wd) % 7
            if diff == 0:
                diff = 7   # "this Monday" → next week
            return msg_date + timedelta(days=diff)

    # Explicit date via dateutil
    if DATEUTIL_AVAILABLE:
        m = DATE_RE.search(text)
        if m:
            try:
                parsed = dateutil_parser.parse(
                    m.group(0),
                    default=datetime(msg_date.year, msg_date.month, msg_date.day),
                    dayfirst=True,
                )
                return parsed.date()
            except Exception:
                pass

    return None


def _classify_event(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["birthday","bday","bday","janmdin"]):    return "🎂 Birthday"
    if any(k in t for k in ["wedding","shaadi","marriage"]):          return "💍 Wedding"
    if any(k in t for k in ["flight","train","bus","travel","trip"]): return "✈️ Travel"
    if any(k in t for k in ["exam","test","paper"]):                  return "📝 Exam"
    if any(k in t for k in ["interview"]):                            return "💼 Interview"
    if any(k in t for k in ["party","celebration","anniversary"]):    return "🎉 Celebration"
    if any(k in t for k in ["deadline","submit","submission","due"]):  return "⏰ Deadline"
    if any(k in t for k in ["call","zoom","meet"]):                   return "📞 Call / Meet"
    if any(k in t for k in ["class","lecture","session","seminar"]):  return "📚 Class / Session"
    return "📅 Event"


# ─── Main Function ────────────────────────────────────────────────────────────

def detect_reminders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans messages for date/time references with event context.

    Parameters
    ----------
    df : DataFrame with columns ['date', 'user', 'message']

    Returns
    -------
    DataFrame with columns:
        msg_date, user, message, event_type, event_date, event_time, source_date
    """
    results = []
    for _, row in df.iterrows():
        msg = str(row.get("message", "")).strip()
        if msg in ("<Media omitted>", "This message was deleted", "") or len(msg) < 5:
            continue

        # Must have at least a time OR a date reference  AND  a trigger/event keyword
        has_time    = bool(TIME_RE.search(msg))
        has_event   = bool(EVENT_KEYWORDS_RE.search(msg))
        has_trigger = bool(TRIGGER_RE.search(msg))

        if not (has_time or has_event or has_trigger):
            continue

        # Parse message date
        try:
            msg_date = pd.to_datetime(row.get("date")).date()
        except Exception:
            msg_date = date.today()

        event_date = _extract_date(msg, msg_date)
        event_time = _extract_time_str(msg)

        # Skip if we couldn't pin any date or time
        if event_date is None and event_time is None:
            continue

        results.append({
            "source_date": msg_date,
            "user":        row.get("user", ""),
            "message":     msg,
            "event_type":  _classify_event(msg),
            "event_date":  event_date,
            "event_time":  event_time or "—",
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("event_date", na_position="last")
    return result_df


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def render_reminders_section(df: pd.DataFrame):
    import streamlit as st

    st.markdown("## 📅 Reminder Generator")
    st.caption("Detected scheduled events, meetings, and deadlines from your chat.")

    with st.spinner("Scanning for dates and events…"):
        reminders = detect_reminders(df)

    if reminders.empty:
        st.info("No scheduled events detected in this chat.")
        return

    # Summary metrics
    today = date.today()
    future = reminders[reminders["event_date"].apply(
        lambda d: d >= today if pd.notna(d) and d is not None else False
    )]
    past = reminders[reminders["event_date"].apply(
        lambda d: d < today if pd.notna(d) and d is not None else False
    )]
    no_date = reminders[reminders["event_date"].isna()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events Found", len(reminders))
    col2.metric("🔜 Upcoming", len(future))
    col3.metric("✅ Past / Done", len(past))

    # Event type filter
    types = ["All"] + sorted(reminders["event_type"].unique().tolist())
    sel_type = st.selectbox("Filter by Event Type", types, key="reminder_type_filter")

    def filtered(frame):
        return frame if sel_type == "All" else frame[frame["event_type"] == sel_type]

    def render_card(row, accent):
        st.markdown(
            f"""<div style="background:#161B22;border-left:4px solid {accent};
            padding:10px 16px;border-radius:8px;margin-bottom:8px;">
            <b>{row['event_type']}</b><br>
            <span style="color:#94a3b8;font-size:0.8rem;">
            📆 {str(row['event_date']) if pd.notna(row['event_date']) and row['event_date'] else '—'} &nbsp;
            🕐 {row['event_time']} &nbsp;|&nbsp; 👤 {row['user']}
            </span><br>
            <span style="font-size:0.9rem;">{row['message']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    if not filtered(future).empty:
        st.markdown("### 🔜 Upcoming Events")
        for _, r in filtered(future).iterrows():
            render_card(r, "#25D366")

    if not filtered(past).empty:
        with st.expander(f"📁 Past Events ({len(filtered(past))})"):
            for _, r in filtered(past).iterrows():
                render_card(r, "#6E7681")

    if not filtered(no_date).empty:
        with st.expander(f"🕐 Time-only mentions ({len(filtered(no_date))})"):
            for _, r in filtered(no_date).iterrows():
                render_card(r, "#F0A500")

    # Calendar-style grouping
    if not future.empty:
        st.markdown("### 📆 Events by Date")
        for evt_date, group in future.groupby("event_date"):
            st.markdown(f"**{evt_date.strftime('%A, %d %B %Y')}**")
            for _, r in group.iterrows():
                st.markdown(
                    f"&nbsp;&nbsp;• {r['event_type']} &nbsp; `{r['event_time']}` — "
                    f"{r['message'][:80]}{'…' if len(r['message'])>80 else ''} "
                    f"<span style='color:#6E7681;font-size:0.75rem;'>({r['user']})</span>",
                    unsafe_allow_html=True,
                )
