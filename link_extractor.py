"""
link_extractor.py
------------------
Extracts, categorizes, and visualizes URLs shared in WhatsApp chats.
Drop this file into your project root and import it in app.py.
"""

import re
import pandas as pd
from urllib.parse import urlparse
from collections import Counter


# ─── URL Extraction ───────────────────────────────────────────────────────────

URL_REGEX = re.compile(
    r"https?://[^\s\)\]\>\"\']+",
    re.IGNORECASE,
)


def _extract_urls_from_text(text: str) -> list[str]:
    return URL_REGEX.findall(str(text))


# ─── Categorization ──────────────────────────────────────────────────────────

CATEGORIES = {
    "YouTube":      ["youtube.com", "youtu.be"],
    "Google Docs":  ["docs.google.com", "drive.google.com", "sheets.google.com", "slides.google.com"],
    "Instagram":    ["instagram.com", "instagr.am"],
    "Twitter/X":    ["twitter.com", "x.com", "t.co"],
    "Facebook":     ["facebook.com", "fb.com", "fb.me"],
    "WhatsApp":     ["chat.whatsapp.com", "wa.me"],
    "LinkedIn":     ["linkedin.com", "lnkd.in"],
    "Shopping":     ["amazon.com", "amazon.in", "flipkart.com", "myntra.com", "meesho.com", "ajio.com"],
    "News":         ["ndtv.com", "thehindu.com", "hindustantimes.com", "bbc.com", "cnn.com", "reuters.com", "timesofindia.com"],
    "GitHub":       ["github.com", "gist.github.com", "raw.githubusercontent.com"],
    "Music":        ["spotify.com", "music.apple.com", "gaana.com", "jiosaavn.com", "soundcloud.com"],
    "Apps / Stores":["play.google.com", "apps.apple.com"],
    "Cloud / Files":["dropbox.com", "wetransfer.com", "onedrive.com", "mediafire.com", "mega.nz"],
    "Notion / Tools":["notion.so", "trello.com", "asana.com", "miro.com", "figma.com"],
    "Health":       ["healthline.com", "webmd.com", "mayoclinic.org", "practo.com"],
    "Streaming":    ["netflix.com", "hotstar.com", "primevideo.com", "zee5.com", "sonyliv.com"],
    "Education":    ["coursera.org", "udemy.com", "edx.org", "khanacademy.org", "unacademy.com"],
    "Maps":         ["maps.google.com", "goo.gl/maps", "maps.app.goo.gl"],
}


def _categorize_url(url: str) -> str:
    domain = urlparse(url).netloc.lower().lstrip("www.")
    for category, domains in CATEGORIES.items():
        if any(d in domain for d in domains):
            return category
    return "Other"


def _shorten(url: str, max_len: int = 60) -> str:
    return url if len(url) <= max_len else url[:max_len] + "…"


# ─── Main Function ────────────────────────────────────────────────────────────

def extract_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns ['date', 'user', 'message']

    Returns
    -------
    DataFrame with columns: date, user, url, domain, category
    """
    rows = []
    for _, row in df.iterrows():
        msg = str(row.get("message", ""))
        for url in _extract_urls_from_text(msg):
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")
            rows.append({
                "date":     row.get("date", ""),
                "user":     row.get("user", ""),
                "url":      url,
                "domain":   domain,
                "category": _categorize_url(url),
            })
    return pd.DataFrame(rows)


# ─── Streamlit Section ────────────────────────────────────────────────────────

def render_links_section(df: pd.DataFrame):
    """Call this from app.py to render the Links UI block."""
    import streamlit as st

    st.markdown("## 🔗 Important Links Shared")
    st.caption("All URLs extracted and categorized from the chat.")

    links_df = extract_links(df)

    if links_df.empty:
        st.info("No links found in this chat.")
        return

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Links", len(links_df))
    col2.metric("Unique Domains", links_df["domain"].nunique())
    col3.metric("Categories", links_df["category"].nunique())

    # Category pie / bar
    cat_counts = links_df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.markdown("#### Links by Category")
        st.bar_chart(cat_counts.set_index("Category"))

    with col_table:
        st.markdown("#### Top Domains")
        top_domains = (
            links_df.groupby("domain")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(10)
        )
        st.dataframe(top_domains, use_container_width=True, hide_index=True)

    # Most active link-sharers
    with st.expander("👤 Who shared the most links?"):
        user_links = links_df.groupby("user").size().reset_index(name="Links Shared")
        st.dataframe(
            user_links.sort_values("Links Shared", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # Filtered link browser
    st.markdown("#### Browse Links")
    categories = ["All"] + sorted(links_df["category"].unique().tolist())
    selected_cat = st.selectbox("Filter by Category", categories, key="link_cat_filter")

    filtered = links_df if selected_cat == "All" else links_df[links_df["category"] == selected_cat]
    filtered = filtered.sort_values("date", ascending=False)

    for _, row in filtered.head(50).iterrows():
        st.markdown(
            f"""<div style="background:#1C2333;border:1px solid #30363D;
            padding:10px 16px;border-radius:8px;margin-bottom:8px;">
            <span style="color:#25D366;font-weight:700;">{row['category']}</span> &nbsp;
            <a href="{row['url']}" target="_blank" style="color:#58A6FF;word-break:break-all;font-size:0.9rem;">
            {_shorten(row['url'])}</a><br>
            <small style="color:#6E7681;">{row['user']} &nbsp;·&nbsp; {str(row['date'])[:10]}</small>
            </div>""",
            unsafe_allow_html=True,
        )

    if len(filtered) > 50:
        st.caption(f"Showing first 50 of {len(filtered)} links.")
