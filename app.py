import streamlit as st
import matplotlib.pyplot as plt
import preprocessor, helper
import seaborn as sns

# ── Level 1 imports ──────────────────────────────────────────────────────────
from task_detector   import render_task_section
from link_extractor  import render_links_section
from summarizer      import render_summary_section

# ── Level 2 imports ──────────────────────────────────────────────────────────
from reminder_generator    import render_reminders_section
from response_intelligence import render_response_section

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="📱",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background-color: #0E1117; color: #E6EDF3; }
section[data-testid="stSidebar"] { background-color: #161B22; padding: 20px; }
h1 { font-size: 2.4rem; margin-bottom: 0.3rem; }
h2 { margin-top: 1.5rem; }
h1, h2, h3 { color: #E6EDF3; font-weight: 600; }
.stButton>button {
    background-color: #25D366; color: #000; border-radius: 10px;
    padding: 10px 18px; font-weight: 600; border: none; width: 100%;
}
.stButton>button:hover { background-color: #1EBE5D; }
[data-testid="stFileUploader"] { border: 2px dashed #30363D; border-radius: 12px; padding: 16px; }
[data-testid="stMetric"] { background-color: #161B22; padding: 15px; border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
hr { border: 0.5px solid #30363D; }
.sidebar-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
    color: #6E7681; text-transform: uppercase; margin: 14px 0 4px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center;">📱 WhatsApp Chat Analyzer</h1>
<p style="text-align:center; color:#8B949E;">
Analyze WhatsApp chats with interactive insights · Smart Edition v2.0
</p>
<hr>
""", unsafe_allow_html=True)

from pathlib import Path
BASE_DIR  = Path(__file__).parent
logo_path = BASE_DIR / "whatsapp_logo.jpg"

if logo_path.exists():
    st.sidebar.image(str(logo_path))

st.sidebar.title("See More Than Just Messages")

uploaded_file = st.file_uploader("📂 Upload your WhatsApp chat export (.txt)", type=["txt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data       = bytes_data.decode("utf-8", errors="ignore")
    df         = preprocessor.preprocess(data)

    user_list = sorted(df["user"].unique().tolist())
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # ── Sidebar Navigation ────────────────────────────────────────────────────
    st.sidebar.markdown('<p class="sidebar-label">📊 Core Analysis</p>', unsafe_allow_html=True)
    CORE = [
        "📊 Top Statistics",
        "📅 Monthly Timeline",
        "📆 Daily Timeline",
        "🗺️ Activity Map",
        "👥 Most Busy Users",
        "☁️ WordCloud",
        "🔤 Most Common Words",
        "😀 Emoji Analysis",
    ]
    st.sidebar.markdown('<p class="sidebar-label">🧠 Smart Features — Level 1</p>', unsafe_allow_html=True)
    L1 = ["📋 Pending Tasks", "🔗 Important Links", "💬 Conversation Summary"]

    st.sidebar.markdown('<p class="sidebar-label">🚀 Assistant Features — Level 2</p>', unsafe_allow_html=True)
    L2 = ["📅 Reminder Generator", "⏱️ Response Intelligence"]

    selected_section = st.sidebar.radio("", CORE + L1 + L2, label_visibility="collapsed")

    if st.sidebar.button("Show Analysis"):
        st.session_state["analysis_ready"] = True

    if not st.session_state.get("analysis_ready"):
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#6E7681;">
        <div style="font-size:3rem;">📊</div>
        <div style="font-size:1.1rem;margin-top:10px;">
        Select a user and click <b style="color:#25D366;">Show Analysis</b> to begin
        </div>
        <div style="font-size:0.85rem;margin-top:8px;color:#30363D;">
        WhatsApp → Chat → ⋮ → Export Chat → Without Media
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    target_df = df if selected_user == "Overall" else df[df["user"] == selected_user]

    # ── matplotlib dark style helper ─────────────────────────────────────────
    def dark_fig():
        fig, ax = plt.subplots(facecolor="#0E1117")
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        return fig, ax

    # ═════════════════════════════════════════════════════════════════════════
    # CORE SECTIONS
    # ═════════════════════════════════════════════════════════════════════════

    if selected_section == "📊 Top Statistics":
        num_messages, words, media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Messages", num_messages)
        c2.metric("Total Words", words)
        c3.metric("Media Shared", media_messages)
        c4.metric("Links Shared", num_links)
        st.markdown("---")
        st.dataframe(df, use_container_width=True)

    elif selected_section == "📅 Monthly Timeline":
        st.header("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = dark_fig()
        ax.plot(timeline["time"], timeline["message"], color="#25D366", linewidth=2)
        plt.xticks(rotation="vertical", color="#8B949E")
        st.pyplot(fig)

    elif selected_section == "📆 Daily Timeline":
        st.header("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = dark_fig()
        ax.plot(daily_timeline["only_date"], daily_timeline["message"], color="#58A6FF", linewidth=1.5)
        plt.xticks(rotation="vertical", color="#8B949E")
        st.pyplot(fig)

    elif selected_section == "🗺️ Activity Map":
        st.header("Activity Map")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = dark_fig()
            ax.bar(busy_day.index, busy_day.values, color="#8B5CF6")
            plt.xticks(rotation="vertical", color="#8B949E")
            st.pyplot(fig)
        with c2:
            st.subheader("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = dark_fig()
            ax.bar(busy_month.index, busy_month.values, color="#F97316")
            plt.xticks(rotation="vertical", color="#8B949E")
            st.pyplot(fig)

        st.header("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if user_heatmap is None:
            st.warning("Not enough data to generate Weekly Activity Map.")
        else:
            fig, ax = plt.subplots(facecolor="#0E1117")
            sns.heatmap(user_heatmap, ax=ax, cmap="Greens")
            st.pyplot(fig)

    elif selected_section == "👥 Most Busy Users":
        if selected_user == "Overall":
            st.header("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = dark_fig()
                ax.bar(x.index, x.values, color="#25D366")
                plt.xticks(rotation="vertical", color="#8B949E")
                st.pyplot(fig)
            with c2:
                st.dataframe(new_df, use_container_width=True)
        else:
            st.info("'Most Busy Users' is only available when **Overall** is selected.")

    elif selected_section == "☁️ WordCloud":
        st.header("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots(facecolor="#0E1117")
        ax.imshow(df_wc)
        ax.axis("off")
        st.pyplot(fig)

    elif selected_section == "🔤 Most Common Words":
        st.header("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = dark_fig()
        ax.barh(most_common_df[0], most_common_df[1], color="#58A6FF")
        st.pyplot(fig)

    elif selected_section == "😀 Emoji Analysis":
        emoji_df = helper.emoji_helper(selected_user, df)
        st.header("Emoji Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(emoji_df, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(facecolor="#0E1117")
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f",
                   textprops={"color": "#E6EDF3"})
            st.pyplot(fig)

    # ═════════════════════════════════════════════════════════════════════════
    # LEVEL 1
    # ═════════════════════════════════════════════════════════════════════════

    elif selected_section == "📋 Pending Tasks":
        render_task_section(target_df)

    elif selected_section == "🔗 Important Links":
        render_links_section(target_df)

    elif selected_section == "💬 Conversation Summary":
        render_summary_section(target_df)

    # ═════════════════════════════════════════════════════════════════════════
    # LEVEL 2
    # ═════════════════════════════════════════════════════════════════════════

    elif selected_section == "📅 Reminder Generator":
        render_reminders_section(target_df)

    elif selected_section == "⏱️ Response Intelligence":
        render_response_section(target_df)

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#6E7681;">
    <div style="font-size:4rem;">💬</div>
    <h2 style="color:#E6EDF3;margin-top:16px;">Upload your WhatsApp Chat</h2>
    <p style="max-width:480px;margin:12px auto;line-height:1.6;">
    Open WhatsApp → Any Chat → ⋮ Menu → Export Chat → Without Media<br>
    Then upload the <b>.txt</b> file above.
    </p>
    <hr style="max-width:300px;margin:24px auto;border-color:#30363D;">
    <p style="font-size:0.8rem;">✅ 100% local · 🔒 Your data never leaves your device</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr>
<p style="text-align:center; color:#6E7681;">
From Messages to Meaning · WhatsApp Chat Analyzer v2.0
</p>
""", unsafe_allow_html=True)
