"""
Stock Market Prediction Dashboard — Streamlit frontend.

A thin client: every prediction comes from the FastAPI backend via
api/client.py. This file only renders UI and never re-implements feature
engineering or model inference.

Run:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api.client import APIClient, APIConnectionError, APIResponseError, APITimeoutError
from constants import (
    AUTHOR_EMAIL,
    COLORS,
    LABEL_COLORS,
    PROBABILITY_COLUMNS,
    PROJECT_AUTHOR,
    PROJECT_NAME,
    PROJECT_VERSION,
    REQUIRED_COLUMNS,
    SAMPLE_CSV,
)

from config import settings
from utils import (
    build_prediction_filename,
    compute_label_counts,
    dataframe_to_csv_bytes,
    format_response_time,
    format_timestamp,
    get_combined_predictions,
    get_latest_result,
    init_session_state,
    read_csv_preview,
    record_prediction_run,
    validate_dataframe_shape,
    validate_file_extension,
)

ASSETS_DIR = Path(__file__).parent / "assets"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ==========================================================================
# App shell
# ==========================================================================
def load_css() -> None:
    css_path = ASSETS_DIR / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> APIClient:
    """One APIClient per base_url, reused across reruns for connection pooling."""
    return APIClient(base_url)


def render_kpi_row(cards: list[dict]) -> None:
    """Render a row of KPI cards. Each dict needs label, value, accent."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left-color:
                {card.get('accent', COLORS['accent'])};">
                    <div class="kpi-label">{card['label']}</div>
                    <div class="kpi-value">{card['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar(api_client: APIClient) -> str:
    """Branding, editable API URL, and a live health badge."""
    with st.sidebar:
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=64)
        st.markdown(f"**{PROJECT_NAME}**  \n`v{PROJECT_VERSION}`")
        st.divider()

        api_base_url = st.text_input(
            "Backend API URL",
            value=st.session_state.get("api_base_url", settings.api_base_url),
            help="Where your FastAPI server (uvicorn main:app) is running.",
        ).rstrip("/")
        st.session_state["api_base_url"] = api_base_url

        health = api_client.check_health()
        if health.is_healthy:
            st.markdown(
                '<div class="status-badge status-ok">🟢 Backend online</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-bad">🔴 Backend unreachable</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Details"):
                st.caption(health.error or "Unknown error")

        st.divider()
        st.caption("Navigate using the tabs at the top of the page.")

    return api_base_url


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="app-footer">
            {PROJECT_NAME} · v{PROJECT_VERSION} · Built by {PROJECT_AUTHOR} ·
            <a href="mailto:{AUTHOR_EMAIL}" style="color: inherit;">{AUTHOR_EMAIL}</a>
            <br/>Frontend is a thin Streamlit client —
            all ML logic runs in the FastAPI backend.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Tabs
# ==========================================================================
def render_dashboard_tab(api_client: APIClient) -> None:
    combined_df = get_combined_predictions()
    label_counts = compute_label_counts(combined_df)
    latest = get_latest_result()
    health = api_client.check_health()

    render_kpi_row(
        [
            {
                "label": "Total Predictions",
                "value": f"{len(combined_df):,}",
                "accent": COLORS["accent"],
            },
            {
                "label": "Up Predictions",
                "value": f"{label_counts['Up']:,}",
                "accent": COLORS["up"],
            },
            {
                "label": "Down Predictions",
                "value": f"{label_counts['Down']:,}",
                "accent": COLORS["down"],
            },
            {
                "label": "Neutral Predictions",
                "value": f"{label_counts['Neutral']:,}",
                "accent": COLORS["neutral"],
            },
        ]
    )
    st.write("")

    last_run_time = format_timestamp(latest.timestamp) if latest else "No runs yet"

    render_kpi_row(
        [
            {
                "label": "Backend Health",
                "value": "Online" if health.is_healthy else "Offline",
                "accent": COLORS["up"] if health.is_healthy else COLORS["down"],
            },
            {
                "label": "Last Prediction Time",
                "value": last_run_time,
                "accent": COLORS["accent"],
            },
        ]
    )

    st.divider()
    if combined_df.empty:
        st.info(
            "No predictions yet this session. Use the **Prediction** tab"
            "to upload a CSV.",
            icon="👉",
        )
    else:
        st.subheader("Recent runs")
        history = st.session_state["prediction_history"]

        rows = [
            {
                "Timestamp": format_timestamp(e.timestamp),
                "File": e.filename,
                "Rows": e.rows,
            }
            for e in reversed(history)
        ]

        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_prediction_tab(api_client: APIClient) -> None:
    uploaded_file = st.file_uploader(
        "Upload OHLCV CSV file",
        type=["csv"],
        help="Required columns: " + ", ".join(REQUIRED_COLUMNS),
    )

    with st.expander("Need a template?"):
        st.download_button(
            "⬇️ Download sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_ohlcv.csv",
            mime="text/csv",
        )

    file_bytes, filename, is_valid = None, None, False

    if uploaded_file is not None:
        filename = uploaded_file.name
        if not validate_file_extension(filename):
            st.error("Only .csv files are supported.")
        else:
            file_bytes = uploaded_file.getvalue()
            try:
                preview_df = read_csv_preview(file_bytes)
                result = validate_dataframe_shape(preview_df)
                for w in result.warnings:
                    st.warning(w)
                for err in result.errors:
                    st.error(err)
                is_valid = result.is_valid
                if is_valid:
                    st.success(f"`{filename}` looks good — {len(preview_df)} rows.")
                with st.expander("Preview uploaded data", expanded=is_valid):
                    st.dataframe(preview_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Could not parse this file as CSV: {e}")

    run_clicked = st.button(
        "🚀 Run Prediction",
        type="primary",
        disabled=not is_valid,
        use_container_width=True,
    )

    if run_clicked and file_bytes is not None:
        with st.spinner("Sending file to the prediction API..."):
            try:
                response = api_client.predict_csv(filename, file_bytes)
                result_df = pd.DataFrame(response.predictions)
                if result_df.empty:
                    st.warning("The API returned no predictions for this file.")
                else:
                    record_prediction_run(result_df, filename)
                    st.success(
                        f"{response.message} "
                        f"({response.rows} rows in "
                        f"{format_response_time(response.response_time_ms)})"
                    )
            except APIConnectionError as e:
                st.error(f"🔌 {e}")
            except APITimeoutError as e:
                st.error(f"⏱️ {e}")
            except APIResponseError as e:
                st.error(f"⚠️ {e.detail}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.divider()
    latest = get_latest_result()
    if latest is None:
        st.caption("Upload a CSV and click **Run Prediction** to see results here.")
        return

    result_df: pd.DataFrame = latest.dataframe
    st.subheader("Results")
    tab_table, tab_charts = st.tabs(["📋 Table", "📈 Charts"])

    with tab_table:
        filtered_df = result_df
        if "Ticker" in result_df.columns:
            tickers = sorted(result_df["Ticker"].dropna().unique().tolist())
            selected = st.multiselect("Filter by ticker", tickers, default=tickers)
            filtered_df = (
                result_df[result_df["Ticker"].isin(selected)] if tickers else result_df
            )

        if "Prediction_Label" in filtered_df.columns:

            def _highlight(row):
                color = LABEL_COLORS.get(row.get("Prediction_Label", ""), "")
                return [
                    (
                        f"color: {color}; font-weight: 600;"
                        if c == "Prediction_Label"
                        else ""
                    )
                    for c in row.index
                ]

            st.dataframe(
                filtered_df.style.apply(_highlight, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download predictions as CSV",
            data=dataframe_to_csv_bytes(filtered_df),
            file_name=build_prediction_filename(),
            mime="text/csv",
        )

    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            _render_distribution_chart(result_df)
        with c2:
            _render_confidence_chart(result_df)
        _render_stock_wise_chart(result_df)
        _render_probability_trend_chart(result_df)


def render_api_status_tab(api_client: APIClient, api_base_url: str) -> None:
    if st.button("🔄 Refresh now", type="primary", use_container_width=True):
        st.rerun()

    health = api_client.check_health()
    render_kpi_row(
        [
            {
                "label": "Backend Status",
                "value": "Online" if health.is_healthy else "Offline",
                "accent": COLORS["up"] if health.is_healthy else COLORS["down"],
            },
            {"label": "API URL", "value": api_base_url, "accent": COLORS["info"]},
            {
                "label": "Response Time",
                "value": (
                    format_response_time(health.response_time_ms)
                    if health.is_healthy
                    else "—"
                ),
                "accent": COLORS["accent"],
            },
            {
                "label": "Last Health Check",
                "value": format_timestamp(health.checked_at),
                "accent": COLORS["neutral"],
            },
        ]
    )

    st.divider()
    if health.is_healthy:
        st.success(
            f"Backend responded normally with status `{health.status_text}`.", icon="✅"
        )
    else:
        st.error("Connection error detail:", icon="🛑")
        st.code(health.error or "Unknown error", language=None)
        st.markdown(
            "**Troubleshooting checklist**\n"
            "- Is the FastAPI server running? (`uvicorn main:app --reload`)\n"
            "- Does the API URL above match the host/port it's bound to?\n"
            "- If remote/Docker, is the port exposed and reachable from here?\n"
            "- Check the backend logs for a startup or model-loading failure."
        )


def render_about_tab() -> None:
    st.markdown(f"### {PROJECT_NAME}")
    st.markdown(
        "A stock market movement prediction service: a **FastAPI backend** "
        "(feature engineering + CatBoost inference) and this **Streamlit "
        "frontend** — a thin client that only renders what the API returns."
    )
    st.info(
        "All prediction logic lives in the FastAPI backend. The frontend "
        "never imports or reimplements any of it.",
        icon="🧩",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Version:** {PROJECT_VERSION}")
        st.markdown(f"- **Maintainer:** {PROJECT_AUTHOR}")
    with col2:
        st.markdown(f"- **Contact:** {AUTHOR_EMAIL}")
        st.markdown("- **API Docs:** append `/docs` to your backend URL")


# ==========================================================================
# Chart helpers (kept local — this is the "easy to understand" flat build)
# ==========================================================================
def _render_distribution_chart(df: pd.DataFrame) -> None:
    if df.empty or "Prediction_Label" not in df.columns:
        st.info("No label data available for this chart.")
        return
    counts = (
        df["Prediction_Label"].value_counts().reindex(LABEL_COLORS.keys(), fill_value=0)
    )
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[
                LABEL_COLORS.get(label, COLORS["accent"]) for label in counts.index
            ],
        )
    )
    fig.update_layout(title="Prediction Distribution", **_PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_chart(df: pd.DataFrame) -> None:
    available = [c for c in PROBABILITY_COLUMNS if c in df.columns]
    if df.empty or not available:
        st.info("No probability data available for this chart.")
        return
    confidence = df[available].max(axis=1)
    fig = px.histogram(
        confidence,
        nbins=20,
        labels={"value": "Confidence"},
        color_discrete_sequence=[COLORS["accent"]],
    )
    fig.update_layout(
        title="Confidence Distribution", showlegend=False, **_PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_stock_wise_chart(df: pd.DataFrame) -> None:
    if df.empty or "Ticker" not in df.columns or "Prediction_Label" not in df.columns:
        st.info("No per-ticker label data available for this chart.")
        return
    grouped = (
        df.groupby(["Ticker", "Prediction_Label"]).size().reset_index(name="Count")
    )
    fig = px.bar(
        grouped,
        x="Ticker",
        y="Count",
        color="Prediction_Label",
        barmode="stack",
        color_discrete_map=LABEL_COLORS,
    )
    fig.update_layout(title="Stock-wise Prediction Breakdown", **_PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _render_probability_trend_chart(df: pd.DataFrame) -> None:
    available = [c for c in PROBABILITY_COLUMNS if c in df.columns]
    if df.empty or not available or "Date" not in df.columns:
        st.info("No time-series probability data available for this chart.")
        return
    plot_df = df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    if "Ticker" in plot_df.columns:
        tickers = sorted(plot_df["Ticker"].dropna().unique().tolist())
        if not tickers:
            return
        selected = st.selectbox("Ticker", tickers, key="chart_ticker")
        plot_df = plot_df[plot_df["Ticker"] == selected]
    plot_df = plot_df.sort_values("Date")
    color_map = {
        "Prob_Up": COLORS["up"],
        "Prob_Down": COLORS["down"],
        "Prob_Neutral": COLORS["neutral"],
    }
    fig = go.Figure()
    for col in available:
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df[col],
                mode="lines+markers",
                name=col.replace("Prob_", ""),
                line=dict(color=color_map.get(col)),
            )
        )
    fig.update_layout(
        title="Probability Trend Over Time", yaxis_range=[0, 1], **_PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    st.set_page_config(
        page_title=settings.app_title, page_icon=settings.app_icon, layout="wide"
    )
    load_css()
    init_session_state()

    api_client = get_api_client(
        st.session_state.get("api_base_url", settings.api_base_url)
    )
    api_base_url = render_sidebar(api_client)

    st.title("📈 Stock Market Prediction Dashboard")
    st.caption("A CatBoost-powered stock movement classifier, served through FastAPI.")

    tabs = st.tabs(["📊 Dashboard", "🔮 Prediction", "🔌 API Status", "ℹ️ About"])
    with tabs[0]:
        render_dashboard_tab(api_client)
    with tabs[1]:
        render_prediction_tab(api_client)
    with tabs[2]:
        render_api_status_tab(api_client, api_base_url)
    with tabs[3]:
        render_about_tab()

    render_footer()


if __name__ == "__main__":
    main()
