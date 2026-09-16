# ============================================================
# Dashboard — Streamlit entry point (placeholder)
# ============================================================
"""
Streamlit dashboard for the ML + Quants Trading Agent.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="ML + Quants Trading Agent",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 ML + Quants Trading Agent")
    st.markdown(
        """
        **Phase 1 complete** — project skeleton is ready.

        Future phases will populate this dashboard with:
        - Live / historical data explorer
        - Feature importance charts
        - Model performance metrics
        - Backtest equity curves
        - Paper-trading order blotter
        """
    )
    st.info("Nothing to display yet — stay tuned for Phase 2!")


if __name__ == "__main__":
    main()
