# ML + Quants Trading Agent 🤖📈

[![CI Pipeline](https://github.com/dhanushkumar-amk/ML---QUANTS-TRADING-AGENT/actions/workflows/ci.yml/badge.svg)](https://github.com/dhanushkumar-amk/ML---QUANTS-TRADING-AGENT/actions/workflows/ci.yml)

An end-to-end quantitative trading system combining classical financial engineering (GARCH, mean-reversion, momentum) with modern ML/DL/NLP models. The agent ingests market data, engineers predictive features, trains and evaluates multiple model families (gradient boosting, deep learning, transformer-based NLP for sentiment), back-tests strategies with realistic transaction costs, and executes paper/live trades via the Alpaca API — all orchestrated through a Streamlit dashboard.

> **Status:** Phase 3.5 of 50 — Environment, multi-source historical pipeline, real-time streaming, and CI/CD testing complete.

---

## Project Structure

```
ML + QUANTS TRADING AGENT/
│
├── data/                       # All data assets (git-ignored)
│   ├── raw/                    #   Raw market / alternative data
│   ├── processed/              #   Cleaned & transformed datasets
│   └── external/               #   Third-party or manually sourced data
│
├── src/                        # Source code (importable package)
│   ├── data_pipeline/          #   Data ingestion, cleaning, storage
│   ├── features/               #   Feature engineering & selection
│   ├── models/                 #   Model training, evaluation, registry
│   ├── backtest/               #   Backtesting engine & analytics
│   ├── execution/              #   Order management & broker integration
│   ├── nlp/                    #   NLP / sentiment analysis pipelines
│   └── utils/                  #   Shared utilities (logging, config, helpers)
│
├── notebooks/                  # Jupyter notebooks for exploration & EDA
├── tests/                      # Unit & integration tests (pytest)
├── configs/                    # YAML configuration files
├── dashboard/                  # Streamlit dashboard app
│
├── .env.example                # Template for API keys / secrets
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md                   # ← You are here
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd "ML + QUANTS TRADING AGENT"

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your secrets
cp .env.example .env
# Edit .env with your API keys

# 5. Run the dashboard (coming in later phases)
streamlit run dashboard/app.py
```

---

## Testing & Continuous Integration (CI)

Continuous Integration runs automatically via GitHub Actions on every push and pull request to `master` / `main` across Python 3.10 and 3.11.

All unit tests use mock fixtures (`unittest.mock`) for external APIs (Yahoo Finance, Hugging Face Hub, Alpaca WebSocket) — **no secrets or API keys are required for tests or CI to pass**.

To run tests and code quality checks locally:

```bash
# Run the full test suite with coverage report (threshold >= 70%)
pytest --cov=src --cov-report=term-missing

# Run Ruff linter
ruff check src tests scripts

# Run Black formatting check
black --check src tests scripts
```

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment & repo setup | ✅ |
| 2–10 | Data pipeline & feature engineering | 🔜 |
| 11–20 | Model training & evaluation | 🔜 |
| 21–30 | Backtesting & strategy optimisation | 🔜 |
| 31–40 | NLP / sentiment integration | 🔜 |
| 41–45 | Execution & paper trading | 🔜 |
| 46–50 | Dashboard, monitoring & deployment | 🔜 |

---

## License

This project is for educational and portfolio purposes.
