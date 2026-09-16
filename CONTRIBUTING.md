# Contributing to ML + Quants Trading Agent

Thank you for your interest in contributing! This project is an open-source quantitative trading research framework designed to bridge classical financial engineering with machine learning, deep learning, and alternative NLP sentiment.

---

## Code of Conduct & Scope

1. **Educational & Research Focus**: This project is built strictly for **research, simulation, and educational purposes**. Code contributions must uphold deterministic testing, paper trading guardrails, and risk controls.
2. **No Real Capital / Security**: Never commit live API keys, broker secret credentials, or real account numbers. All automated tests must run against mock fixtures or paper trading sandbox endpoints.

---

## Development Workflow

### 1. Fork & Branch
```bash
# Clone your fork
git clone https://github.com/<your-username>/ML---QUANTS-TRADING-AGENT.git
cd "ML + QUANTS TRADING AGENT"

# Create a focused feature or fix branch
git checkout -b feature/your-feature-name
```

### 2. Python Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 3. Next.js Dashboard Setup
```bash
cd dashboard
npm install
npm run dev
```

### 4. Running Backend Tests & Linting
Before submitting a PR, verify all Python tests, type checks, and formatting pass:
```bash
# Run pytest with coverage
pytest --cov=src --cov-report=term-missing

# Linting and formatting checks
ruff check src tests scripts
black --check src tests scripts
```

### 5. Running Frontend Tests & Type Checking
```bash
cd dashboard
npx tsc --noEmit
npm run test
```

---

## Pull Request Guidelines

1. **Keep Changes Atomic**: One PR per feature, bugfix, or model integration.
2. **Add Unit Tests**: Any new indicator, risk check, or feature calculation must include corresponding tests in `/tests`.
3. **Deterministic Fixtures**: Use mock data (`unittest.mock`) rather than live network calls in CI test pipelines.
4. **Documentation**: Update docstrings and the relevant phase documentation when altering core APIs.
5. **No Broken Windows**: Ensure CI passes on all matrix configurations (Python 3.10 and 3.11).

---

## License

By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
