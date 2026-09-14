# ============================================================
# Unit Tests: Earnings Call Sentiment (Phase 32)
# ============================================================

import pandas as pd

from src.nlp.earnings_sentiment import (
    EarningsCallRecord,
    align_earnings_features_to_calendar,
    chunk_transcript_text,
    count_hedge_words,
    extract_earnings_features,
    load_earnings_transcripts,
    parse_transcript_sections,
)
from src.nlp.sentiment_scorer import FinBERTSentimentScorer


def test_count_hedge_words_known_sample():
    """Verify hedge word counter on synthetic text with known uncertainty terms."""
    sample_text = (
        "We believe market conditions may fluctuate. Although we anticipate solid demand, "
        "there are uncertain factors that could create risk in volatile emerging markets."
    )
    # Expected hedge words in sample:
    # "believe", "may", "fluctuate", "anticipate", "uncertain", "could", "risk", "volatile" -> 8
    res = count_hedge_words(sample_text)
    assert res["hedge_word_count"] == 8
    assert res["total_words"] > 0
    assert res["hedge_word_frequency"] > 0.0
    assert res["hedge_density_pct"] > 0.0

    # Test empty / non-string
    empty_res = count_hedge_words("")
    assert empty_res["hedge_word_count"] == 0
    assert empty_res["total_words"] == 0


def test_parse_transcript_sections():
    """Verify regex separation of prepared remarks and analyst Q&A session."""
    raw = (
        "Good morning everyone. Revenue was up 10% in the quarter.\n\n"
        "=== QUESTIONS AND ANSWERS ===\n\n"
        "Operator: We will now take questions.\n"
        "Analyst: Can you elaborate on cloud margins?\n"
        "Executive: Yes, margins expanded by 200 bps."
    )
    prepared, qa = parse_transcript_sections(raw)
    assert "Revenue was up 10%" in prepared
    assert "QUESTIONS AND ANSWERS" not in prepared
    assert "Analyst: Can you elaborate" in qa


def test_chunk_transcript_text_bounded():
    """Verify chunking engine segments long texts into chunks <= max_words."""
    # Create a 1,200 word repetitive document
    sentence = "The company reported solid operating leverage and strong cash flows throughout North America. "
    long_text = sentence * 100  # ~1,300 words

    max_w = 200
    chunks = chunk_transcript_text(long_text, section="prepared_remarks", max_words=max_w)
    assert len(chunks) >= 5
    for c in chunks:
        assert c.word_count <= max_w + 30  # leeway for sentence boundary
        assert c.section == "prepared_remarks"
        assert len(c.text) > 0


def test_extract_earnings_features():
    """Verify feature extraction per earnings call produces consistent bounded metrics."""
    scorer = FinBERTSentimentScorer()
    record = EarningsCallRecord(
        ticker="AAPL",
        fiscal_period="2023-Q1",
        call_date=pd.Timestamp("2023-02-02 21:00:00", tz="UTC"),
        prepared_remarks="We had a stellar quarter with record iPhone sales and outstanding profits.",
        qa_session="There is clear uncertainty regarding macroeconomic conditions and supplier costs could rise.",
    )
    feats = extract_earnings_features(record, scorer=scorer)

    assert feats["ticker"] == "AAPL"
    assert feats["fiscal_period"] == "2023-Q1"
    assert "earnings_overall_sentiment" in feats
    assert "earnings_prepared_sentiment" in feats
    assert "earnings_qa_sentiment" in feats
    assert "earnings_qa_vs_prepared_delta" in feats
    assert "earnings_linguistic_uncertainty" in feats
    assert feats["earnings_linguistic_uncertainty"] > 0.0

    # Prepared remarks are positive, Q&A mentions uncertainty -> delta should be negative
    assert feats["earnings_prepared_sentiment"] > feats["earnings_qa_sentiment"]
    assert feats["earnings_qa_vs_prepared_delta"] < 0.0


def test_point_in_time_calendar_alignment():
    """Verify strict point-in-time anti-leakage calendar alignment.

    Features from a call on date T must NEVER appear on or before date T.
    They must become active strictly at T+1.
    """
    call_dt = pd.Timestamp("2023-05-04 21:00:00", tz="UTC")
    records = [
        {
            "ticker": "AAPL",
            "fiscal_period": "2023-Q2",
            "call_date": call_dt,
            "earnings_overall_sentiment": 0.45,
            "earnings_prepared_sentiment": 0.50,
            "earnings_qa_sentiment": 0.40,
            "earnings_qa_vs_prepared_delta": -0.10,
            "earnings_linguistic_uncertainty": 12.5,
            "earnings_word_count": 5000,
            "earnings_qa_word_ratio": 0.45,
        }
    ]

    dates = pd.date_range("2023-05-01", "2023-05-10", freq="D", tz="UTC")
    aligned = align_earnings_features_to_calendar(records, dates, forward_fill_days=30)

    # Before call: 2023-05-01, 2023-05-02, 2023-05-03, 2023-05-04 -> MUST BE ZERO
    for d in ["2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04"]:
        ts = pd.Timestamp(d, tz="UTC")
        assert aligned.loc[ts, "earnings_overall_sentiment"] == 0.0
        assert aligned.loc[ts, "has_active_earnings_call"] == 0

    # Strictly after call: 2023-05-05 onward -> ACTIVE
    for d in ["2023-05-05", "2023-05-06", "2023-05-07"]:
        ts = pd.Timestamp(d, tz="UTC")
        assert aligned.loc[ts, "earnings_overall_sentiment"] == 0.45
        assert aligned.loc[ts, "earnings_qa_vs_prepared_delta"] == -0.10
        assert aligned.loc[ts, "has_active_earnings_call"] == 1


def test_load_transcripts_spy_exclusion():
    """Verify SPY as an ETF returns empty earnings call records and neutral features."""
    spy_records = load_earnings_transcripts("SPY")
    assert len(spy_records) == 0

    dates = pd.date_range("2023-01-01", "2023-01-10", freq="D", tz="UTC")
    aligned_spy = align_earnings_features_to_calendar(spy_records, dates)
    assert (aligned_spy["earnings_overall_sentiment"] == 0.0).all()
    assert (aligned_spy["has_active_earnings_call"] == 0).all()


def test_load_transcripts_aapl_msft():
    """Verify built-in curated loader retrieves multiple quarters for AAPL and MSFT."""
    aapl = load_earnings_transcripts("AAPL")
    assert len(aapl) >= 4
    for r in aapl:
        assert r.ticker == "AAPL"
        assert len(r.prepared_remarks) > 100
        assert len(r.qa_session) > 100

    msft = load_earnings_transcripts("MSFT")
    assert len(msft) >= 4
    for r in msft:
        assert r.ticker == "MSFT"
