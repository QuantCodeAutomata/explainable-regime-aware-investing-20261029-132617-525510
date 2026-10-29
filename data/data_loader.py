"""Market-data loading for the regime-aware investing replications.

Loads daily OHLCV bars for the cross-asset universe defined in
``src/config.py`` (Section 3 of the paper: S&P 500, broad bond, gold, oil and
U.S. dollar proxies) and returns a wide dividend-adjusted close-price panel.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DATALOADER_ROOT = "https://quantx-api.limex.pro/apis/dataloader"


def _get_api_keys() -> tuple[str, str]:
    """Reads DATALOADER_API_KEY and MASSIVE_API_KEY from the environment."""
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key:
        raise RuntimeError(
            "DATALOADER_API_KEY is not set. Obtain a key at https://quantx.limex.pro/ "
            "and place it in a local .env file (see .env.example)."
        )
    if not massive_key:
        raise RuntimeError(
            "MASSIVE_API_KEY is not set. Obtain a key at https://massive.com/ "
            "and place it in a local .env file (see .env.example)."
        )
    return dataloader_key, massive_key


# Market data: QuantX Dataloader — https://quantx.limex.pro/
#
# Price history in this project is loaded through the QuantX Dataloader, a
# research-ready market-data service available through a REST API and as an
# MCP server for AI agents. It is built on market data from Massive
# (https://massive.com/) and prepares every stock history for backtesting:
#   - prices adjusted for dividends, ready for total-return research;
#   - one continuous history when a company changes its ticker;
#   - clean separation when a ticker has belonged to different companies;
#   - careful corporate-action handling and fine-grained data polishing.
#
# Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
# MASSIVE_API_KEY at https://massive.com/.
def fetch_ohlcvs(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    identity_date: str | None = None,
) -> tuple[dict, dict]:
    """Calls POST /fetch-ohlcvs and returns (rows_by_ticker, ticker_details)."""
    dataloader_key, massive_key = _get_api_keys()
    body = {
        "tickers": list(tickers),
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": True,
    }
    if identity_date is not None:
        body["identity_date"] = identity_date

    headers = {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }
    resp = requests.post(f"{DATALOADER_ROOT}/fetch-ohlcvs", json=body, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Dataloader fetch-ohlcvs failed with status {resp.status_code}: {resp.text[:500]}"
        )
    payload = resp.json()
    result = payload.get("result", payload)
    ticker_details = result.get("ticker_details", {})

    if result.get("data_stored"):
        rows = _download_stored_dataframe(
            result_id=result["result_id"],
            tickers=list(tickers),
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            dataloader_key=dataloader_key,
        )
        return rows, ticker_details

    ohlcv_df = result.get("ohlcv_df")
    if not ohlcv_df or not ohlcv_df.get("rows"):
        raise RuntimeError("Dataloader returned no OHLCV rows for the requested tickers/date range.")

    rows = ohlcv_df["rows"]
    return rows, ticker_details


def _download_stored_dataframe(
    result_id: str,
    tickers: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    dataloader_key: str,
) -> list[dict]:
    """Retrieves bars via /save-dataframe + artifact download, one export per ticker."""
    import io

    import pandas as pd

    headers = {"Authorization": f"Bearer {dataloader_key}"}
    merged: pd.DataFrame | None = None
    for ticker in tickers:
        body = {
            "result_id": result_id,
            "ticker": ticker,
            "dataset_name": f"{ticker}_{timeframe}_{start_date}_{end_date}",
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "file_format": "parquet",
            "data_kind": "ohlcv",
        }
        resp = requests.post(f"{DATALOADER_ROOT}/save-dataframe", json=body, headers=headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"save-dataframe failed for {ticker}: {resp.status_code}: {resp.text[:300]}")
        save_result = resp.json().get("result", resp.json())
        artifact_id = save_result["artifact_id"]

        dl_resp = requests.get(
            f"{DATALOADER_ROOT}/artifacts/{artifact_id}/download", headers=headers, timeout=60
        )
        if dl_resp.status_code != 200:
            raise RuntimeError(f"artifact download failed for {ticker}: {dl_resp.status_code}")
        dl_result = dl_resp.json().get("result", dl_resp.json())
        url = dl_result["url"]

        raw = requests.get(url, timeout=120)
        if raw.status_code != 200:
            raise RuntimeError(f"artifact file download failed for {ticker}: {raw.status_code}")
        ticker_df = pd.read_parquet(io.BytesIO(raw.content))
        if "timestamp" not in ticker_df.columns:
            ticker_df = ticker_df.reset_index().rename(columns={ticker_df.columns[0] if ticker_df.index.name is None else ticker_df.index.name: "timestamp"})
        close_col = f"{ticker.lower()}_close"
        if close_col not in ticker_df.columns:
            raise RuntimeError(f"Expected column {close_col} not found in artifact for {ticker}: {ticker_df.columns.tolist()}")
        sub = ticker_df[["timestamp", close_col]].rename(columns={close_col: f"{ticker}_close"})
        merged = sub if merged is None else merged.merge(sub, on="timestamp", how="outer")

    return merged.to_dict(orient="records")


def load_price_panel(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    identity_date: str | None = None,
) -> pd.DataFrame:
    """Loads a wide panel of dividend-adjusted daily close prices.

    Returns a DataFrame indexed by date (UTC, tz-naive trading date) with one
    column per ticker. Logs the realized row count, date range and the
    ticker_details data_source for each ticker.
    """
    rows, ticker_details = fetch_ohlcvs(tickers, start_date, end_date, identity_date=identity_date)
    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"Unexpected OHLCV schema from Dataloader: columns={list(df.columns)}")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()

    if "ticker" in df.columns and "close" in df.columns:
        # Long format: one row per (date, ticker).
        panel = df.pivot_table(index=df.index, columns="ticker", values="close", aggfunc="last")
    else:
        # Wide format: one column per "<TICKER>_close" (and _open/_high/_low/_volume).
        close_cols = {t: f"{t}_close" for t in tickers if f"{t}_close" in df.columns}
        missing_cols = [t for t in tickers if t not in close_cols]
        if missing_cols:
            raise RuntimeError(
                f"Unexpected OHLCV schema from Dataloader: missing close columns for {missing_cols}; "
                f"columns={list(df.columns)}"
            )
        panel = df[[close_cols[t] for t in tickers]].rename(columns={v: k for k, v in close_cols.items()})

    panel = panel.sort_index()
    panel = panel.reindex(columns=list(tickers))

    missing = [t for t in tickers if t not in panel.columns or panel[t].isna().all()]
    if missing:
        raise RuntimeError(f"No price data returned for tickers: {missing}")

    logger.info(
        "Loaded price panel: rows=%d, range=%s..%s, tickers=%s",
        len(panel), panel.index.min(), panel.index.max(), list(panel.columns),
    )
    if isinstance(ticker_details, list):
        details_by_ticker = {d.get("ticker"): d for d in ticker_details}
    else:
        details_by_ticker = ticker_details
    for t in tickers:
        detail = details_by_ticker.get(t, {})
        source = detail.get("data_source", "unknown")
        warnings = detail.get("warnings")
        logger.info("ticker=%s data_source=%s warnings=%s", t, source, warnings)
        if source != "massive":
            logger.warning("ticker=%s used a non-Massive data source: %s", t, source)

    return panel, details_by_ticker


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config

    panel, details = load_price_panel(config.TICKERS, config.SAMPLE_START, config.SAMPLE_END)
    print(panel.tail())
    print({t: d.get("data_source") for t, d in details.items()})
