from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

import vn_market_research_lake_v2 as base

LONG_ANCHOR = pd.Timestamp("2013-12-31")
MODERN_ANCHOR = pd.Timestamp("2019-12-31")
VALID_EXCHANGES = {"HOSE", "HNX"}


def active_at_anchor(df: pd.DataFrame, anchor: pd.Timestamp) -> pd.Series:
    listed = pd.to_datetime(df["listed_date_best"], errors="coerce")
    delisted = pd.to_datetime(df["delisted_date_best"], errors="coerce")
    return (
        df["exchange_best"].isin(VALID_EXCHANGES)
        & listed.notna()
        & listed.le(anchor)
        & (delisted.isna() | delisted.gt(anchor))
    )


def build_cohorts(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    universe = base.build_universe(out / "universe_snapshot")

    universe["cohort_long_2014"] = active_at_anchor(universe, LONG_ANCHOR)
    universe["cohort_modern_2020"] = active_at_anchor(universe, MODERN_ANCHOR)
    universe["cohort_union"] = universe["cohort_long_2014"] | universe["cohort_modern_2020"]

    # The current VNStock listing tables provide a reliable current/known exchange
    # and listing interval, but do not guarantee a complete historical exchange-transfer
    # path for every ticker. We therefore flag transfer-like cases for audit rather
    # than silently assuming exact historical exchange membership.
    universe["transfer_audit_flag"] = (
        universe.get("previous_market_exit_date", pd.Series(index=universe.index, dtype=object)).notna()
        | universe.get("listing_interval_issue_flag", pd.Series(False, index=universe.index)).fillna(False)
    )

    cols = [
        "symbol", "exchange_best", "listed_date_best", "delisted_date_best",
        "status_best", "icb_code2", "organ_name",
        "cohort_long_2014", "cohort_modern_2020", "cohort_union",
        "transfer_audit_flag", "previous_market_exit_date",
        "listing_interval_issue_flag", "seen_vci", "seen_vnd", "seen_kbs",
    ]
    cols = [c for c in cols if c in universe.columns]
    union = universe.loc[universe["cohort_union"], cols].sort_values("symbol").reset_index(drop=True)
    long = union.loc[union["cohort_long_2014"]].copy()
    modern = union.loc[union["cohort_modern_2020"]].copy()
    audit = union.loc[union["transfer_audit_flag"]].copy()

    long.to_csv(out / "cohort_long_2014.csv", index=False, encoding="utf-8-sig")
    modern.to_csv(out / "cohort_modern_2020.csv", index=False, encoding="utf-8-sig")
    union.to_csv(out / "cohort_union.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "cohort_transfer_audit.csv", index=False, encoding="utf-8-sig")

    for name, df in {
        "cohort_long_2014": long,
        "cohort_modern_2020": modern,
        "cohort_union": union,
        "cohort_transfer_audit": audit,
    }.items():
        df.to_parquet(out / f"{name}.parquet", index=False, compression="zstd")

    summary = {
        "version": "v4",
        "selection_rule": "HOSE/HNX equity-like firms active at the anchor date; no survivor-to-2026 requirement",
        "long_anchor": str(LONG_ANCHOR.date()),
        "long_data_window": "2014-01-01/2025-12-31",
        "long_n": int(len(long)),
        "modern_anchor": str(MODERN_ANCHOR.date()),
        "modern_data_window": "2020-01-01/2025-12-31",
        "modern_n": int(len(modern)),
        "union_n": int(len(union)),
        "overlap_n": int((union["cohort_long_2014"] & union["cohort_modern_2020"]).sum()),
        "long_only_n": int((union["cohort_long_2014"] & ~union["cohort_modern_2020"]).sum()),
        "modern_only_n": int((~union["cohort_long_2014"] & union["cohort_modern_2020"]).sum()),
        "transfer_audit_n": int(len(audit)),
        "historical_exchange_note": (
            "exchange_best is the current/known exchange from the listing sources. "
            "Potential transfer-history cases are flagged for separate audit."
        ),
    }
    (out / "cohort_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/vn_research_cohort_v4")
    args = p.parse_args()
    build_cohorts(Path(args.out))


if __name__ == "__main__":
    main()
