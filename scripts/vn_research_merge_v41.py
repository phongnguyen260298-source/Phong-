from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATASETS = {
    "balance_sheet_quarter": "fundamental_quarterly/balance_sheet/*.parquet",
    "income_statement_quarter": "fundamental_quarterly/income_statement/*.parquet",
    "cash_flow_quarter": "fundamental_quarterly/cash_flow/*.parquet",
    "ratio_quarter": "fundamental_quarterly/ratio/*.parquet",
    "price_daily": "price_daily/*.parquet",
    "company_info": "company_info/*.parquet",
}

META_COLS = {
    "symbol", "retrieval_source", "retrieval_utc", "period", "id", "name",
    "unit", "order", "level", "com_type", "source"
}


def read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            x = pd.read_parquet(p)
            if "symbol" not in x.columns:
                x.insert(0, "symbol", p.stem.upper())
            frames.append(x)
        except Exception as exc:
            frames.append(pd.DataFrame({
                "symbol": [p.stem.upper()],
                "_read_error": [f"{type(exc).__name__}: {exc}"],
                "_source_file": [str(p)],
            }))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def dataset_catalog(name: str, df: pd.DataFrame, file_count: int) -> dict:
    out = {
        "dataset": name,
        "files": int(file_count),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
        "periods": int(df["period"].nunique()) if "period" in df.columns else 0,
        "read_error_rows": int(df.get("_read_error", pd.Series(dtype=object)).notna().sum()) if "_read_error" in df.columns else 0,
    }
    if "period" in df.columns and df["period"].notna().any():
        p = df["period"].dropna().astype(str)
        out["first_period"] = p.min()
        out["last_period"] = p.max()
    else:
        out["first_period"] = None
        out["last_period"] = None
    return out


def build_semantic_catalog(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        return pd.DataFrame()
    value_col = "value" if "value" in df.columns else None
    keep = [c for c in ["id", "name", "unit", "level"] if c in df.columns]
    work = df.copy()
    work = work[work["id"].notna()]
    if work.empty:
        return pd.DataFrame()
    gcols = keep
    agg = {
        "symbol": pd.Series.nunique,
    }
    if "period" in work.columns:
        agg["period"] = pd.Series.nunique
    if value_col:
        agg[value_col] = "count"
    cat = work.groupby(gcols, dropna=False).agg(agg).reset_index()
    rename = {"symbol": "symbol_count", "period": "period_count"}
    if value_col:
        rename[value_col] = "non_null_value_rows"
    cat = cat.rename(columns=rename)
    cat.insert(0, "dataset", dataset)
    cat["row_count"] = work.groupby(gcols, dropna=False).size().values
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--cohort-root", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.input_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged_dir = out / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Merge per-chunk manifests/errors/summaries first.
    manifest_files = sorted(root.glob("c*/manifest.csv"))
    manifests = [pd.read_csv(p) for p in manifest_files if p.exists()]
    master_manifest = pd.concat(manifests, ignore_index=True, sort=False) if manifests else pd.DataFrame()
    master_manifest.to_csv(out / "master_manifest.csv", index=False, encoding="utf-8-sig")

    error_frames = []
    for p in sorted(root.glob("c*/errors.csv")):
        try:
            x = pd.read_csv(p)
            if len(x):
                error_frames.append(x)
        except Exception:
            pass
    all_errors = pd.concat(error_frames, ignore_index=True, sort=False) if error_frames else pd.DataFrame()
    all_errors.to_csv(out / "all_errors.csv", index=False, encoding="utf-8-sig")

    run_summaries = []
    for p in sorted(root.glob("c*/run_summary.json")):
        try:
            run_summaries.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    pd.DataFrame(run_summaries).to_csv(out / "chunk_summaries.csv", index=False, encoding="utf-8-sig")

    if not master_manifest.empty:
        cov = master_manifest.copy()
        cov.to_csv(out / "coverage_long.csv", index=False, encoding="utf-8-sig")
        by_symbol = cov.groupby("symbol").agg(
            datasets=("dataset", "nunique"),
            ok_datasets=("status", lambda s: int((s == "OK").sum())),
            empty_datasets=("status", lambda s: int((s == "EMPTY").sum())),
            total_rows=("rows", "sum"),
            total_period_count=("period_count", "sum"),
        ).reset_index()
        by_symbol["complete_6_of_6"] = by_symbol["ok_datasets"].eq(6)
        by_symbol.to_csv(out / "coverage_by_symbol.csv", index=False, encoding="utf-8-sig")
        status_matrix = cov.pivot_table(index="symbol", columns="dataset", values="status", aggfunc="first")
        status_matrix.reset_index().to_csv(out / "coverage_status_matrix.csv", index=False, encoding="utf-8-sig")

    dataset_rows = []
    semantic_parts = []
    schema_rows = []

    for dataset, pattern in DATASETS.items():
        paths = sorted(root.glob(f"c*/{pattern}"))
        df = read_many(paths)
        target = merged_dir / f"{dataset}.parquet"
        if not df.empty:
            df.to_parquet(target, index=False, compression="zstd")
        dataset_rows.append(dataset_catalog(dataset, df, len(paths)))
        for c in df.columns:
            s = df[c]
            sample_vals = s.dropna().astype(str).drop_duplicates().head(3).tolist()
            schema_rows.append({
                "dataset": dataset,
                "column": c,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "sample": " | ".join(sample_vals),
            })
        cat = build_semantic_catalog(dataset, df)
        if not cat.empty:
            semantic_parts.append(cat)

    pd.DataFrame(dataset_rows).to_csv(out / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(schema_rows).to_csv(out / "schema_catalog.csv", index=False, encoding="utf-8-sig")
    semantic = pd.concat(semantic_parts, ignore_index=True, sort=False) if semantic_parts else pd.DataFrame()
    semantic.to_csv(out / "semantic_id_catalog.csv", index=False, encoding="utf-8-sig")

    # Copy cohort definitions into output for one self-contained audit package.
    cohort_root = Path(args.cohort_root) if args.cohort_root else None
    if cohort_root and cohort_root.exists():
        cohort_out = out / "cohort"
        cohort_out.mkdir(exist_ok=True)
        for p in cohort_root.glob("*"):
            if p.is_file():
                (cohort_out / p.name).write_bytes(p.read_bytes())

    summary = {
        "version": "v4.1-merged",
        "chunks": len(run_summaries),
        "symbols_processed": int(sum(x.get("symbols_n", 0) for x in run_summaries)),
        "manifest_rows": int(len(master_manifest)),
        "errors": int(len(all_errors)),
        "datasets_merged": int(len(DATASETS)),
        "semantic_ids": int(semantic["id"].nunique()) if not semantic.empty and "id" in semantic.columns else 0,
        "output_files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
    (out / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
