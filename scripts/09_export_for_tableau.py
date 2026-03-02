

from pathlib import Path
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET_FULL = "data/processed/susy_parquet"
MODEL_RESULTS = "data/processed/model_results.csv"
OUT_DIR = Path("tableau")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if Path(MODEL_RESULTS).exists():
        mr = pd.read_csv(MODEL_RESULTS)

        num_cols = ["train_seconds", "accuracy", "f1", "weightedPrecision", "weightedRecall", "roc_auc", "pr_auc"]
        for c in num_cols:
            if c in mr.columns:
                mr[c] = pd.to_numeric(mr[c], errors="coerce")

        mr.to_csv(OUT_DIR / "model_metrics.csv", index=False)
        print("Wrote:", OUT_DIR / "model_metrics.csv")
    else:
        print("Missing model results:", MODEL_RESULTS)

    spark = (
        SparkSession.builder
        .appName("SUSY_Tableau_Exports")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )

    df = spark.read.parquet(PARQUET_FULL)
    total = df.count()
    dist = (
        df.groupBy("label")
        .count()
        .withColumn("percent", F.col("count") / F.lit(total))
        .orderBy("label")
    )
    dist_pd = dist.toPandas()
    dist_pd.to_csv(OUT_DIR / "data_quality.csv", index=False)
    print("Wrote:", OUT_DIR / "data_quality.csv")

    feature_cols = [f"f{i}" for i in range(1, 19)]

    aggs = []
    for c in feature_cols:
        aggs.extend([
            F.mean(c).alias(f"{c}_mean"),
            F.stddev(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
        ])

    overall = df.agg(*aggs).toPandas()

    rows = []
    for c in feature_cols:
        rows.append({
            "feature": c,
            "mean": float(overall.loc[0, f"{c}_mean"]),
            "std": float(overall.loc[0, f"{c}_std"]),
            "min": float(overall.loc[0, f"{c}_min"]),
            "max": float(overall.loc[0, f"{c}_max"]),
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_summary_overall.csv", index=False)
    print("Wrote:", OUT_DIR / "feature_summary_overall.csv")
    by_class_rows = []
    grouped = df.groupBy("label").agg(*[F.mean(c).alias(f"{c}_mean") for c in feature_cols],
                                     *[F.stddev(c).alias(f"{c}_std") for c in feature_cols]).toPandas()

    for _, r in grouped.iterrows():
        label = float(r["label"])
        for c in feature_cols:
            by_class_rows.append({
                "label": label,
                "feature": c,
                "mean": float(r[f"{c}_mean"]),
                "std": float(r[f"{c}_std"]),
            })

    pd.DataFrame(by_class_rows).to_csv(OUT_DIR / "feature_summary_by_class.csv", index=False)
    print("Wrote:", OUT_DIR / "feature_summary_by_class.csv")

    spark.stop()
    print("Done exporting Tableau files.")


if __name__ == "__main__":
    main()
