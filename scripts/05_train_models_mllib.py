

from pathlib import Path
import csv
import sys
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

TRAIN_FULL = "data/processed/susy_train"
TEST_FULL = "data/processed/susy_test"


SAMPLE_PARQUET = "data/samples/susy_parquet_sample"

OUT_CSV = "data/processed/model_results.csv"

SHUFFLE_PARTITIONS = 16  
SEED = 42


def eval_multiclass(pred_df):
    out = {}
    for name, metric in [
        ("accuracy", "accuracy"),
        ("f1", "f1"),
        ("weightedPrecision", "weightedPrecision"),
        ("weightedRecall", "weightedRecall"),
    ]:
        ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
        out[name] = float(ev.evaluate(pred_df))
    return out


def eval_binary(pred_df):
    out = {}
    roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    out["roc_auc"] = float(roc.evaluate(pred_df))
    out["pr_auc"] = float(pr.evaluate(pred_df))
    return out


def main():
    use_sample = ("--sample" in sys.argv)

    spark = (
        SparkSession.builder
        .appName("SUSY_Train_Models_MLlib_OOM_Safe")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.default.parallelism", str(SHUFFLE_PARTITIONS))
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .getOrCreate()
    )

    ckpt_dir = "data/processed/_spark_checkpoints"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(ckpt_dir)

    if use_sample:
        if not Path(SAMPLE_PARQUET).exists():
            raise FileNotFoundError(f"Missing sample parquet: {SAMPLE_PARQUET}")
        df = spark.read.parquet(SAMPLE_PARQUET)

 
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)
        feature_cols = [f"f{i}" for i in range(1, 19)]
        from pyspark.ml.feature import VectorAssembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        train_df = assembler.transform(train_df).select("label", "features")
        test_df = assembler.transform(test_df).select("label", "features")

    else:
        for p in [TRAIN_FULL, TEST_FULL]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing: {p}. Run Step 7 first (04_preprocess_and_split.py).")

        train_df = spark.read.parquet(TRAIN_FULL).select("label", "scaledFeatures")
        test_df = spark.read.parquet(TEST_FULL).select("label", "scaledFeatures")

 
        train_df = train_df.withColumnRenamed("scaledFeatures", "features")
        test_df = test_df.withColumnRenamed("scaledFeatures", "features")

    train_df = train_df.repartition(SHUFFLE_PARTITIONS).checkpoint(eager=True)
    test_df = test_df.repartition(SHUFFLE_PARTITIONS).checkpoint(eager=True)

    models = [
        ("LogisticRegression",
         LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.01)),
        ("DecisionTree",
         DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=6)),
        ("RandomForest",
         RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=30, maxDepth=8)),
        ("GBTClassifier",
         GBTClassifier(featuresCol="features", labelCol="label", maxIter=20, maxDepth=5)),
    ]

    run_time = datetime.now().isoformat(timespec="seconds")
    mode = "sample" if use_sample else "full"
    rows = []

    for model_name, estimator in models:
        print(f"\n=== Training: {model_name} ({mode}) ===")
        start = datetime.now()

        model = estimator.fit(train_df)
        pred = model.transform(test_df).select("label", "prediction", "rawPrediction", "probability")

   
        pred = pred.persist(StorageLevel.MEMORY_AND_DISK)

        m = {}
        m.update(eval_multiclass(pred))
        m.update(eval_binary(pred))

        elapsed = (datetime.now() - start).total_seconds()
        pred.unpersist()

        row = {
            "run_time": run_time,
            "mode": mode,
            "model": model_name,
            "train_seconds": elapsed,
            **m,
        }
        print("Metrics:", row)
        rows.append(row)

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_time", "mode", "model", "train_seconds",
                  "accuracy", "f1", "weightedPrecision", "weightedRecall", "roc_auc", "pr_auc"]

    write_header = not Path(OUT_CSV).exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved results to: {OUT_CSV}")
    spark.stop()


if __name__ == "__main__":
    main()