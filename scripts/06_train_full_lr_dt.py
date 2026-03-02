

from pathlib import Path
import csv
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

TRAIN_PATH = "data/processed/susy_train"
TEST_PATH = "data/processed/susy_test"
OUT_CSV = "data/processed/model_results.csv"

SHUFFLE_PARTITIONS = 16  


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
    for p in [TRAIN_PATH, TEST_PATH]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing: {p}. Run preprocessing step first.")

    spark = (
        SparkSession.builder
        .appName("SUSY_Full_Train_LR_DT")
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

    train = spark.read.parquet(TRAIN_PATH).select("label", "scaledFeatures") \
        .withColumnRenamed("scaledFeatures", "features") \
        .repartition(SHUFFLE_PARTITIONS).checkpoint(eager=True)

    test = spark.read.parquet(TEST_PATH).select("label", "scaledFeatures") \
        .withColumnRenamed("scaledFeatures", "features") \
        .repartition(SHUFFLE_PARTITIONS).checkpoint(eager=True)

    models = [
        ("LogisticRegression_FULL",
         LogisticRegression(featuresCol="features", labelCol="label", maxIter=40, regParam=0.01)),
        ("DecisionTree_FULL",
         DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=6)),
    ]

    run_time = datetime.now().isoformat(timespec="seconds")
    rows = []

    for name, est in models:
        print(f"\n=== Training: {name} ===")
        start = datetime.now()

        model = est.fit(train)
        pred = model.transform(test).select("label", "prediction", "rawPrediction", "probability")
        pred = pred.persist(StorageLevel.MEMORY_AND_DISK)

        m = {}
        m.update(eval_multiclass(pred))
        m.update(eval_binary(pred))
        elapsed = (datetime.now() - start).total_seconds()
        pred.unpersist()

        row = {"run_time": run_time, "mode": "full", "model": name, "train_seconds": elapsed, **m}
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
