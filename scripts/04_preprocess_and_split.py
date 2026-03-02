

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

PARQUET_IN = "data/processed/susy_parquet"
OUT_TRAIN = "data/processed/susy_train"
OUT_TEST = "data/processed/susy_test"

TEST_FRACTION = 0.2
SEED = 42


SHUFFLE_PARTITIONS = 32
WRITE_PARTITIONS = 32


def main():
    if not Path(PARQUET_IN).exists():
        raise FileNotFoundError(f"Missing Parquet input: {PARQUET_IN}")

    spark = (
        SparkSession.builder
        .appName("SUSY_Preprocess_Split_OOM_Safe")
        .master("local[*]")
       
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.default.parallelism", str(SHUFFLE_PARTITIONS))
      
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.2")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .getOrCreate()
    )

    df = spark.read.parquet(PARQUET_IN)


    print("Class distribution:")
    df.groupBy("label").count().orderBy("label").show()

    feature_cols = [c for c in df.columns if c.startswith("f")]
    if len(feature_cols) != 18:
        raise ValueError(f"Expected 18 feature columns, found {len(feature_cols)}")


    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip",
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withStd=True,
        withMean=False,
    )

    pipe = Pipeline(stages=[assembler, scaler])

   
    fit_df = df.sample(withReplacement=False, fraction=0.2, seed=SEED)
    pipe_model = pipe.fit(fit_df)

    
    df_pre = pipe_model.transform(df).select("label", "scaledFeatures")

  
    train_df, test_df = df_pre.randomSplit([1 - TEST_FRACTION, TEST_FRACTION], seed=SEED)

    
    train_df = train_df.repartition(WRITE_PARTITIONS)
    test_df = test_df.repartition(WRITE_PARTITIONS)

   
    Path(OUT_TRAIN).parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing train -> {OUT_TRAIN}")
    train_df.write.mode("overwrite").parquet(OUT_TRAIN)

    print(f"Writing test -> {OUT_TEST}")
    test_df.write.mode("overwrite").parquet(OUT_TEST)

    spark.stop()
    print("Done (OOM-safe).")


if __name__ == "__main__":
    main()