

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.storagelevel import StorageLevel

RAW_CSV = "data/raw/SUSY.csv"
OUT_FULL = "data/processed/susy_parquet"
OUT_SAMPLE = "data/samples/susy_parquet_sample"


def make_schema() -> StructType:
    fields = [StructField("label", DoubleType(), nullable=False)]
    for i in range(1, 19):
        fields.append(StructField(f"f{i}", DoubleType(), nullable=True))
    return StructType(fields)


def main():
    if not Path(RAW_CSV).exists():
        raise FileNotFoundError(f"Missing: {RAW_CSV}")

    spark = (
        SparkSession.builder
        .appName("SUSY_CSV_to_Parquet")
        .master("local[*]")
     
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .getOrCreate()
    )

    schema = make_schema()

    df = (
        spark.read
        .option("header", "false")
        .schema(schema)
        .csv(RAW_CSV)
    )

    print("Schema:")
    df.printSchema()

    df = df.persist(StorageLevel.MEMORY_AND_DISK)

    print("Preview:")
    df.show(5, truncate=False)

  
    Path(OUT_SAMPLE).parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing sample Parquet -> {OUT_SAMPLE}")
    (
        df.limit(200_000)
        .write.mode("overwrite")
        .parquet(OUT_SAMPLE)
    )


    Path(OUT_FULL).parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing FULL Parquet -> {OUT_FULL}")
    (
        df.repartition(64)
        .write.mode("overwrite")
        .parquet(OUT_FULL)
    )

    df.unpersist()
    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
