from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round as spark_round, count, when
from logutils import append_log_to_file

def main():
    spark = SparkSession.builder \
        .appName("Flight Delay Analysis") \
        .master("spark://10.0.0.4:7077") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "32g") \
        .config("spark.sql.shuffle.partitions", "192") \
        .getOrCreate()

    file_path = r"F:\development\datasets\flights-1m-parquet.json"
    df = spark.read.json(file_path)

    append_log_to_file("log.txt", "✅ Loaded flight data")

    # Total flights
    total = df.count()
    append_log_to_file("log.txt", f"✈️ Total flights: {total}")

    # Average arrival delay
    avg_delay = df.select(avg(col("ARR_DELAY")).alias("avg_arr_delay")).first()["avg_arr_delay"]
    append_log_to_file("log.txt", f"🕒 Average arrival delay: {str(round(avg_delay, 2))} minutes")


    # Top 10 worst days by average arrival delay
    top_delay_days = df.groupBy("FL_DATE") \
        .agg(spark_round(avg("ARR_DELAY"), 2).alias("avg_arr_delay")) \
        .orderBy(col("avg_arr_delay").desc()) \
        .limit(10)
    append_log_to_file("log.txt", "\n📅 Top 10 worst days by arrival delay:")
    for row in top_delay_days.collect():
        append_log_to_file("log.txt", f"{str(row['FL_DATE'])}: {str(row['avg_arr_delay'])} min")

    # Average flight time and distance
    avg_stats = df.agg(
        spark_round(avg("AIR_TIME"), 2).alias("avg_air_time"),
        spark_round(avg("DISTANCE"), 2).alias("avg_distance")
    ).first()
    append_log_to_file("log.txt", f"\n⏱️ Avg air time: {str(avg_stats['avg_air_time'])} minutes")
    append_log_to_file("log.txt", f"📏 Avg distance: {str(avg_stats['avg_distance'])} miles")

    # Arrival delay buckets
    delay_buckets = df.select(
        when(col("ARR_DELAY") <= 0, "On time or early")
        .when((col("ARR_DELAY") > 0) & (col("ARR_DELAY") <= 15), "0-15 min late")
        .when((col("ARR_DELAY") > 15) & (col("ARR_DELAY") <= 60), "15-60 min late")
        .otherwise(">60 min late")
        .alias("delay_bucket")
    ).groupBy("delay_bucket").agg(count("*").alias("count")).orderBy("count", ascending=False)

    append_log_to_file("log.txt", "\n📊 Arrival delay distribution:")
    for row in delay_buckets.collect():
        append_log_to_file("log.txt", f"{str(row['delay_bucket'])}: {str(row['count'])} flights")

    append_log_to_file("log.txt", "\n✅ Flight delay analysis completed.")

    spark.stop()

if __name__ == "__main__":
    main()
 
