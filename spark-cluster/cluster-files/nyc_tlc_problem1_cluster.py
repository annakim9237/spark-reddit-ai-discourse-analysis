#!/usr/bin/env python3
"""
Problem 1: Daily summaries of key metrics for NYC TLC data (CLUSTER VERSION)

This script downloads 6 months of NYC TLC data (Jan-June 2021),
combines them into a single DataFrame, and calculates daily summaries.
"""

import os
import subprocess
import sys
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    year, month, dayofmonth,
    avg, count, max as spark_max, min as spark_min,
    expr, ceil, percentile_approx
)
import pandas as pd

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def create_spark_session(master_url):
    """Create a Spark session optimized for cluster execution."""

    spark = (
        SparkSession.builder
        .appName("Problem1_DailySummaries_Cluster")

        # Cluster Configuration
        .master(master_url)  # Connect to Spark cluster

        # Memory Configuration
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")

        # Executor Configuration
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")  # Use all available cores across cluster

        # S3 Configuration - Use S3A for AWS S3 access
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

        # Performance settings for cluster execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Arrow optimization for Pandas conversion
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        .getOrCreate()
    )

    logger.info("Spark session created successfully for cluster execution")
    return spark


def get_s3_paths(months_to_download):
    """Generate S3 paths for NYC TLC data (cluster reads directly from S3)."""

    logger.info(f"Preparing S3 paths for {len(months_to_download)} months of NYC TLC data")
    print(f"\nPreparing to read {len(months_to_download)} months of NYC TLC data from S3...")
    print("=" * 60)

    s3_paths = []

    for month_num in months_to_download:
        month_str = f"{month_num:02d}"
        s3_path = f"s3a://bigdatateaching/nyc_tlc/trip_data/yyyy=2021/yellow_tripdata_2021-{month_str}.parquet"
        s3_paths.append(s3_path)
        print(f"  Month {month_str}/2021: {s3_path}")
        logger.info(f"Added S3 path for month {month_str}: {s3_path}")

    logger.info(f"Prepared {len(s3_paths)} S3 paths for direct reading")
    print(f"\n✅ Prepared {len(s3_paths)} S3 paths - will read directly from S3")
    return s3_paths


def solve_problem1(spark, data_files):
    """
    Solve Problem 1: Calculate daily summaries of key metrics.

    Requirements:
    1. Derive dt_year, dt_month, dt_day from tpep_pickup_datetime
    2. Filter for year 2021
    3. Calculate daily summaries:
       - Number of trips
       - Average trip_distance
       - Max mta_tax
       - 95th percentile of fare_amount
       - Min tip_amount
       - Average passenger_count (rounded up)
    4. Sort by dt_year, dt_month, dt_day in descending order
    """

    logger.info("Starting Problem 1: Daily Summaries of Key Metrics")
    print("\nSolving Problem 1: Daily Summaries of Key Metrics")
    print("=" * 60)

    start_time = time.time()

    # Read all parquet files into a single DataFrame
    logger.info(f"Reading {len(data_files)} parquet files into single DataFrame")
    print("Reading all data files into a single DataFrame...")
    nyc_tlc = spark.read.parquet(*data_files)

    total_rows = nyc_tlc.count()
    logger.info(f"Successfully loaded {total_rows:,} total rows from {len(data_files)} files")
    print(f"✅ Loaded {total_rows:,} total rows from {len(data_files)} files")

    # Step 1: Derive date columns
    logger.info("Step 1: Deriving date columns from tpep_pickup_datetime")
    print("\nStep 1: Deriving date columns from tpep_pickup_datetime...")
    nyc_tlc = (nyc_tlc
        .withColumn("dt_year", year("tpep_pickup_datetime"))
        .withColumn("dt_month", month("tpep_pickup_datetime"))
        .withColumn("dt_day", dayofmonth("tpep_pickup_datetime"))
    )

    # Step 2: Filter for year 2021
    logger.info("Step 2: Filtering data for year 2021")
    print("Step 2: Filtering data for year 2021...")
    nyc_tlc_2021 = nyc_tlc.filter(nyc_tlc.dt_year == 2021)
    filtered_rows = nyc_tlc_2021.count()
    logger.info(f"Filtered dataset to {filtered_rows:,} rows for year 2021")
    print(f"✅ Filtered to {filtered_rows:,} rows for year 2021")

    # Step 3: Calculate daily summaries
    logger.info("Step 3: Calculating daily summaries with aggregations")
    print("Step 3: Calculating daily summaries...")
    daily_averages = (nyc_tlc_2021
        .groupBy("dt_year", "dt_month", "dt_day")
        .agg(
            count("*").alias("num_trips"),
            avg("trip_distance").alias("mean_trip_distance"),
            spark_max("mta_tax").alias("max_mta_tax"),
            expr("percentile_approx(fare_amount, 0.95)").alias("q95_fare_amount"),
            spark_min("tip_amount").alias("min_tip_amount"),
            ceil(avg("passenger_count")).alias("mean_passenger_count")
        )
    )

    # Step 4: Sort by date in descending order
    logger.info("Step 4: Sorting results by date in descending order")
    print("Step 4: Sorting by date (descending)...")
    daily_averages = daily_averages.orderBy(
        "dt_year", "dt_month", "dt_day",
        ascending=[False, False, False]
    )

    # Display the resulting DataFrame
    logger.info("Step 5: Displaying results")
    print("\nStep 5: Displaying results...")
    print("\nTop 20 daily summaries (sorted in descending order):")
    daily_averages.show(20)

    # Get total number of days
    total_days = daily_averages.count()
    logger.info(f"Calculated summaries for {total_days} days")
    print(f"\n✅ Calculated summaries for {total_days} days")

    # Step 6: Convert to Pandas and save
    logger.info("Step 6: Converting Spark DataFrame to Pandas DataFrame")
    print("\nStep 6: Converting to Pandas DataFrame...")
    pandas_df = daily_averages.toPandas()

    # Step 7: Save to CSV
    output_file = "daily_averages_cluster.csv"
    logger.info(f"Step 7: Saving results to {output_file}")
    pandas_df.to_csv(output_file, index=False)
    print(f"Step 7: ✅ Results saved to {output_file}")

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Problem 1 execution completed in {execution_time:.2f} seconds")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("PROBLEM 1 COMPLETED - Summary Statistics")
    print("=" * 60)
    print(f"Total rows processed: {filtered_rows:,}")
    print(f"Days with data: {total_days}")
    print(f"Date range: {pandas_df['dt_month'].min():02d}/{pandas_df['dt_day'].min():02d}/2021 to {pandas_df['dt_month'].max():02d}/{pandas_df['dt_day'].max():02d}/2021")
    print(f"Total trips: {pandas_df['num_trips'].sum():,}")
    print(f"Average daily trips: {pandas_df['num_trips'].mean():.0f}")
    print(f"Execution time: {execution_time:.2f} seconds")

    # Show sample results
    print("\n" + "=" * 60)
    print("Sample Results (first 5 rows):")
    print("=" * 60)
    print(f"Note: Full results saved to {output_file}")

    # Show first 5 rows with proper formatting
    sample_df = pandas_df.head(5)
    for _, row in sample_df.iterrows():
        print(f"Year: {int(row['dt_year'])}, Month: {int(row['dt_month']):02d}, Day: {int(row['dt_day']):02d}")
        print(f"  Trips: {int(row['num_trips']):,}")
        print(f"  Avg Distance: {row['mean_trip_distance']:.2f} miles")
        print(f"  Max MTA Tax: ${row['max_mta_tax']:.2f}")
        print(f"  95th Percentile Fare: ${row['q95_fare_amount']:.2f}")
        print(f"  Min Tip: ${row['min_tip_amount']:.2f}")
        print(f"  Avg Passengers: {row['mean_passenger_count']:.0f}")
        print("-" * 40)

    return pandas_df


def main():
    """Main function for Problem 1 - Cluster Version."""

    logger.info("Starting Problem 1: Daily Summaries of Key Metrics (Cluster Mode)")
    print("=" * 70)
    print("PROBLEM 1: Daily Summaries of Key Metrics (CLUSTER MODE)")
    print("NYC Taxi Trip Data Analysis (Jan-June 2021)")
    print("=" * 70)

    # Get master URL from command line or environment variable
    if len(sys.argv) > 1:
        master_url = sys.argv[1]
    else:
        # Try to get from environment variable
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("❌ Error: Master URL not provided")
            print("Usage: python nyc_tlc_problem1_cluster.py spark://MASTER_IP:7077")
            print("   or: export MASTER_PRIVATE_IP=xxx.xxx.xxx.xxx")
            return 1

    print(f"Connecting to Spark Master at: {master_url}")
    logger.info(f"Using Spark master URL: {master_url}")

    overall_start = time.time()

    # Create Spark session
    logger.info("Initializing Spark session for cluster execution")
    spark = create_spark_session(master_url)

    # Get S3 paths for 12 months of data (full year 2021)
    months_to_download = list(range(1, 13))
    logger.info(f"Preparing S3 paths for {len(months_to_download)} months of data: {months_to_download}")
    data_files = get_s3_paths(months_to_download)

    if len(data_files) == 0:
        logger.error("No S3 paths available. Cannot proceed with analysis")
        print("❌ No S3 paths available. Exiting...")
        spark.stop()
        return 1

    # Solve Problem 1
    try:
        logger.info("Starting Problem 1 analysis with downloaded data files")
        result_df = solve_problem1(spark, data_files)
        success = True
        logger.info("Problem 1 analysis completed successfully")
    except Exception as e:
        logger.exception(f"Error occurred while solving Problem 1: {str(e)}")
        print(f"❌ Error solving Problem 1: {str(e)}")
        success = False

    # Clean up
    logger.info("Stopping Spark session")
    spark.stop()

    # Overall timing
    overall_end = time.time()
    total_time = overall_end - overall_start
    logger.info(f"Total execution time: {total_time:.2f} seconds")

    print("\n" + "=" * 70)
    if success:
        print("✅ PROBLEM 1 COMPLETED SUCCESSFULLY!")
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print("\nFiles created:")
        print("  - daily_averages_cluster.csv (Problem 1 solution)")
        print("\nNext steps:")
        print("  1. Check daily_averages_cluster.csv for the complete results")
        print("  2. Verify the output matches the expected format")
        print("  3. Submit daily_averages_cluster.csv as part of your solution")
    else:
        print("❌ Problem 1 failed - check error messages above")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
