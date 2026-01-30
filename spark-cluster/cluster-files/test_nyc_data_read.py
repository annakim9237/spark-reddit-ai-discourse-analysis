#!/usr/bin/env python3
"""
NYC TLC Data Reading Test Script

This script tests reading NYC TLC data from S3 and diagnoses any issues.
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException


def test_nyc_data_access():
    """Test NYC TLC data access and diagnose issues."""

    print("Testing NYC TLC Data Access...")
    print("=" * 50)

    try:
        # Create Spark session with AWS support
        print("1. Creating Spark session with AWS support...")
        spark = (
            SparkSession.builder
            .appName("NYCDataTest")
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")
            .config(
                "fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.ContainerCredentialsProvider",
            )
            .getOrCreate()
        )
        print("   ✅ Spark session created successfully")

        # Test different S3 paths to identify the correct one
        test_paths = [
            "s3a://bigdatateaching/nyc_tlc/trip_data/yyyy=2021/*.parquet",
            #"s3a://nyc-tlc/trip data/yellow_tripdata_2021-*.parquet",
            #"s3a://bigdatateaching/nyc_tlc/",
            #"s3a://bigdatateaching/",
            #"s3a://nyc-tlc/",
        ]

        print("\n2. Testing different S3 paths...")

        for i, path in enumerate(test_paths, 1):
            print(f"\n   Test {i}: {path}")
            try:
                # Try to list the path first
                hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
                hadoop_conf.set("fs.s3a.aws.credentials.provider",
                               "com.amazonaws.auth.ContainerCredentialsProvider")

                # Try to read the data
                if path.endswith("*.parquet"):
                    df = spark.read.parquet(path)
                    count = df.count()
                    print(f"      ✅ SUCCESS: Found {count:,} rows")
                    print("      Schema:")
                    df.printSchema()
                    print("      Sample data:")
                    df.show(5)
                    break
                else:
                    # Just try to list directory
                    file_system = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
                        spark.sparkContext._jsc.hadoopConfiguration()
                    )
                    path_obj = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path)
                    if file_system.exists(path_obj):
                        print(f"      ✅ Directory exists: {path}")
                        # List contents
                        files = file_system.listStatus(path_obj)
                        print(f"      Contents ({len(files)} items):")
                        for j, file_status in enumerate(files[:10]):  # Show first 10
                            print(f"        - {file_status.getPath().getName()}")
                        if len(files) > 10:
                            print(f"        ... and {len(files) - 10} more")
                    else:
                        print(f"      ❌ Directory does not exist: {path}")

            except AnalysisException as e:
                error_msg = str(e)
                if "Path does not exist" in error_msg:
                    print(f"      ❌ Path not found: {path}")
                elif "Access Denied" in error_msg:
                    print(f"      ❌ Access denied: {path}")
                elif "NoSuchBucket" in error_msg:
                    print(f"      ❌ Bucket does not exist: {path}")
                else:
                    print(f"      ❌ Analysis error: {error_msg}")
            except Exception as e:
                print(f"      ❌ Unexpected error: {str(e)}")

        # Test the specific problematic code
        print("\n3. Testing the specific problematic code...")
        try:
            print("   Trying: spark.read.parquet with header=True...")
            nyc_tlc = spark.read.parquet(
                "s3a://bigdatateaching/nyc_tlc/trip_data/yyyy=2021/*.parquet",
                header=True
            )
            print("   ✅ Read successful, showing data...")
            nyc_tlc.show()

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            print("\n   Note: 'header=True' is not valid for parquet files.")
            print("   Parquet files are self-describing and don't need header parameter.")
            print("   Try without header parameter...")

            try:
                nyc_tlc = spark.read.parquet(
                    "s3a://bigdatateaching/nyc_tlc/trip_data/yyyy=2021/*.parquet"
                )
                print("   ✅ Read successful without header parameter!")
                nyc_tlc.show()
            except Exception as e2:
                print(f"   ❌ Still failed: {str(e2)}")

        spark.stop()

    except Exception as e:
        print(f"❌ Failed to create Spark session: {str(e)}")
        return False

    return True


def main():
    """Main function to run the test."""
    print("NYC TLC Data Reading Diagnostic Tool")
    print("This will help identify why the data reading is failing.\n")

    success = test_nyc_data_access()

    if success:
        print("\n" + "="*50)
        print("Diagnostic complete. Check results above.")
    else:
        print("\n" + "="*50)
        print("❌ Critical failure in setup.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())