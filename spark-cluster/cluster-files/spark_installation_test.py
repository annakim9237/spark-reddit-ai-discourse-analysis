#!/usr/bin/env python3
"""
Spark Installation Test Script

This script tests the Spark installation to ensure everything is working correctly.
Run this after completing the environment setup to verify your installation.
"""

import sys
from pyspark.sql import SparkSession


def test_spark_installation():
    """Test Spark installation and basic functionality."""

    print("Testing Spark Installation...")
    print("=" * 50)

    try:
        # Test 1: Create Spark session with AWS support
        print("1. Creating Spark session with AWS support...")
        spark = (
            SparkSession.builder
            .appName("SparkInstallationTest")
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")
            .config(
                "fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.ContainerCredentialsProvider",
            )
            .getOrCreate()
        )
        print(f"   ✅ Spark session created successfully")
        print(f"   ✅ Spark version: {spark.version}")
        print(f"   ✅ Java version: {spark.sparkContext._jvm.System.getProperty('java.version')}")

        # Test 2: Basic DataFrame operations
        print("\n2. Testing basic DataFrame operations...")
        df = spark.range(1000)
        count = df.count()
        print(f"   ✅ Created DataFrame with {count} rows")

        # Test 3: SQL operations
        print("\n3. Testing SQL operations...")
        df.createOrReplaceTempView("test_table")
        result = spark.sql("SELECT COUNT(*) as total FROM test_table").collect()[0]['total']
        print(f"   ✅ SQL query executed successfully: {result} rows")

        # Test 4: Test data transformations
        print("\n4. Testing data transformations...")
        transformed_df = df.filter(df.id % 2 == 0).select(df.id * 2)
        transformed_count = transformed_df.count()
        print(f"   ✅ Data transformation successful: {transformed_count} rows after filtering")

        # Clean up
        spark.stop()
        print("\nAll tests passed! Spark installation is working correctly.")
        print("\nYou can now proceed with the lab exercises.")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("\nPlease check your installation and try again.")
        return False


if __name__ == "__main__":
    success = test_spark_installation()
    sys.exit(0 if success else 1)
