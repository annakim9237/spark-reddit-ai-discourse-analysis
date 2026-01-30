''''
FILE: reddit_bwords_cluster.py
'''
import os
from pyspark.sql import SparkSession
import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from pyspark.ml import Pipeline

MASTER_IP = os.getenv("MASTER_PRIVATE_IP")
print("Using MASTER IP:", MASTER_IP)

spark = SparkSession.builder \
    .appName("NLPTest") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3") \
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    ) \
    .getOrCreate()

df = spark.createDataFrame([("hello spark nlp!",)], ["text"])

doc = DocumentAssembler().setInputCol("text").setOutputCol("doc")
tok = Tokenizer().setInputCols(["doc"]).setOutputCol("token")

pipeline = Pipeline(stages=[doc, tok])
result = pipeline.fit(df).transform(df)

result.show(truncate=False)
