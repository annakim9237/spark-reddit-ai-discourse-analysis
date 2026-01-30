#!/usr/bin/env bash
set -euo pipefail

################################################################################
# run-user-segmentation.sh
#
# Reddit User Segmentation Pipeline (Cluster-style wrapper)
#
# Run from your LOCAL machine (repo root), like:
#   ./spark-cluster/cluster-files/ml/run-user-segmentation.sh
################################################################################

# ----------------------------
# REMOTE + LOCAL PATHS
# ----------------------------
REMOTE_BASE="~/spark-cluster"
REMOTE_ML_DIR="~/spark-cluster/cluster-files/ml"

REMOTE_USER_FEATURES="data/csv/reddit_user_topic_features.csv"
REMOTE_CLUSTERS="data/csv/reddit_user_clusters_k10.csv"
REMOTE_TSNE="data/plots/reddit_user_clusters_tsne_k10.png"
REMOTE_SUMMARY="data/csv/reddit_user_clusters_k10_summary.csv"


LOCAL_USER_FEATURES="data/csv/reddit_user_topic_features.csv"
LOCAL_CLUSTERS="data/csv/reddit_user_clusters_k10.csv"
LOCAL_TSNE="data/plots/reddit_user_clusters_tsne_k10.png"
LOCAL_SUMMARY="data/csv/reddit_user_clusters_k10_summary.csv"


# ----------------------------
# STEP 0 — MUST RUN LOCALLY
# ----------------------------
echo "🔹 Loading cluster config..."
source cluster-config.txt   # needs KEY_FILE and MASTER_PUBLIC_IP

echo "🔹 Ensuring ML directory exists on cluster..."
ssh -i "$KEY_FILE" ubuntu@"$MASTER_PUBLIC_IP" \
  "mkdir -p $REMOTE_ML_DIR"

echo "🔹 Copying ML scripts to cluster..."
scp -i "$KEY_FILE" \
  cluster-files/ml/reddit_nlp_cleaning.py \
  cluster-files/ml/reddit_user_prepare_features.py \
  cluster-files/ml/reddit_user_cluster_kmeans_tsne.py \
  ubuntu@"$MASTER_PUBLIC_IP":$REMOTE_ML_DIR/

echo "-----------------------------------------"
echo "  SSH into cluster and run pipeline ..."
echo "-----------------------------------------"

ssh -i "$KEY_FILE" ubuntu@"$MASTER_PUBLIC_IP" << 'EOF'
set -euo pipefail
cd ~/spark-cluster

echo "🔹 Loading cluster-ips.txt"
source cluster-ips.txt

MASTER_URL="spark://$MASTER_PRIVATE_IP:7077"

# Where we will store the cleaned parquet
CLEANED_PARQUET="data/parquet/reddit_cleaned.parquet"

echo "========================================="
echo "  AUTO-START SPARK CLUSTER"
echo "========================================="
/home/ubuntu/spark/sbin/stop-worker.sh || true
/home/ubuntu/spark/sbin/stop-master.sh || true

echo "🔹 Starting Spark Master..."
/home/ubuntu/spark/sbin/start-master.sh
sleep 5

echo "🔹 Starting Spark Worker..."
/home/ubuntu/spark/sbin/start-worker.sh "spark://$MASTER_PRIVATE_IP:7077"
sleep 5

echo "✓ Spark master + worker started"
jps || true

echo "========================================="
echo "  USER SEGMENTATION PIPELINE"
echo "========================================="

# Ensure output directories exist
mkdir -p data/csv
mkdir -p data/plots
mkdir -p data/parquet

# 0) If cleaned parquet is missing, run cleaning once
if [ ! -d "$CLEANED_PARQUET" ] && [ ! -f "$CLEANED_PARQUET" ]; then
  echo "🔹 $CLEANED_PARQUET missing — running reddit_nlp_cleaning.py ..."
  set -x
  uv run python cluster-files/ml/reddit_nlp_cleaning.py \
      "$MASTER_URL" \
      --s3-input s3a://ea973-dsan6000-datasets-final/project/reddit/parquet/ \
      --data-type comments \
      --sample 0.02 \
      --output-parquet "$CLEANED_PARQUET"
  set +x
else
  echo "🔹 Found existing cleaned parquet at: $CLEANED_PARQUET"
fi

echo "🔍 Listing parquet target:"
ls -lh data/parquet || true

# Define output paths (on MASTER)
USER_FEATURES_OUT="data/csv/reddit_user_topic_features.csv"
CLUSTERS_OUT="data/csv/reddit_user_clusters_k10.csv"
TSNE_PLOT_OUT="data/plots/reddit_user_clusters_tsne_k10.png"

echo "🔹 STEP 1: Build user topic feature vectors from $CLEANED_PARQUET ..."
set -x
uv run python cluster-files/ml/reddit_user_prepare_features.py \
    --input-parquet "$CLEANED_PARQUET" \
    --text-column "processed_text" \
    --author-column "author" \
    --num-topics 15 \
    --max-comments 200000 \
    --output-user-features "$USER_FEATURES_OUT"
set +x

echo "🔹 STEP 2: Cluster users + 2D t-SNE visualization..."
set -x
uv run python cluster-files/ml/reddit_user_cluster_kmeans_tsne.py \
    --input-user-features "$USER_FEATURES_OUT" \
    --n-clusters 10 \
    --output-clusters-csv "$CLUSTERS_OUT" \
    --output-tsne-plot "$TSNE_PLOT_OUT"
set +x

echo "✨ User segmentation pipeline finished on master"
EOF

# ----------------------------
# STEP 3 — COPY RESULTS LOCALLY
# ----------------------------
echo "========================================="
echo "  STEP 3 — Download user segmentation outputs"
echo "========================================="

mkdir -p "$(dirname "$LOCAL_USER_FEATURES")"
mkdir -p "$(dirname "$LOCAL_TSNE")"

scp -i "$KEY_FILE" ubuntu@"$MASTER_PUBLIC_IP":$REMOTE_BASE/$REMOTE_USER_FEATURES "$LOCAL_USER_FEATURES"
scp -i "$KEY_FILE" ubuntu@"$MASTER_PUBLIC_IP":$REMOTE_BASE/$REMOTE_CLUSTERS      "$LOCAL_CLUSTERS"
scp -i "$KEY_FILE" ubuntu@"$MASTER_PUBLIC_IP":$REMOTE_BASE/$REMOTE_TSNE          "$LOCAL_TSNE"
scp -i "$KEY_FILE" ubuntu@"$MASTER_PUBLIC_IP":$REMOTE_BASE/$REMOTE_SUMMARY       "$LOCAL_SUMMARY"


echo "🎉 User segmentation finished!"
echo "Local CSV (features) → $LOCAL_USER_FEATURES"
echo "Local CSV (clusters) → $LOCAL_CLUSTERS"
echo "Local plot           → $LOCAL_TSNE"
