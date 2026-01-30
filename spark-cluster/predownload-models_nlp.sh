#!/bin/bash

################################################################################
# Pre-download and Distribute Spark NLP Models
#
# Downloads models locally first, then distributes to all cluster nodes
# This avoids corrupted downloads from parallel Spark workers
#
# Usage: ./predownload-models_nlp.sh
################################################################################

#!/bin/bash

################################################################################
# Pre-download and Distribute Spark NLP Models
#
# Downloads models locally first, then distributes to all cluster nodes
# This avoids corrupted downloads from parallel Spark workers.
#
# Usage (from ~/spark-cluster):
#   chmod +x predownload-models_nlp.sh
#   ./predownload-models_nlp.sh
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ---------------------------------------------------------------------------
# 0. Basic checks and setup
# ---------------------------------------------------------------------------

# Check if cluster-config.txt exists
if [ ! -f "cluster-config.txt" ]; then
    log_error "cluster-config.txt not found in current directory"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load cluster configuration
# Expecting: KEY_FILE, MASTER_PUBLIC_IP, WORKER1_PUBLIC_IP, WORKER2_PUBLIC_IP, WORKER3_PUBLIC_IP
source cluster-config.txt

# Convert KEY_FILE to absolute path if it's relative
if [[ "$KEY_FILE" != /* ]]; then
    KEY_FILE="$SCRIPT_DIR/$KEY_FILE"
fi

# SSH options
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Hard-coded cache paths (Spark NLP cache naming)
VIVEKN_DIR=~/cache_pretrained/sentiment_vivekn_en_2.0.2_2.4_1556663184035
LEMMA_DIR=~/cache_pretrained/lemma_antbnc_en_2.0.2_2.4_1556480454569

echo ""
log_info "Pre-downloading Spark NLP models locally on MASTER..."
echo ""

# ---------------------------------------------------------------------------
# 1. Create temporary Python script to download models
# ---------------------------------------------------------------------------

cat > /tmp/download_models.py << 'PYTHON_SCRIPT'
import sparknlp
from sparknlp.annotator import (
    ViveknSentimentModel,
    LemmatizerModel,
    UniversalSentenceEncoder,
    ClassifierDLModel,
)
from pyspark.sql import SparkSession


# Create local Spark session for downloading
spark = (
    SparkSession.builder
    .appName("ModelDownloader")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3")
    .getOrCreate()
)

print("Downloading Vivekn Sentiment model...")
_ = ViveknSentimentModel.pretrained(name="sentiment_vivekn", lang="en")
print("✓ Vivekn Sentiment model downloaded.")

print("Downloading Lemmatizer: lemma_antbnc...")
_ = LemmatizerModel.pretrained("lemma_antbnc", "en")
print("✓ Lemmatizer model downloaded.")


spark.stop()
PYTHON_SCRIPT

# ---------------------------------------------------------------------------
# 2. Run Python downloader via uv
# ---------------------------------------------------------------------------

log_info "Downloading models to local machine (MASTER)..."
cd "$SCRIPT_DIR"
uv run python /tmp/download_models.py

# ---------------------------------------------------------------------------
# 3. Verify both models (including metadata/part-00000) before tarball
# ---------------------------------------------------------------------------

log_info "Verifying local cache structure for Vivekn + Lemma..."

if [ ! -f "$VIVEKN_DIR/metadata/part-00000" ]; then
    log_error "Vivekn model cache is incomplete or missing (no $VIVEKN_DIR/metadata/part-00000)"
    exit 1
fi

if [ ! -f "$LEMMA_DIR/metadata/part-00000" ]; then
    log_error "Lemmatizer model cache is incomplete or missing (no $LEMMA_DIR/metadata/part-00000)"
    exit 1
fi

log_success "Models downloaded to ~/cache_pretrained/ with valid metadata:"
log_info "  - $VIVEKN_DIR"
log_info "  - $LEMMA_DIR"
echo ""

# ---------------------------------------------------------------------------
# 4. Create tarball of models from MASTER
# ---------------------------------------------------------------------------

log_info "Creating tarball of models from MASTER cache_pretrained..."
cd ~
tar -czf spark_nlp_models.tar.gz cache_pretrained/
log_success "Tarball created: ~/spark_nlp_models.tar.gz"
echo ""

# Return to the script directory
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 5. Function to distribute models to a node
# ---------------------------------------------------------------------------

distribute_to_node() {
    local NODE_NAME=$1
    local NODE_IP=$2

    log_info "Distributing models to $NODE_NAME ($NODE_IP)..."

    # Copy tarball to node
    scp $SSH_OPTS ~/spark_nlp_models.tar.gz ubuntu@$NODE_IP:~/

    # Extract on node and verify both Vivekn + Lemma metadata
    ssh $SSH_OPTS ubuntu@$NODE_IP 'bash -s' << 'EXTRACT'
set -e

VIVEKN_DIR=~/cache_pretrained/sentiment_vivekn_en_2.0.2_2.4_1556663184035
LEMMA_DIR=~/cache_pretrained/lemma_antbnc_en_2.0.2_2.4_1556480454569

# Remove old cache if exists
rm -rf ~/cache_pretrained

# Extract models
tar -xzf ~/spark_nlp_models.tar.gz

# Clean up tarball
rm ~/spark_nlp_models.tar.gz

# Verify extraction (Vivekn + Lemma have metadata)
if [ -f "$VIVEKN_DIR/metadata/part-00000" ] && [ -f "$LEMMA_DIR/metadata/part-00000" ]; then
    echo "[REMOTE] Models extracted successfully (Vivekn + lemma with metadata)"
    ls "$VIVEKN_DIR/metadata" "$LEMMA_DIR/metadata"
else
    echo "[REMOTE] ERROR: Model extraction failed (missing Vivekn or lemma metadata)"
    exit 1
fi
EXTRACT

    log_success "$NODE_NAME models installed and verified"
}

# ---------------------------------------------------------------------------
# 6. Distribute to all nodes (MASTER + WORKERS)
# ---------------------------------------------------------------------------

# Install on MASTER itself (overwrite just to be fully consistent)
log_info "Installing models on MASTER (self)..."
cd ~
rm -rf ~/cache_pretrained
tar -xzf ~/spark_nlp_models.tar.gz

if [ -f "$VIVEKN_DIR/metadata/part-00000" ] && [ -f "$LEMMA_DIR/metadata/part-00000" ]; then
    log_success "MASTER cache_pretrained verified (Vivekn + lemma with metadata)"
else
    log_error "MASTER extraction failed – missing Vivekn or lemma metadata"
    exit 1
fi

# Now push to workers
distribute_to_node "Worker 1" "$WORKER1_PUBLIC_IP"
distribute_to_node "Worker 2" "$WORKER2_PUBLIC_IP"
distribute_to_node "Worker 3" "$WORKER3_PUBLIC_IP"

# ---------------------------------------------------------------------------
# 7. Clean up local tarball
# ---------------------------------------------------------------------------

rm ~/spark_nlp_models.tar.gz

echo ""
log_success "All models distributed to cluster!"
echo ""
log_info "Models installed on all nodes at:"
log_info "  - $VIVEKN_DIR"
log_info "  - $LEMMA_DIR"


echo ""
log_info "Now you can run cluster jobs without Spark NLP downloading models at runtime."
log_info "Example:"
echo "  ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP"
echo "  cd ~/spark-cluster"
echo "  source cluster-ips.txt"
echo "  uv run python reddit_nlp_statista_cluster.py spark://\$MASTER_PRIVATE_IP:7077 --data-type comments --benefits ai-tools-ways-of-coding-and-development-enhancements-globally-2024.xlsx --sample 0.3"
echo ""
