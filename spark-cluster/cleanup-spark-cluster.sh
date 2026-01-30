#!/bin/bash

################################################################################
# Spark Cluster Cleanup Script
#
# This script terminates all resources created by setup-spark-cluster.sh
#
# Usage: ./cleanup-spark-cluster.sh
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if cluster-config.txt exists
if [ ! -f "cluster-config.txt" ]; then
    log_error "cluster-config.txt not found. Cannot proceed with cleanup."
    log_info "If you want to clean up manually, you'll need to find the resources by tags."
    exit 1
fi

# Load cluster configuration
log_info "Loading cluster configuration..."
source cluster-config.txt
log_success "Configuration loaded"

echo ""
echo "=================================================="
echo "           RESOURCES TO BE DELETED"
echo "=================================================="
echo ""
echo "Region: $AWS_REGION"
echo "Security Group: $SPARK_SG_ID"
echo "Key Pair: $KEY_NAME"
echo ""
echo "Instances:"
echo "  Master: $MASTER_INSTANCE_ID ($MASTER_PUBLIC_IP)"
echo "  Worker 1: $WORKER1_INSTANCE_ID ($WORKER1_PUBLIC_IP)"
echo "  Worker 2: $WORKER2_INSTANCE_ID ($WORKER2_PUBLIC_IP)"
echo "  Worker 3: $WORKER3_INSTANCE_ID ($WORKER3_PUBLIC_IP)"
echo ""
echo "=================================================="
echo ""

read -p "Are you sure you want to delete all these resources? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    log_info "Cleanup cancelled"
    exit 0
fi

log_info "Starting cleanup..."

# Stop Spark cluster first
log_info "Stopping Spark cluster..."
if [ -f "$KEY_FILE" ]; then
    ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        ubuntu@$MASTER_PUBLIC_IP '$HOME/spark/sbin/stop-all.sh' 2>/dev/null || true
    log_success "Spark cluster stopped"
fi

# Terminate instances
log_info "Terminating EC2 instances..."
aws ec2 terminate-instances \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID $WORKER3_INSTANCE_ID \
  --region $AWS_REGION > /dev/null
log_success "Instance termination initiated"

# Wait for instances to terminate
log_info "Waiting for instances to terminate (this may take a few minutes)..."
aws ec2 wait instance-terminated \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID $WORKER3_INSTANCE_ID \
  --region $AWS_REGION
log_success "All instances terminated"

# Delete security group
log_info "Deleting security group..."
aws ec2 delete-security-group \
  --group-id $SPARK_SG_ID \
  --region $AWS_REGION
log_success "Security group deleted"

# Delete key pair
log_info "Deleting key pair..."
aws ec2 delete-key-pair \
  --key-name $KEY_NAME \
  --region $AWS_REGION
log_success "Key pair deleted"

# Remove local files
log_info "Removing local configuration files..."
rm -f $KEY_FILE
rm -f cluster-config.txt
rm -f cluster-ips.txt
log_success "Local files cleaned up"

echo ""
echo "=================================================="
log_success "Cleanup complete! All resources have been deleted."
echo "=================================================="
