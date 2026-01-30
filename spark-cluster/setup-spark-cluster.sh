#!/bin/bash

################################################################################
# Apache Spark Cluster Automated Setup Script
#
# This script automates the complete setup of a 4-node Spark cluster on AWS EC2:
# - 1 Master node
# - 3 Worker nodes
#
# Usage: ./setup-spark-cluster.sh <LAPTOP_IP>
#   LAPTOP_IP: Your laptop's public IP address (get from https://ipchicken.com/)
#
# Example: ./setup-spark-cluster.sh 73.250.239.127
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

log_step() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Check if laptop IP is provided
if [ $# -ne 1 ]; then
    log_error "Usage: $0 <LAPTOP_IP>"
    log_error "Example: $0 123.45.67.89"
    log_error "Get your laptop IP from https://ipchicken.com/"
    exit 1
fi

LAPTOP_IP=$1
log_info "Laptop IP: $LAPTOP_IP"

# Validate IP format
if ! [[ $LAPTOP_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_error "Invalid IP address format: $LAPTOP_IP"
    exit 1
fi

################################################################################
# Part 1: Prerequisites
################################################################################

log_step "Part 1: Checking Prerequisites"

# Get EC2 instance public IP
log_info "Getting EC2 instance public IP..."
export MY_EC2_IP=$(curl -s https://checkip.amazonaws.com)
log_success "EC2 Instance IP: $MY_EC2_IP"

export MY_LAPTOP_IP=$LAPTOP_IP
log_success "Laptop IP: $MY_LAPTOP_IP"

# Set AWS region
export AWS_REGION=us-east-1
log_info "AWS Region: $AWS_REGION"

# Check AWS CLI
log_info "Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not found. Please install it first."
    exit 1
fi
log_success "AWS CLI found: $(aws --version)"

# Check if cluster-files directory exists
if [ ! -d "cluster-files" ]; then
    log_error "cluster-files directory not found. Please run this script from the lab-spark-cluster directory."
    exit 1
fi
log_success "cluster-files directory found"

################################################################################
# Part 2: Create Security Group
################################################################################

log_step "Part 2: Creating Security Group"

SG_NAME="spark-cluster-sg-$(date +%s)"

# Check if security group with this pattern exists and delete it
log_info "Checking for existing security groups..."
EXISTING_SG=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=spark-cluster-sg-*" \
  --query 'SecurityGroups[0].GroupId' \
  --output text \
  --region $AWS_REGION 2>/dev/null || echo "None")

if [ "$EXISTING_SG" != "None" ] && [ "$EXISTING_SG" != "" ]; then
    log_warn "Found existing security group: $EXISTING_SG"
    log_info "Attempting to delete existing security group..."
    if aws ec2 delete-security-group --group-id $EXISTING_SG --region $AWS_REGION 2>/dev/null; then
        log_success "Deleted existing security group"
        sleep 2
    else
        log_warn "Could not delete existing security group (may be in use)"
    fi
fi

log_info "Creating security group..."
export SPARK_SG_ID=$(aws ec2 create-security-group \
  --group-name $SG_NAME \
  --description "Security group for Spark cluster" \
  --region $AWS_REGION \
  --query 'GroupId' \
  --output text)
log_success "Security Group ID: $SPARK_SG_ID"

# Wait a bit for security group to be ready
sleep 2

log_info "Configuring security group rules..."

# Allow all traffic within the security group
log_info "Allowing intra-cluster communication..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol -1 \
  --source-group $SPARK_SG_ID \
  --region $AWS_REGION > /dev/null
log_success "Intra-cluster communication enabled"

# Allow SSH from EC2 instance
log_info "Allowing SSH from EC2 instance ($MY_EC2_IP)..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "SSH access from EC2 enabled"

# Allow SSH from laptop
log_info "Allowing SSH from laptop ($MY_LAPTOP_IP)..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "SSH access from laptop enabled"

# Allow Spark Web UI from EC2
log_info "Allowing Spark Web UI access from EC2..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Web UI access from EC2 enabled"

# Allow Spark Web UI from laptop
log_info "Allowing Spark Web UI access from laptop..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Web UI access from laptop enabled"

# Allow Spark Application UI from EC2
log_info "Allowing Spark Application UI from EC2..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Application UI from EC2 enabled"

# Allow Spark Application UI from laptop
log_info "Allowing Spark Application UI from laptop..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Application UI from laptop enabled"

################################################################################
# Part 3: Create SSH Key Pair
################################################################################

log_step "Part 3: Creating SSH Key Pair"

KEY_NAME="spark-cluster-key-$(date +%s)"
KEY_FILE="${KEY_NAME}.pem"

# Check if key pairs with this pattern exist and delete them
log_info "Checking for existing key pairs..."
EXISTING_KEYS=$(aws ec2 describe-key-pairs \
  --filters "Name=key-name,Values=spark-cluster-key-*" \
  --query 'KeyPairs[*].KeyName' \
  --output text \
  --region $AWS_REGION 2>/dev/null || echo "")

if [ -n "$EXISTING_KEYS" ]; then
    for KEY in $EXISTING_KEYS; do
        log_warn "Found existing key pair: $KEY"
        log_info "Attempting to delete existing key pair..."
        if aws ec2 delete-key-pair --key-name $KEY --region $AWS_REGION 2>/dev/null; then
            log_success "Deleted existing key pair: $KEY"
        else
            log_warn "Could not delete key pair: $KEY"
        fi
    done
    sleep 1
fi

log_info "Creating key pair: $KEY_NAME..."
aws ec2 create-key-pair \
  --key-name $KEY_NAME \
  --query 'KeyMaterial' \
  --output text \
  --region $AWS_REGION > $KEY_FILE

chmod 400 $KEY_FILE
log_success "Key pair created and saved to $KEY_FILE"

################################################################################
# Part 4: Launch EC2 Instances
################################################################################

log_step "Part 4: Launching EC2 Instances"

# Get Ubuntu 22.04 AMI (free tier eligible, no marketplace subscription needed)
log_info "Finding Ubuntu 22.04 AMI..."
export AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text \
  --region $AWS_REGION)
log_success "AMI ID: $AMI_ID"

# Launch master node
log_info "Launching master node..."
MASTER_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --key-name $KEY_NAME \
  --security-group-ids $SPARK_SG_ID \
  --iam-instance-profile Name=LabInstanceProfile \
  --count 1 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spark-master},{Key=Role,Value=master}]' \
  --region $AWS_REGION \
  --query 'Instances[0].InstanceId' \
  --output text)
log_success "Master instance launched: $MASTER_INSTANCE_ID"

# Launch worker nodes
log_info "Launching 3 worker nodes..."
WORKER_RESERVATION=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --key-name $KEY_NAME \
  --security-group-ids $SPARK_SG_ID \
  --iam-instance-profile Name=LabInstanceProfile \
  --count 3 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spark-worker},{Key=Role,Value=worker}]' \
  --region $AWS_REGION \
  --output json)

WORKER1_INSTANCE_ID=$(echo $WORKER_RESERVATION | jq -r '.Instances[0].InstanceId')
WORKER2_INSTANCE_ID=$(echo $WORKER_RESERVATION | jq -r '.Instances[1].InstanceId')
WORKER3_INSTANCE_ID=$(echo $WORKER_RESERVATION | jq -r '.Instances[2].InstanceId')

log_success "Worker 1 instance launched: $WORKER1_INSTANCE_ID"
log_success "Worker 2 instance launched: $WORKER2_INSTANCE_ID"
log_success "Worker 3 instance launched: $WORKER3_INSTANCE_ID"

# Wait for instances to be running
log_info "Waiting for instances to be in running state..."
aws ec2 wait instance-running \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID $WORKER3_INSTANCE_ID \
  --region $AWS_REGION
log_success "All instances are running"

# Wait a bit more for SSH to be ready
log_info "Waiting 30 seconds for SSH to be ready..."
sleep 30

################################################################################
# Part 5: Get Instance IP Addresses
################################################################################

log_step "Part 5: Getting Instance IP Addresses"

log_info "Retrieving IP addresses..."

export MASTER_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $MASTER_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export MASTER_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $MASTER_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER1_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER1_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER1_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER1_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER2_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER2_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER2_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER2_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER3_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER3_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER3_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER3_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

log_success "Master: $MASTER_PUBLIC_IP (public) / $MASTER_PRIVATE_IP (private)"
log_success "Worker 1: $WORKER1_PUBLIC_IP (public) / $WORKER1_PRIVATE_IP (private)"
log_success "Worker 2: $WORKER2_PUBLIC_IP (public) / $WORKER2_PRIVATE_IP (private)"
log_success "Worker 3: $WORKER3_PUBLIC_IP (public) / $WORKER3_PRIVATE_IP (private)"

# Save cluster configuration
log_info "Saving cluster configuration..."
cat > cluster-config.txt <<EOF
# Cluster Configuration - Created $(date)
AWS_REGION=$AWS_REGION
SPARK_SG_ID=$SPARK_SG_ID
KEY_NAME=$KEY_NAME
KEY_FILE=$KEY_FILE

# Instance IDs
MASTER_INSTANCE_ID=$MASTER_INSTANCE_ID
WORKER1_INSTANCE_ID=$WORKER1_INSTANCE_ID
WORKER2_INSTANCE_ID=$WORKER2_INSTANCE_ID
WORKER3_INSTANCE_ID=$WORKER3_INSTANCE_ID

# IP Addresses
MASTER_PUBLIC_IP=$MASTER_PUBLIC_IP
MASTER_PRIVATE_IP=$MASTER_PRIVATE_IP
WORKER1_PUBLIC_IP=$WORKER1_PUBLIC_IP
WORKER1_PRIVATE_IP=$WORKER1_PRIVATE_IP
WORKER2_PUBLIC_IP=$WORKER2_PUBLIC_IP
WORKER2_PRIVATE_IP=$WORKER2_PRIVATE_IP
WORKER3_PUBLIC_IP=$WORKER3_PUBLIC_IP
WORKER3_PRIVATE_IP=$WORKER3_PRIVATE_IP
EOF

# Save IP addresses for copying to nodes
cat > cluster-ips.txt <<EOF
MASTER_PUBLIC_IP=$MASTER_PUBLIC_IP
MASTER_PRIVATE_IP=$MASTER_PRIVATE_IP
WORKER1_PUBLIC_IP=$WORKER1_PUBLIC_IP
WORKER1_PRIVATE_IP=$WORKER1_PRIVATE_IP
WORKER2_PUBLIC_IP=$WORKER2_PUBLIC_IP
WORKER2_PRIVATE_IP=$WORKER2_PRIVATE_IP
WORKER3_PUBLIC_IP=$WORKER3_PUBLIC_IP
WORKER3_PRIVATE_IP=$WORKER3_PRIVATE_IP
EOF

log_success "Configuration saved to cluster-config.txt and cluster-ips.txt"

# Create SSH helper script for master node
log_info "Creating SSH helper script..."
cat > ssh_to_master_node.sh <<EOF
#!/bin/bash
# SSH to Spark Master Node
# Generated on $(date)
ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP
EOF

chmod +x ssh_to_master_node.sh
log_success "SSH helper script created: ssh_to_master_node.sh"

################################################################################
# Part 6: Copy Files to All Nodes
################################################################################

log_step "Part 6: Copying Files to All Nodes"

SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Function to copy files to a node
copy_files_to_node() {
    local NODE_NAME=$1
    local NODE_IP=$2

    log_info "Copying files to $NODE_NAME ($NODE_IP)..."
    ssh $SSH_OPTS ubuntu@$NODE_IP "mkdir -p ~/spark-cluster" 2>/dev/null
    scp -r $SSH_OPTS cluster-files/* ubuntu@$NODE_IP:~/spark-cluster/ 2>/dev/null
    scp $SSH_OPTS cluster-ips.txt ubuntu@$NODE_IP:~/spark-cluster/ 2>/dev/null
    log_success "Files copied to $NODE_NAME"
}

copy_files_to_node "Master" $MASTER_PUBLIC_IP
copy_files_to_node "Worker 1" $WORKER1_PUBLIC_IP
copy_files_to_node "Worker 2" $WORKER2_PUBLIC_IP
copy_files_to_node "Worker 3" $WORKER3_PUBLIC_IP

################################################################################
# Part 7: Setup Master Node
################################################################################

log_step "Part 7: Setting Up Master Node"

log_info "Configuring master node..."

ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'bash -s' <<'MASTER_SETUP'
set -e

echo "[Master Setup] Installing system packages..."
sudo apt update -qq
sudo apt install -y openjdk-17-jdk-headless python3-pip python3-venv curl unzip > /dev/null 2>&1

echo "[Master Setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

echo "[Master Setup] Installing Python dependencies..."
cd ~/spark-cluster
uv sync > /dev/null 2>&1

echo "[Master Setup] Downloading Spark 4.0.1..."
cd ~
wget -q https://downloads.apache.org/spark/spark-4.0.1/spark-4.0.1-bin-hadoop3.tgz
tar -xzf spark-4.0.1-bin-hadoop3.tgz
mv spark-4.0.1-bin-hadoop3 spark
rm spark-4.0.1-bin-hadoop3.tgz

echo "[Master Setup] Downloading AWS JARs..."
cd ~/spark/jars
wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar
wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
wget -q https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.29.27/bundle-2.29.27.jar
wget -q https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.29.27/url-connection-client-2.29.27.jar

echo "[Master Setup] Configuring environment variables..."
cat >> ~/.bashrc <<'EOF'
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=$HOME/spark
export PATH="$HOME/.local/bin:$PATH"
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PS1="[MASTER] \u@\h:\w\$ "
EOF

source ~/.bashrc

echo "[Master Setup] Master node setup complete!"
MASTER_SETUP

log_success "Master node configured"

################################################################################
# Part 8: Setup Worker Nodes
################################################################################

log_step "Part 8: Setting Up Worker Nodes"

# Function to setup a worker node
setup_worker() {
    local WORKER_NUM=$1
    local WORKER_IP=$2

    log_info "Configuring Worker $WORKER_NUM ($WORKER_IP)..."

    ssh $SSH_OPTS ubuntu@$WORKER_IP "bash -s $WORKER_NUM" <<'WORKER_SETUP'
set -e

WORKER_NUM=$1

echo "[Worker $WORKER_NUM Setup] Installing system packages..."
sudo apt update -qq
sudo apt install -y openjdk-17-jdk-headless python3-pip python3-venv curl unzip > /dev/null 2>&1

echo "[Worker $WORKER_NUM Setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

echo "[Worker $WORKER_NUM Setup] Installing Python dependencies..."
cd ~/spark-cluster
uv sync > /dev/null 2>&1

echo "[Worker $WORKER_NUM Setup] Downloading Spark 4.0.1..."
cd ~
wget -q https://downloads.apache.org/spark/spark-4.0.1/spark-4.0.1-bin-hadoop3.tgz
tar -xzf spark-4.0.1-bin-hadoop3.tgz
mv spark-4.0.1-bin-hadoop3 spark
rm spark-4.0.1-bin-hadoop3.tgz

echo "[Worker $WORKER_NUM Setup] Downloading AWS JARs..."
cd ~/spark/jars
wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar
wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
wget -q https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.29.27/bundle-2.29.27.jar
wget -q https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.29.27/url-connection-client-2.29.27.jar

echo "[Worker $WORKER_NUM Setup] Configuring environment variables..."
cat >> ~/.bashrc <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=\$HOME/spark
export PATH="\$HOME/.local/bin:\$PATH"
export PATH=\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PS1="[WORKER-$WORKER_NUM] \u@\h:\w\$ "
EOF

source ~/.bashrc

echo "[Worker $WORKER_NUM Setup] Worker node setup complete!"
WORKER_SETUP

    log_success "Worker $WORKER_NUM configured"
}

setup_worker 1 $WORKER1_PUBLIC_IP &
setup_worker 2 $WORKER2_PUBLIC_IP &
setup_worker 3 $WORKER3_PUBLIC_IP &

wait
log_success "All worker nodes configured"

################################################################################
# Part 9: Configure Spark on Master
################################################################################

log_step "Part 9: Configuring Spark on Master Node"

log_info "Creating Spark configuration files on master..."

ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "bash -s" <<SPARK_CONFIG
set -e

source ~/spark-cluster/cluster-ips.txt

echo "[Spark Config] Creating spark-env.sh..."
cat > \$HOME/spark/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_MASTER_HOST=\$MASTER_PRIVATE_IP
export SPARK_MASTER_PORT=7077
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x \$HOME/spark/conf/spark-env.sh

echo "[Spark Config] Creating workers file..."
cat > \$HOME/spark/conf/workers <<EOF
\$WORKER1_PRIVATE_IP
\$WORKER2_PRIVATE_IP
\$WORKER3_PRIVATE_IP
EOF

echo "[Spark Config] Spark configuration complete!"
SPARK_CONFIG

log_success "Spark configured on master"

################################################################################
# Part 10: Configure Spark on Workers
################################################################################

log_step "Part 10: Configuring Spark on Worker Nodes"

# Function to configure Spark on worker
configure_worker_spark() {
    local WORKER_NUM=$1
    local WORKER_IP=$2

    log_info "Configuring Spark on Worker $WORKER_NUM..."

    ssh $SSH_OPTS ubuntu@$WORKER_IP 'bash -s' <<'WORKER_SPARK_CONFIG'
set -e

source ~/spark-cluster/cluster-ips.txt

cat > $HOME/spark/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x $HOME/spark/conf/spark-env.sh
WORKER_SPARK_CONFIG

    log_success "Spark configured on Worker $WORKER_NUM"
}

configure_worker_spark 1 $WORKER1_PUBLIC_IP &
configure_worker_spark 2 $WORKER2_PUBLIC_IP &
configure_worker_spark 3 $WORKER3_PUBLIC_IP &

wait
log_success "Spark configured on all workers"

################################################################################
# Part 11: Setup SSH Keys for Passwordless Access
################################################################################

log_step "Part 11: Setting Up SSH Keys for Passwordless Access"

log_info "Copying SSH key to master node..."
scp $SSH_OPTS $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/.ssh/ 2>/dev/null
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "chmod 400 ~/.ssh/$KEY_FILE" 2>/dev/null
log_success "SSH key copied to master"

log_info "Generating SSH key on master..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'bash -s' <<'SSH_SETUP'
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa -q
fi
SSH_SETUP
log_success "SSH key generated on master"

# Get master's public key
MASTER_PUB_KEY=$(ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "cat ~/.ssh/id_rsa.pub")

log_info "Adding master's key to authorized_keys on all workers..."
for WORKER_IP in $WORKER1_PUBLIC_IP $WORKER2_PUBLIC_IP $WORKER3_PUBLIC_IP; do
    ssh $SSH_OPTS ubuntu@$WORKER_IP "echo '$MASTER_PUB_KEY' >> ~/.ssh/authorized_keys" 2>/dev/null
done
log_success "Passwordless SSH configured"

################################################################################
# Part 12: Start Spark Cluster
################################################################################

log_step "Part 12: Starting Spark Cluster"

log_info "Starting Spark master..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP '$HOME/spark/sbin/start-master.sh' 2>/dev/null
sleep 5
log_success "Spark master started"

log_info "Starting Spark workers..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP '$HOME/spark/sbin/start-workers.sh' 2>/dev/null
sleep 5
log_success "Spark workers started"

################################################################################
# Part 13: Verify Cluster
################################################################################

log_step "Part 13: Verifying Cluster"

log_info "Checking master process..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'jps | grep Master' 2>/dev/null
log_success "Master process running"

log_info "Checking worker processes..."
WORKER_COUNT=0
for WORKER_IP in $WORKER1_PUBLIC_IP $WORKER2_PUBLIC_IP $WORKER3_PUBLIC_IP; do
    if ssh $SSH_OPTS ubuntu@$WORKER_IP 'jps | grep Worker' 2>/dev/null; then
        ((WORKER_COUNT++))
    fi
done
log_success "Workers running: $WORKER_COUNT/3"

################################################################################
# Final Summary
################################################################################

log_step "✓ Spark Cluster Setup Complete!"

echo ""
echo "=================================================="
echo "           CLUSTER INFORMATION"
echo "=================================================="
echo ""
echo "Master Node:"
echo "  Public IP:  $MASTER_PUBLIC_IP"
echo "  Private IP: $MASTER_PRIVATE_IP"
echo ""
echo "Worker Nodes:"
echo "  Worker 1: $WORKER1_PUBLIC_IP (private: $WORKER1_PRIVATE_IP)"
echo "  Worker 2: $WORKER2_PUBLIC_IP (private: $WORKER2_PRIVATE_IP)"
echo "  Worker 3: $WORKER3_PUBLIC_IP (private: $WORKER3_PRIVATE_IP)"
echo ""
echo "=================================================="
echo "           ACCESS INFORMATION"
echo "=================================================="
echo ""
echo "SSH to Master:"
echo "  ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP"
echo ""
echo "Or use the helper script:"
echo "  ./ssh_to_master_node.sh"
echo ""
echo "Spark Master Web UI:"
echo "  http://$MASTER_PUBLIC_IP:8080"
echo ""
echo "Spark Application UI (when job running):"
echo "  http://$MASTER_PUBLIC_IP:4040"
echo ""
echo "=================================================="
echo "           CONFIGURATION FILES"
echo "=================================================="
echo ""
echo "Cluster configuration: cluster-config.txt"
echo "IP addresses: cluster-ips.txt"
echo "SSH key: $KEY_FILE"
echo ""
echo "=================================================="
echo "           NEXT STEPS"
echo "=================================================="
echo ""
echo "1. Access Spark Master Web UI to verify 3 workers are connected"
echo "2. SSH to master and run a test job:"
echo "   ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP"
echo "   cd ~/spark-cluster"
echo "   source cluster-ips.txt"
echo "   uv run python nyc_tlc_problem1_cluster.py spark://\$MASTER_PRIVATE_IP:7077"
echo ""
echo "3. To stop the cluster later:"
echo "   ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP '\$SPARK_HOME/sbin/stop-all.sh'"
echo ""
echo "4. To terminate all resources, use the cleanup script or:"
echo "   source cluster-config.txt"
echo "   aws ec2 terminate-instances --instance-ids \$MASTER_INSTANCE_ID \$WORKER1_INSTANCE_ID \$WORKER2_INSTANCE_ID \$WORKER3_INSTANCE_ID --region \$AWS_REGION"
echo "   aws ec2 delete-security-group --group-id \$SPARK_SG_ID --region \$AWS_REGION"
echo "   aws ec2 delete-key-pair --key-name \$KEY_NAME --region \$AWS_REGION"
echo ""
log_success "Setup script completed successfully!"
