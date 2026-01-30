# Apache Spark Cluster Setup on AWS EC2 for lab-spark-on-ec2
*Created: 2025-10-04*
*Updated for PySpark 4.x with Java 17 and uv package management*

## Overview
This guide provides step-by-step instructions for setting up a 3-node Apache Spark cluster on AWS EC2 using t3.large instances. The cluster will be configured to run the code from the `lab-spark-on-ec2` repository.

The cluster will consist of:
- 1 Master node (Spark Master)
- 2 Worker nodes (Spark Workers)

Each node will have:
- Java 17 (required for PySpark 4.x)
- Python 3.10+
- uv package manager
- PySpark 4.x and all dependencies
- The lab-spark-on-ec2 repository

## Prerequisites
- AWS Account with appropriate permissions
- EC2 instance (Linux) to run these commands from
- Basic understanding of terminal/command line

---

## Part 1: AWS CLI Setup and Configuration

### Step 1.1: Install AWS CLI (if not already installed)

First, install unzip if not already installed:
```bash
sudo apt install unzip -y
```

Then download and install AWS CLI:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Verify installation:**
```bash
aws --version
```

### Step 1.2: Set Your Region as Environment Variable

Since you're running on an EC2 instance with an IAM role, AWS credentials are automatically provided. You only need to set the region:

```bash
export AWS_REGION=us-east-1
echo "export AWS_REGION=us-east-1" >> ~/.bashrc
```

---

## Part 2: Network Setup (VPC and Security Group)

### Step 2.1: Create a Security Group

Create a security group for the Spark cluster and automatically save the ID:
```bash
export SPARK_SG_ID=$(aws ec2 create-security-group \
  --group-name spark-cluster-sg \
  --description "Security group for Spark cluster" \
  --region $AWS_REGION \
  --query 'GroupId' \
  --output text)

echo "Security Group ID: $SPARK_SG_ID"
```

The Security Group ID will be automatically captured and displayed.

**Verify the security group was created:**
```bash
aws ec2 describe-security-groups \
  --region $AWS_REGION \
  --query 'SecurityGroups[*].[GroupName,GroupId,Description]' \
  --output table
```

Look for `spark-cluster-sg` in the output - this is the security group we just created!

### Step 2.2: Configure Security Group Rules

**Get IP addresses for access control:**

We need two IP addresses:
1. **This EC2 instance IP** - for SSH access to cluster nodes from this management instance
2. **Your laptop IP** - for SSH and Web UI access from your laptop

```bash
# Get this EC2 instance's public IP
export MY_EC2_IP=$(curl -s https://checkip.amazonaws.com)
echo "My EC2 Instance IP: $MY_EC2_IP"

# Get your laptop IP from ipchicken.com and set it manually
export MY_LAPTOP_IP=[your IP from ipchicken.com]
echo "My Laptop IP: $MY_LAPTOP_IP"
```

**IMPORTANT:** Visit https://ipchicken.com/ from your laptop to get your laptop's public IP address.

**Allow all traffic within the security group (for cluster communication):**
```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol -1 \
  --source-group $SPARK_SG_ID \
  --region $AWS_REGION
```

**Allow SSH access (port 22) from both EC2 instance and laptop:**
```bash
# Allow SSH from this EC2 instance
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION

# Allow SSH from your laptop
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION
```

**Allow Spark Web UI access (ports 8080, 8081) from both EC2 instance and laptop:**

**SECURITY NOTE:** These rules restrict access to **only your EC2 instance and laptop IP addresses** (not the entire internet).

```bash
# Allow Spark Master UI (8080) and Worker UI (8081) from EC2 instance
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION

# Allow Spark Master UI (8080) and Worker UI (8081) from laptop
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION
```

**Allow Spark Application UI (port 4040) from both EC2 instance and laptop:**
```bash
# Allow Spark Application UI from EC2 instance
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION

# Allow Spark Application UI from laptop
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION
```

**View all security group rules we just created:**
```bash
aws ec2 describe-security-groups \
  --group-ids $SPARK_SG_ID \
  --region $AWS_REGION \
  --query 'SecurityGroups[0].IpPermissions' \
  --output table
```

You should see all the rules we just added: SSH (22), Spark Web UI (8080-8081), Application UI (4040), and internal cluster communication.

---

## Part 3: Create SSH Key Pair

### Step 3.1: Create Key Pair

```bash
aws ec2 create-key-pair \
  --key-name spark-cluster-key \
  --query 'KeyMaterial' \
  --output text \
  --region $AWS_REGION > spark-cluster-key.pem

echo "Key pair created and saved to spark-cluster-key.pem"
```

**Verify the key pair was created:**
```bash
aws ec2 describe-key-pairs \
  --key-names spark-cluster-key \
  --region $AWS_REGION \
  --output table
```

**Expected output:**
```
----------------------------------------------------------------------------------------------------------------------------------------------------------------
|                                                                       DescribeKeyPairs                                                                       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
||                                                                          KeyPairs                                                                          ||
|+----------------------------------+---------------------------------------------------------------+--------------------+------------------------+-----------+|
||            CreateTime            |                        KeyFingerprint                         |      KeyName       |       KeyPairId        |  KeyType  ||
|+----------------------------------+---------------------------------------------------------------+--------------------+------------------------+-----------+|
||  2025-10-04T13:16:27.878000+00:00|  xx:xx:xx:xx:...                                              |  spark-cluster-key |  key-xxxxxxxxx         |  rsa      ||
|+----------------------------------+---------------------------------------------------------------+--------------------+------------------------+-----------+|
```

You should see `spark-cluster-key` in the KeyName column - this is the key we just created!

### Step 3.2: Set Correct Permissions

```bash
chmod 400 spark-cluster-key.pem
ls -l spark-cluster-key.pem
```

The permissions should show `-r--------` (read-only for owner).

---

## Part 4: Launch EC2 Instances

### Step 4.1: Get Latest Ubuntu 24.04 (Noble) AMI ID

```bash
export AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text \
  --region $AWS_REGION)

echo "Using AMI: $AMI_ID"
```

This will get the latest Ubuntu 24.04 (Noble) AMI with GP3 storage.

### Step 4.2: Launch Master Node

```bash
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --key-name spark-cluster-key \
  --security-group-ids $SPARK_SG_ID \
  --iam-instance-profile Name=LabInstanceProfile \
  --count 1 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spark-master},{Key=Role,Value=master}]' \
  --region $AWS_REGION

echo "Master node launched!"
```

**Important parameters:**
- `--iam-instance-profile Name=LabInstanceProfile` - **CRITICAL**: Attaches the IAM role that grants this instance permissions to access AWS services like S3. Without this, Spark won't be able to read data from S3 buckets.
- `--block-device-mappings` - Sets the root volume to 100GB (default is only 8GB)

**Expected output:** You'll see a large JSON output. Key things to look for:
- `"InstanceId": "i-xxxxxxxxx"` - Your instance ID
- `"InstanceType": "t3.large"` - Instance type
- `"KeyName": "spark-cluster-key"` - SSH key
- `"State": { "Name": "pending" }` - Instance is starting
- `"Tags": [{"Key": "Name", "Value": "spark-master"}]` - Our tag
- `"SecurityGroups": [{"GroupName": "spark-cluster-sg"}]` - Our security group

The instance will start in "pending" state and transition to "running" shortly.

**Check the status of the master node:**
```bash
watch -n 2 'aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-master" \
  --query "Reservations[*].Instances[*].[Tags[?Key==\`Name\`].Value | [0], InstanceId, State.Name]" \
  --output table \
  --region $AWS_REGION'
```

This will refresh every 2 seconds. Watch for the State to change from "pending" to "running". **Press Ctrl+C to exit** once you see the instance is in "running" state.

### Step 4.3: Launch Worker Nodes

Launch 2 worker nodes (note the `--count 2` parameter):
```bash
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --key-name spark-cluster-key \
  --security-group-ids $SPARK_SG_ID \
  --iam-instance-profile Name=LabInstanceProfile \
  --count 2 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spark-worker},{Key=Role,Value=worker}]' \
  --region $AWS_REGION

echo "Worker nodes launched!"
```

**Important parameters:**
- `--count 2` - Launches 2 identical worker instances
- `--iam-instance-profile Name=LabInstanceProfile` - **CRITICAL**: Attaches the IAM role that grants these instances permissions to access AWS services like S3. Without this, Spark won't be able to read data from S3 buckets.
- `--block-device-mappings` - Sets the root volume to 100GB on each worker

**Monitor all 3 nodes until they're running:**
```bash
watch -n 2 'aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-master,spark-worker" \
  --query "Reservations[*].Instances[*].[Tags[?Key==\`Name\`].Value | [0], InstanceId, State.Name]" \
  --output table \
  --region $AWS_REGION'
```

This will refresh every 2 seconds. Watch for all 3 instances (1 spark-master and 2 spark-worker) to change from "pending" to "running". **Press Ctrl+C to exit** once all instances show "running" state.

### Step 4.4: Wait for Instances to be Running

```bash
aws ec2 wait instance-running \
  --filters "Name=tag:Name,Values=spark-master,spark-worker" \
  --region $AWS_REGION

echo "All instances are now running!"
```

### Step 4.5: Get Instance Information

Now we need to get both public and private IP addresses for our instances:
- **Public IP**: Used to SSH into instances from your local machine
- **Private IP**: Used for internal cluster communication between master and workers

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-master,spark-worker" \
  --query 'Reservations[*].Instances[*].[Tags[?Key==`Name`].Value | [0], InstanceId, PublicIpAddress, PrivateIpAddress]' \
  --output table \
  --region $AWS_REGION
```

**Save the Master IP addresses:**
```bash
export MASTER_PUBLIC_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-master" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export MASTER_PRIVATE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-master" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

echo "Master Public IP: $MASTER_PUBLIC_IP"
echo "Master Private IP: $MASTER_PRIVATE_IP"
```

**Get Worker IPs and save them automatically:**

Since we launched 2 workers with `--count 2`, they're both in the same Reservation. We need to access them as `Instances[0]` and `Instances[1]`:

```bash
export WORKER1_PUBLIC_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-worker" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER1_PRIVATE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-worker" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER2_PUBLIC_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-worker" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[1].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER2_PRIVATE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-worker" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[1].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)
```

**Verify all IPs are set:**
```bash
echo "Master: $MASTER_PUBLIC_IP (public) / $MASTER_PRIVATE_IP (private)"
echo "Worker 1: $WORKER1_PUBLIC_IP (public) / $WORKER1_PRIVATE_IP (private)"
echo "Worker 2: $WORKER2_PUBLIC_IP (public) / $WORKER2_PRIVATE_IP (private)"
```

**Expected output:**
```
Master: 44.204.89.174 (public) / 172.31.93.156 (private)
Worker 1: 54.86.212.214 (public) / 172.31.89.92 (private)
Worker 2: 54.161.66.220 (public) / 172.31.87.206 (private)
```

Make sure all 6 IP addresses are displayed correctly (your IPs will be different) before proceeding!

### Step 4.6: Save IP Addresses to File

Create a file with all the IP addresses so we can easily use them on the cluster nodes:

```bash
cat > cluster-ips.txt <<EOF
MASTER_PUBLIC_IP=$MASTER_PUBLIC_IP
MASTER_PRIVATE_IP=$MASTER_PRIVATE_IP
WORKER1_PUBLIC_IP=$WORKER1_PUBLIC_IP
WORKER1_PRIVATE_IP=$WORKER1_PRIVATE_IP
WORKER2_PUBLIC_IP=$WORKER2_PUBLIC_IP
WORKER2_PRIVATE_IP=$WORKER2_PRIVATE_IP
EOF
```

**Verify the file:**
```bash
cat cluster-ips.txt
```

You should see all 6 IP addresses listed.

### Step 4.7: Save Cluster Configuration (IDs and IPs)

Save all important resource IDs and IPs to a file for later use (cleanup, management, etc.):

```bash
cat > cluster-config.txt <<EOF
# Cluster Configuration - Created $(date)
AWS_REGION=$AWS_REGION
SPARK_SG_ID=$SPARK_SG_ID

# Instance IDs
MASTER_INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-master" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region $AWS_REGION)

WORKER1_INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-worker" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region $AWS_REGION)

WORKER2_INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=spark-worker" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[1].InstanceId' \
  --output text \
  --region $AWS_REGION)

# IP Addresses
MASTER_PUBLIC_IP=$MASTER_PUBLIC_IP
MASTER_PRIVATE_IP=$MASTER_PRIVATE_IP
WORKER1_PUBLIC_IP=$WORKER1_PUBLIC_IP
WORKER1_PRIVATE_IP=$WORKER1_PRIVATE_IP
WORKER2_PUBLIC_IP=$WORKER2_PUBLIC_IP
WORKER2_PRIVATE_IP=$WORKER2_PRIVATE_IP
EOF

# Export instance IDs for current session
export MASTER_INSTANCE_ID=$(grep MASTER_INSTANCE_ID cluster-config.txt | cut -d= -f2)
export WORKER1_INSTANCE_ID=$(grep WORKER1_INSTANCE_ID cluster-config.txt | cut -d= -f2)
export WORKER2_INSTANCE_ID=$(grep WORKER2_INSTANCE_ID cluster-config.txt | cut -d= -f2)
```

**Verify the configuration file:**
```bash
cat cluster-config.txt
```

You should see all resource IDs and IP addresses. **Keep this file safe** - you'll need it to manage and clean up your cluster later.

---

## Part 5: Copy Lab Files and IP Addresses to All Nodes

Before we configure the nodes, let's copy the necessary lab files and IP addresses to all three nodes.

### Step 5.1: Copy Files to Master Node

First, create the spark-cluster directory on the master node, then copy the files:

```bash
ssh -i spark-cluster-key.pem ubuntu@$MASTER_PUBLIC_IP "mkdir -p ~/spark-cluster"
scp -i spark-cluster-key.pem cluster-files/* ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/
scp -i spark-cluster-key.pem cluster-ips.txt ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/
```

**Note:** The first time you connect to each node, you'll be asked:
```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
Type `yes` and press Enter.

**Verify the files were copied:**
```bash
ssh -i spark-cluster-key.pem ubuntu@$MASTER_PUBLIC_IP "ls -la ~/spark-cluster/ && cat ~/spark-cluster/cluster-ips.txt"
```

You should see the lab files and IP addresses listed.

### Step 5.2: Copy Files to Worker 1

```bash
ssh -i spark-cluster-key.pem ubuntu@$WORKER1_PUBLIC_IP "mkdir -p ~/spark-cluster"
scp -i spark-cluster-key.pem cluster-files/* ubuntu@$WORKER1_PUBLIC_IP:~/spark-cluster/
scp -i spark-cluster-key.pem cluster-ips.txt ubuntu@$WORKER1_PUBLIC_IP:~/spark-cluster/
```

**Verify the files were copied:**
```bash
ssh -i spark-cluster-key.pem ubuntu@$WORKER1_PUBLIC_IP "ls -la ~/spark-cluster/ && cat ~/spark-cluster/cluster-ips.txt"
```

### Step 5.3: Copy Files to Worker 2

```bash
ssh -i spark-cluster-key.pem ubuntu@$WORKER2_PUBLIC_IP "mkdir -p ~/spark-cluster"
scp -i spark-cluster-key.pem cluster-files/* ubuntu@$WORKER2_PUBLIC_IP:~/spark-cluster/
scp -i spark-cluster-key.pem cluster-ips.txt ubuntu@$WORKER2_PUBLIC_IP:~/spark-cluster/
```

**Verify the files were copied:**
```bash
ssh -i spark-cluster-key.pem ubuntu@$WORKER2_PUBLIC_IP "ls -la ~/spark-cluster/ && cat ~/spark-cluster/cluster-ips.txt"
```

All three nodes now have the lab files and IP addresses ready!

---

## Part 6: Setup Master Node

### Step 6.1: Connect to Master Node

```bash
ssh -i spark-cluster-key.pem ubuntu@$MASTER_PUBLIC_IP
```

**Note:** The first time you connect, you'll see a prompt asking if you want to continue connecting:
```
The authenticity of host 'X.X.X.X (X.X.X.X)' can't be established.
...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
Type `yes` and press Enter to continue.

### Step 6.2: Update System and Install Java 17

Once connected to the master node, run:
```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk
java -version
```

You should see output like: `openjdk version "17.x.x"`

### Step 6.3: Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add uv to PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

Verify uv installation:
```bash
uv --version
```

### Step 6.4: Install Python Dependencies with uv

```bash
cd ~/spark-cluster
uv sync
```

This will create a virtual environment and install all dependencies from pyproject.toml including:
- PySpark 4.x
- Boto3
- Pandas
- Jupyter
- NLTK
- and more

### Step 6.5: Download Apache Spark

PySpark from pip doesn't include the cluster management scripts. Download the full Spark distribution:

```bash
cd ~
wget https://downloads.apache.org/spark/spark-4.0.1/spark-4.0.1-bin-hadoop3.tgz
tar -xzf spark-4.0.1-bin-hadoop3.tgz
mv spark-4.0.1-bin-hadoop3 spark
rm spark-4.0.1-bin-hadoop3.tgz  # Clean up to save disk space
```

### Step 6.6: Configure Environment Variables and Custom Prompt

Set up the required environment variables and customize the prompt to show "MASTER":

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
echo 'export SPARK_HOME=$HOME/spark' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python' >> ~/.bashrc

# Customize prompt to show MASTER
echo 'export PS1="[MASTER] \u@\h:\w\$ "' >> ~/.bashrc

source ~/.bashrc
```

**Verify the variables are set:**
```bash
echo "JAVA_HOME: $JAVA_HOME"
echo "SPARK_HOME: $SPARK_HOME"
ls $SPARK_HOME/sbin/start-master.sh
```

You should see the paths printed and the start-master.sh script should exist.

### Step 6.7: Verify Installation

```bash
cd ~/spark-cluster
uv run python spark_installation_test.py
```

You should see all tests pass with green checkmarks.

### Step 6.8: Load IP Addresses from File

Load the IP addresses from the cluster-ips.txt file we copied earlier:

```bash
source ~/spark-cluster/cluster-ips.txt
```

Verify all IPs are loaded:
```bash
echo "Master: $MASTER_PRIVATE_IP"
echo "Worker 1: $WORKER1_PRIVATE_IP"
echo "Worker 2: $WORKER2_PRIVATE_IP"
```

You should see the private IP addresses for all three nodes.

### Step 6.9: Create Spark Configuration Files

Create the Spark configuration files:

```bash
cd $SPARK_HOME/conf
```

### Step 6.10: Create spark-env.sh

Create the Spark environment configuration file:

```bash
cat > $SPARK_HOME/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_MASTER_HOST=$MASTER_PRIVATE_IP
export SPARK_MASTER_PORT=7077
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x $SPARK_HOME/conf/spark-env.sh
```

**Verify the spark-env.sh file:**
```bash
cat $SPARK_HOME/conf/spark-env.sh
```

### Step 6.11: Create workers file

Load the cluster IPs and create the workers configuration file:

```bash
# Load the cluster IPs
source ~/spark-cluster/cluster-ips.txt

# Verify IPs are loaded
echo "Master: $MASTER_PRIVATE_IP"
echo "Worker 1: $WORKER1_PRIVATE_IP"
echo "Worker 2: $WORKER2_PRIVATE_IP"

# Create the workers file
cat > $SPARK_HOME/conf/workers <<EOF
$WORKER1_PRIVATE_IP
$WORKER2_PRIVATE_IP
EOF
```

**Verify the workers file:**
```bash
cat $SPARK_HOME/conf/workers
```

You should see both worker private IPs listed.

### Step 6.12: Download AWS S3 Libraries for Spark

Spark needs additional libraries to read from S3. Download the Hadoop AWS and all required AWS SDK JARs:

```bash
cd $SPARK_HOME/jars

# Download Hadoop AWS library
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar

# Download AWS SDK v1 bundle (for backward compatibility)
wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Download AWS SDK v2 bundles (required by Hadoop 3.4.1)
wget https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.29.27/bundle-2.29.27.jar
wget https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.29.27/url-connection-client-2.29.27.jar
```

**Verify all JARs were downloaded:**
```bash
ls -lh $SPARK_HOME/jars/hadoop-aws-3.4.1.jar \
       $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.262.jar \
       $SPARK_HOME/jars/bundle-2.29.27.jar \
       $SPARK_HOME/jars/url-connection-client-2.29.27.jar
```

You should see all 4 JAR files listed with their sizes.

### Step 6.13: Exit Master Node

```bash
exit
```

---

## Part 7: Setup Worker Node 1

### Step 7.1: Connect to Worker 1

```bash
ssh -i spark-cluster-key.pem ubuntu@$WORKER1_PUBLIC_IP
```

### Step 7.2: Install Java 17 and uv

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk
java -version

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv --version
```

### Step 7.3: Install Python Dependencies

```bash
cd ~/spark-cluster
uv sync
```

### Step 7.4: Download Apache Spark

```bash
cd ~
wget https://downloads.apache.org/spark/spark-4.0.1/spark-4.0.1-bin-hadoop3.tgz
tar -xzf spark-4.0.1-bin-hadoop3.tgz
mv spark-4.0.1-bin-hadoop3 spark
rm spark-4.0.1-bin-hadoop3.tgz  # Clean up to save disk space
```

### Step 7.5: Configure Environment Variables and Custom Prompt

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
echo 'export SPARK_HOME=$HOME/spark' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python' >> ~/.bashrc

# Customize prompt to show WORKER-1
echo 'export PS1="[WORKER-1] \u@\h:\w\$ "' >> ~/.bashrc

source ~/.bashrc
```

### Step 7.6: Create Spark Configuration

Create the Spark environment file:

```bash
cat > $SPARK_HOME/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x $SPARK_HOME/conf/spark-env.sh
```

### Step 7.7: Download AWS S3 Libraries for Spark

Download the same AWS libraries as on the master node:

```bash
cd $SPARK_HOME/jars

# Download Hadoop AWS library
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar

# Download AWS SDK v1 bundle (for backward compatibility)
wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Download AWS SDK v2 bundles (required by Hadoop 3.4.1)
wget https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.29.27/bundle-2.29.27.jar
wget https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.29.27/url-connection-client-2.29.27.jar
```

**Verify all JARs were downloaded:**
```bash
ls -lh $SPARK_HOME/jars/hadoop-aws-3.4.1.jar \
       $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.262.jar \
       $SPARK_HOME/jars/bundle-2.29.27.jar \
       $SPARK_HOME/jars/url-connection-client-2.29.27.jar
```

### Step 7.8: Exit Worker 1

```bash
exit
```

---

## Part 8: Setup Worker Node 2

### Step 8.1: Connect to Worker 2

```bash
ssh -i spark-cluster-key.pem ubuntu@$WORKER2_PUBLIC_IP
```

### Step 8.2: Install Java 17 and uv

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk
java -version

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv --version
```

### Step 8.3: Install Python Dependencies

```bash
cd ~/spark-cluster
uv sync
```

### Step 8.4: Download Apache Spark

```bash
cd ~
wget https://downloads.apache.org/spark/spark-4.0.1/spark-4.0.1-bin-hadoop3.tgz
tar -xzf spark-4.0.1-bin-hadoop3.tgz
mv spark-4.0.1-bin-hadoop3 spark
rm spark-4.0.1-bin-hadoop3.tgz  # Clean up to save disk space
```

### Step 8.5: Configure Environment Variables and Custom Prompt

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
echo 'export SPARK_HOME=$HOME/spark' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python' >> ~/.bashrc

# Customize prompt to show WORKER-2
echo 'export PS1="[WORKER-2] \u@\h:\w\$ "' >> ~/.bashrc

source ~/.bashrc
```

### Step 8.6: Create Spark Configuration

Create the Spark environment file:

```bash
cat > $SPARK_HOME/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x $SPARK_HOME/conf/spark-env.sh
```

### Step 8.7: Download AWS S3 Libraries for Spark

Download the same AWS libraries as on the master and worker 1:

```bash
cd $SPARK_HOME/jars

# Download Hadoop AWS library
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.4.1/hadoop-aws-3.4.1.jar

# Download AWS SDK v1 bundle (for backward compatibility)
wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Download AWS SDK v2 bundles (required by Hadoop 3.4.1)
wget https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.29.27/bundle-2.29.27.jar
wget https://repo1.maven.org/maven2/software/amazon/awssdk/url-connection-client/2.29.27/url-connection-client-2.29.27.jar
```

**Verify all JARs were downloaded:**
```bash
ls -lh $SPARK_HOME/jars/hadoop-aws-3.4.1.jar \
       $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.262.jar \
       $SPARK_HOME/jars/bundle-2.29.27.jar \
       $SPARK_HOME/jars/url-connection-client-2.29.27.jar
```

### Step 8.8: Exit Worker 2

```bash
exit
```

---

## Part 9: Setup SSH Keys for Passwordless Access

### Step 9.1: Copy Private Key to Master

From your EC2 instance (where you have the spark-cluster-key.pem file):
```bash
scp -i spark-cluster-key.pem spark-cluster-key.pem ubuntu@$MASTER_PUBLIC_IP:~/.ssh/
```

### Step 9.2: Connect to Master and Configure SSH

```bash
ssh -i spark-cluster-key.pem ubuntu@$MASTER_PUBLIC_IP
```

Set permissions on the key:
```bash
chmod 400 ~/.ssh/spark-cluster-key.pem
```

Create an SSH config file to use this key for worker connections:
```bash
cat >> ~/.ssh/config <<EOF
Host *
    IdentityFile ~/.ssh/spark-cluster-key.pem
    StrictHostKeyChecking no
EOF

chmod 600 ~/.ssh/config
```

---

## Part 10: Start Spark Cluster

### Step 10.1: Start Master Node

On the master node (should still be connected):
```bash
$SPARK_HOME/sbin/start-master.sh
```

**Expected output:**
```
starting org.apache.spark.deploy.master.Master, logging to /home/ubuntu/spark/logs/spark-ubuntu-org.apache.spark.deploy.master.Master-1-ip-172-31-86-205.out
```

Verify master is running:
```bash
jps
```

**Expected output:**
```
XXXX Master
XXXX Jps
```

**Note:** `jps` (Java Process Status) is a command that lists all running Java processes. You should see `Master` in the output, which confirms the Spark Master process is running.

### Step 10.2: Start Worker Nodes from Master

```bash
$SPARK_HOME/sbin/start-workers.sh
```

**Expected output:**
```
172.31.80.136: starting org.apache.spark.deploy.worker.Worker, logging to /home/ubuntu/spark/logs/spark-ubuntu-org.apache.spark.deploy.worker.Worker-1-ip-172-31-80-136.out
172.31.94.40: starting org.apache.spark.deploy.worker.Worker, logging to /home/ubuntu/spark/logs/spark-ubuntu-org.apache.spark.deploy.worker.Worker-1-ip-172-31-94-40.out
```

You should see both worker IPs in the output. If you see `localhost` instead, check your `$SPARK_HOME/conf/workers` file (see troubleshooting below).

### Step 10.3: Verify Cluster Status

```bash
jps
```

**Expected output:**
```
XXXX Master
XXXX Jps
```

On master, you should only see `Master`. The workers are running on the other nodes.

### Step 10.4: Access the Spark Web UI

**Access the Web UI from your laptop:**

Open a browser **on your laptop** and navigate to:
```
http://[MASTER_PUBLIC_IP]:8080
```

Replace `[MASTER_PUBLIC_IP]` with your actual master public IP (use the value from Step 4.5).

**IMPORTANT SECURITY NOTE:**

This Web UI is accessible **only from your IP address** that you configured in Step 2.2. We restricted access using the `/32` CIDR notation, which means only your specific IP can access the UI, not the entire internet.

**If you can't access the Web UI:**

1. **Check your current laptop IP:** Visit https://ipchicken.com/ from your laptop
2. **Verify it matches:** Make sure it's the same IP you used in Step 2.2 (`$MY_IP`)
3. **If your laptop IP changed:** You'll need to add a new security group rule for the new IP:

```bash
# From your EC2 instance where you ran the setup
export NEW_IP=[your new laptop IP from ipchicken.com]

# Add rules for the new IP
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${NEW_IP}/32 \
  --region $AWS_REGION

aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${NEW_IP}/32 \
  --region $AWS_REGION

aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${NEW_IP}/32 \
  --region $AWS_REGION
```

**What you should see in the Web UI:**

You should see the Spark Master Web UI showing:
- URL: `spark://[MASTER_PRIVATE_IP]:7077`
- Status: ALIVE
- Workers: 2 (both should be in ALIVE state)
- Cores: Total available cores from both workers
- Memory: Total available memory from both workers

If you don't see 2 workers, see the Troubleshooting section below.

---

## Part 11: Test the Cluster with Lab Code

### Step 11.1: Test Spark Installation

Still on master node:
```bash
cd ~/spark-cluster
uv run python spark_installation_test.py
```

All tests should pass.

### Step 11.2: Run Problem 1 (Example Solution)

```bash
uv run python nyc_tlc_problem1.py 2>&1 | tee problem1.txt
```

This will:
- Download NYC taxi data from S3
- Process it using Spark
- Generate `daily_averages.csv`

### Step 11.3: Run Problem 1 on the Cluster (Distributed Mode)

The previous step ran in local mode (all processing on one node). To run in **cluster mode** with distributed processing across all worker nodes, we'll use a special cluster version that reads data directly from S3.

**Why read from S3?** In cluster mode, tasks are distributed across multiple nodes. If data files are stored locally on only one node, worker nodes can't access them. Reading directly from S3 allows all nodes to access the same data.

First, copy the cluster version script and cluster-ips.txt from your EC2 instance:

```bash
# From your EC2 instance (where you have cluster-files/)
source cluster-config.txt
scp -i spark-cluster-key.pem cluster-files/nyc_tlc_problem1_cluster.py ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/
scp -i spark-cluster-key.pem cluster-ips.txt ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/
```

Then on the master node, run the cluster version:

```bash
# Load the cluster IPs
source ~/spark-cluster/cluster-ips.txt

# Verify the master IP is set
echo "Spark Master URL: spark://$MASTER_PRIVATE_IP:7077"

# IMPORTANT: If you added AWS JARs after starting the cluster, restart it first
# This ensures the new JARs are loaded by all nodes
$SPARK_HOME/sbin/stop-all.sh
$SPARK_HOME/sbin/start-all.sh

# Wait a few seconds for workers to connect
sleep 5

# Run the cluster version (reads from S3, distributes work across all nodes)
cd ~/spark-cluster
uv run python nyc_tlc_problem1_cluster.py spark://$MASTER_PRIVATE_IP:7077 2>&1 | tee problem1_cluster.txt
```

**What's different in the cluster version?**
- Uses `s3a://` paths instead of local files
- Configures S3A filesystem with IAM instance profile credentials
- Connects to the Spark master to distribute work across workers
- Data is read directly from S3 by all worker nodes

### Step 11.4: Monitor Jobs

While jobs are running, you can monitor them in the Spark Web UI:
- Master UI: `http://[MASTER_PUBLIC_IP]:8080`
- Application UI: `http://[MASTER_PUBLIC_IP]:4040`

---

## Part 12: Cluster Management

### Stop the Cluster

On master node:
```bash
$SPARK_HOME/sbin/stop-workers.sh
$SPARK_HOME/sbin/stop-master.sh
```

### Start the Cluster Again

```bash
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-workers.sh
```

### Check Cluster Logs

Master logs:
```bash
tail -f $SPARK_HOME/logs/spark-*-master-*.out
```

Worker logs (on worker nodes):
```bash
tail -f $SPARK_HOME/logs/spark-*-worker-*.out
```

---

## Part 13: Cleanup (When Done)

### Load Cluster Configuration

First, load your saved cluster configuration:

```bash
# Load all the resource IDs and settings
source cluster-config.txt

# Verify variables are set
echo "Region: $AWS_REGION"
echo "Security Group: $SPARK_SG_ID"
echo "Master Instance: $MASTER_INSTANCE_ID"
echo "Worker 1 Instance: $WORKER1_INSTANCE_ID"
echo "Worker 2 Instance: $WORKER2_INSTANCE_ID"
```

If you don't have the `cluster-config.txt` file, you can still find resources by tags (see alternative method below).

### Terminate EC2 Instances

**Option 1: Using saved configuration (recommended)**

Check which instances will be terminated:
```bash
aws ec2 describe-instances \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID \
  --query 'Reservations[*].Instances[*].[InstanceId,Tags[?Key==`Name`].Value|[0],State.Name]' \
  --output table \
  --region $AWS_REGION
```

Terminate them:
```bash
aws ec2 terminate-instances \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID \
  --region $AWS_REGION
```

### Delete Security Group

Wait for instances to fully terminate (takes a few minutes), then:
```bash
aws ec2 delete-security-group \
  --group-id $SPARK_SG_ID \
  --region $AWS_REGION
```

### Delete Key Pair

```bash
aws ec2 delete-key-pair \
  --key-name spark-cluster-key \
  --region $AWS_REGION

rm spark-cluster-key.pem
```

---

## Troubleshooting

### Workers Not Connecting

1. Check security group rules allow all traffic within the group
2. Verify private IPs in workers file on master: `cat $SPARK_HOME/conf/workers`
3. Check worker logs: `tail -f $SPARK_HOME/logs/spark-*-worker-*.out`
4. Verify SSH access from master to workers: `ssh ubuntu@[WORKER_PRIVATE_IP]`

### Cannot Access Web UI

1. Verify security group allows port 8080 from your IP
2. Check your current IP: `curl https://checkip.amazonaws.com`
3. Update security group if IP changed

### SSH Connection Issues

1. Verify key permissions: `chmod 400 spark-cluster-key.pem`
2. Check instance is running: `aws ec2 describe-instances --instance-ids [ID]`
3. Verify security group allows SSH from your IP

### Java Version Issues

If you see Java version errors:
```bash
java -version  # Should show version 17.x.x
echo $JAVA_HOME  # Should point to Java 17
```

If not set correctly:
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
```

### Python/PySpark Issues

Verify Python environment:
```bash
cd ~/spark-cluster
uv run python --version  # Should be 3.10+
uv run python -c "import pyspark; print(pyspark.__version__)"  # Should be 4.x
```

### Out of Memory Errors

If you get OOM errors when running lab scripts:
1. Reduce the number of months of data processed
2. Increase executor memory: `--executor-memory 6g`
3. Reduce executor cores: `--executor-cores 1`

---

## Additional Resources

- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
- Spark Cluster Mode Overview: https://spark.apache.org/docs/latest/cluster-overview.html
- AWS EC2 User Guide: https://docs.aws.amazon.com/ec2/
- Lab Repository: https://github.com/dsan-gu/lab-spark-on-ec2

---

## Cost Estimate

**t3.large pricing (us-east-1):**
- On-Demand: ~$0.0832 per hour per instance
- 3 instances: ~$0.25 per hour
- Daily cost: ~$6.00

**Remember to terminate instances when not in use to avoid unnecessary charges!**

---

## Summary

You now have a 3-node Spark cluster running on EC2 with:
- Java 17 for PySpark 4.x compatibility
- Python 3.10+ with uv package management
- All dependencies from lab-spark-on-ec2 installed
- Cluster configured and ready to run lab assignments
- Ability to run both local and distributed Spark jobs

To run lab code on the cluster, either:
1. Modify scripts to include `.master("spark://[MASTER_PRIVATE_IP]:7077")` in SparkSession creation
2. Use `spark-submit` with `--master` flag

Monitor your jobs via the web UIs at ports 8080 (cluster) and 4040 (application).
