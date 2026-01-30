# Spark Cluster Automation Scripts

This directory contains automation scripts to set up and tear down a complete Apache Spark cluster on AWS EC2.

## Scripts

### 1. `setup-spark-cluster.sh`

Automatically creates and configures a complete Spark cluster with:
- 1 Master node
- 3 Worker nodes
- All necessary security groups and SSH keys
- Complete software installation and configuration
- Spark cluster started and ready to use

**Usage:**
```bash
./setup-spark-cluster.sh <LAPTOP_IP>
```

**Parameters:**
- `LAPTOP_IP`: Your laptop's public IP address (get it from https://ipchicken.com/)

**Example:**
```bash
./setup-spark-cluster.sh 123.45.67.89
```

**What it does:**
1. Creates a security group with proper access rules
2. Creates an SSH key pair
3. Launches 1 master + 3 worker EC2 instances (t3.large, 100GB disk)
4. Installs all dependencies (Java, Python, Spark, uv, etc.)
5. Downloads AWS S3 libraries for Spark
6. Configures Spark on all nodes
7. Sets up passwordless SSH from master to workers
8. Starts the Spark cluster
9. Saves all configuration to `cluster-config.txt`

**Duration:** Approximately 10-15 minutes

**Output:**
- `cluster-config.txt` - All resource IDs and IP addresses
- `cluster-ips.txt` - IP addresses for the nodes
- `spark-cluster-key-TIMESTAMP.pem` - SSH private key

### 2. `cleanup-spark-cluster.sh`

Safely terminates all resources created by the setup script.

**Usage:**
```bash
./cleanup-spark-cluster.sh
```

**What it does:**
1. Loads configuration from `cluster-config.txt`
2. Displays all resources to be deleted
3. Asks for confirmation
4. Stops the Spark cluster
5. Terminates all EC2 instances
6. Deletes the security group
7. Deletes the key pair
8. Removes local configuration files

## Prerequisites

Before running the setup script, ensure you have:

1. **AWS CLI configured** with appropriate credentials
   - If running on EC2 with IAM role, this is automatic
   - Otherwise run `aws configure`

2. **Required AWS permissions** to:
   - Create/delete EC2 instances
   - Create/delete security groups
   - Create/delete key pairs

3. **LabInstanceProfile IAM role** must exist (for S3 access from cluster)

4. **`cluster-files/` directory** must exist with lab code

5. **`jq` installed** (for JSON parsing)
   ```bash
   sudo apt install -y jq
   ```

## Quick Start

1. Get your laptop's public IP from https://ipchicken.com/

2. Run the setup script:
   ```bash
   ./setup-spark-cluster.sh 123.45.67.89
   ```

3. Wait for setup to complete (~10-15 minutes)

4. Access your cluster:
   ```bash
   # SSH to master (use key from output)
   ssh -i spark-cluster-key-TIMESTAMP.pem ubuntu@MASTER_IP

   # Or open Web UI in browser
   http://MASTER_IP:8080
   ```

5. Run a test job:
   ```bash
   ssh -i spark-cluster-key-TIMESTAMP.pem ubuntu@MASTER_IP
   cd ~/spark-cluster
   source cluster-ips.txt
   uv run python nyc_tlc_problem1_cluster.py spark://$MASTER_PRIVATE_IP:7077
   ```

6. When done, clean up:
   ```bash
   ./cleanup-spark-cluster.sh
   ```

## Debugging

The setup script provides detailed logging with color-coded messages:

- **[INFO]** (Blue) - Informational messages
- **[SUCCESS]** (Green) - Successful operations
- **[WARN]** (Yellow) - Warnings
- **[ERROR]** (Red) - Errors

If the script fails:
1. Check the error message for details
2. Check `cluster-config.txt` to see what resources were created
3. You can manually clean up using AWS console or CLI
4. Or run `cleanup-spark-cluster.sh` to remove partial setup

## Features

### Automated Setup
- ✓ Security group with proper ingress rules
- ✓ SSH access from both EC2 instance and laptop
- ✓ Web UI access (ports 8080, 8081, 4040)
- ✓ Automatic instance launch with proper settings
- ✓ 100GB disk per instance (avoids disk space issues)
- ✓ IAM instance profile for S3 access
- ✓ Parallel worker node setup (faster)

### Software Installation
- ✓ Java 17
- ✓ Python 3 with uv package manager
- ✓ Apache Spark 4.0.1
- ✓ AWS S3 libraries (hadoop-aws, aws-sdk)
- ✓ All Python dependencies via uv

### Cluster Configuration
- ✓ Custom bash prompts ([MASTER], [WORKER-1], etc.)
- ✓ Spark master/worker configuration
- ✓ Workers file with all worker IPs
- ✓ Passwordless SSH from master to workers
- ✓ Cluster automatically started

### Configuration Management
- ✓ All settings saved to `cluster-config.txt`
- ✓ Easy cleanup with saved configuration
- ✓ IP addresses distributed to all nodes

## Cost Considerations

The default setup uses:
- **Instance type:** t3.large ($0.0832/hour each)
- **Number of instances:** 4 (1 master + 3 workers)
- **Storage:** 100GB GP3 per instance
- **Estimated cost:** ~$0.33/hour or ~$8/day

**Remember to clean up when done to avoid unnecessary charges!**

## Troubleshooting

### Script fails with "AMI not found"
- The Ubuntu 24.04 AMI might not be available in your region
- Edit the script to use a different AMI ID

### Script fails with "LabInstanceProfile not found"
- The IAM instance profile doesn't exist
- Create it or remove the `--iam-instance-profile` parameter from the script

### Workers don't connect to master
- Check security group rules allow traffic within the group
- SSH to master and verify workers file: `cat $SPARK_HOME/conf/workers`
- Check worker logs: `tail -50 $SPARK_HOME/logs/spark-*-worker-*.out`

### Cannot access Web UI
- Verify your laptop IP hasn't changed (check ipchicken.com)
- Check security group has your current IP allowed
- Try accessing from the EC2 instance IP instead

### Cleanup script fails
- Some resources might have dependencies
- Wait a few minutes and try again
- Manually clean up from AWS console if needed

## Manual Cleanup

If `cleanup-spark-cluster.sh` doesn't work, you can manually clean up:

```bash
# Load configuration
source cluster-config.txt

# Terminate instances
aws ec2 terminate-instances \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID $WORKER3_INSTANCE_ID \
  --region $AWS_REGION

# Wait for instances to terminate
aws ec2 wait instance-terminated \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID $WORKER3_INSTANCE_ID \
  --region $AWS_REGION

# Delete security group
aws ec2 delete-security-group --group-id $SPARK_SG_ID --region $AWS_REGION

# Delete key pair
aws ec2 delete-key-pair --key-name $KEY_NAME --region $AWS_REGION
```

## Customization

To modify the cluster setup, edit `setup-spark-cluster.sh`:

- **Change instance type:** Modify `--instance-type t3.large`
- **Change disk size:** Modify `"VolumeSize":100`
- **Change number of workers:** Modify the `--count 3` parameter
- **Change region:** Modify `AWS_REGION=us-east-1`
- **Change Spark version:** Modify the wget URL for Spark download

## See Also

- `SPARK_CLUSTER_SETUP.md` - Detailed manual setup instructions
- `cluster-files/` - Lab code and configurations
