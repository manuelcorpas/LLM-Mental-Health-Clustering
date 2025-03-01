#!/bin/bash

# Set AWS region
AWS_REGION="us-east-1"

# Define instance parameters
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID=$(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'Images[0].ImageId' --output text)
KEY_NAME="aws-gpu-key"
SECURITY_GROUP="sg-07987b4d44519095c"

# Launch EC2 GPU Spot instance
echo "Launching EC2 GPU Spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
#    --instance-market-options "MarketType=spot,SpotOptions={SpotInstanceType=one-time}" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50}}]' \
    --region $AWS_REGION \
    --query 'Instances[0].InstanceId' --output text)

echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION

# Tagging instance
aws ec2 create-tags --resources $INSTANCE_ID --tags Key=Project,Value="ML-FineTune" --region $AWS_REGION

# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --region $AWS_REGION)

echo "Instance is running at: $INSTANCE_IP"

# Copy data and scripts to AWS instance
echo "Transferring data to EC2..."
scp -i my-aws-key.pem -r ~/DATA ubuntu@$INSTANCE_IP:~/
scp -i my-aws-key.pem -r ~/PYTHON ubuntu@$INSTANCE_IP:~/

# Connect via SSH and install dependencies
echo "Setting up the environment..."
ssh -i my-aws-key.pem ubuntu@$INSTANCE_IP << 'EOF'
  sudo apt update && sudo apt upgrade -y
  sudo apt install -y nvidia-driver-535
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install transformers datasets scikit-learn pandas numpy tqdm
  echo "Setup complete!"
EOF

# Start the training script and schedule termination after completion
echo "Starting LLM fine-tuning on EC2..."
ssh -i my-aws-key.pem ubuntu@$INSTANCE_IP << 'EOF'
  CUDA_VISIBLE_DEVICES=0 python3 ~/PYTHON/00-llm_cluster_pipeline.py \
    --train_csv ~/DATA/RAECMBD_454_20241226-163036.csv \
    --desc_file ~/DATA/Code-descriptions-April-2025/icd10cm-codes-April-2025.txt \
    --mlm_epochs 2 \
    --fine_tune_epochs 2 \
    --batch_size 8 \
    --max_clusters 10

  echo "Training complete! Instance will terminate shortly."
  sudo shutdown -h +1
EOF

echo "Training started. Instance will auto-terminate upon completion. Monitor your EC2 usage closely!"
