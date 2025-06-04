import argparse
import os
import sys
import boto3
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.configs import load_configs, TrainingConfigs

configs = load_configs()
training_configs: TrainingConfigs = configs['training']

s3_bucket = training_configs.s3_bucket
s3_prefix = training_configs.s3_prefix  # Reuse the model's S3 prefix

def upload_experiments_to_s3(local_dir, bucket, prefix):
    s3 = boto3.client('s3')
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{prefix}/{relative_path}"
            print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            s3.upload_file(local_path, bucket, s3_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload specified experiment files to S3. If none specified, uploads everything.")
    parser.add_argument('--files', nargs='*', default=None, help="List of file paths relative to 'experiments/' to upload.")
    args = parser.parse_args()

    if args.files:
        s3 = boto3.client('s3')
        for rel_path in args.files:
            local_path = os.path.join("experiments", rel_path)
            s3_key = f"{s3_prefix}/{rel_path}"
            print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
            s3.upload_file(local_path, s3_bucket, s3_key)
    else:
        upload_experiments_to_s3("experiments", s3_bucket, s3_prefix)