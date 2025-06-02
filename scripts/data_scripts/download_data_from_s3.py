import os
import sys
sys.path.append(os.getcwd())
import boto3
from utils.configs import load_configs, TrainingConfigs

configs = load_configs()
training_configs: TrainingConfigs = configs['training']


s3_bucket = training_configs.s3_bucket
s3_prefix = 'data'
local_dir = "data/flattened_corpa"


def download_directory_from_s3(bucket, prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            s3_key = obj['Key']
            rel_path = os.path.relpath(s3_key, prefix)
            local_path = os.path.join(local_dir, rel_path)

            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
                s3.download_file(bucket, s3_key, local_path)


if __name__ == "__main__":
    download_directory_from_s3(s3_bucket, s3_prefix, local_dir)
