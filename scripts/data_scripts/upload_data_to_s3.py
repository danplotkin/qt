import os
import sys
import boto3
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.configs import load_configs, TrainingConfigs

configs = load_configs()
training_configs: TrainingConfigs = configs['training']


s3_bucket = training_configs.s3_bucket
s3_prefix = 'data'


def upload_directory_to_s3(local_dir, bucket, prefix):
    s3 = boto3.client('s3')
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".pt"):
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{prefix}/{relative_path}"
                try:
                    s3.head_object(Bucket=bucket, Key=s3_key)
                    print(f"Skipping {local_path}, already exists in s3://{bucket}/{s3_key}")
                except s3.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                        s3.upload_file(local_path, bucket, s3_key)
                    else:
                        raise


if __name__ == "__main__":
    upload_directory_to_s3("data/flattened_corpa", s3_bucket, s3_prefix)
