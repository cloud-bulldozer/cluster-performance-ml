import os
import argparse
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError


def upload_to_s3(file_path: str, bucket: str, key: str) -> None:
    s3 = boto3.client("s3")
    try:
        print(f"Uploading '{file_path}' to 's3://{bucket}/{key}' ...")
        s3.upload_file(file_path, bucket, key)
        print("Upload completed.")
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        print(f"Error uploading to S3: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a model file to AWS S3."
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Local path to models folder (e.g. /path/to/models).",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Name of the model file to upload (e.g. RandomForest_model.joblib).",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="S3 prefix (folder) to upload to (e.g. models/ or cluster-density-v2/models/).",
    )
    parser.add_argument(
        "--bucket",
        default="kube-burner-ai-s3-bucket",
        help="S3 bucket name (default: kube-burner-ai-s3-bucket).",
    )

    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    if not os.path.isdir(models_dir):
        raise NotADirectoryError(f"Not a directory: {models_dir}")

    local_path = os.path.join(models_dir, args.model_file)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Model file not found: {local_path}")

    s3_key = os.path.join(args.prefix, args.model_file).replace("\\", "/")

    upload_to_s3(local_path, args.bucket, s3_key)


if __name__ == "__main__":
    main()