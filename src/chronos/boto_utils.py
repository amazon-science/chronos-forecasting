# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import logging
import os
import re
from pathlib import Path

import boto3
import requests
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

MODEL_FILENAMES = ["config.json", "model.safetensors", "LICENSE.txt"]
ALWAYS_DOWNLOAD = {"config.json"}
CLOUDFRONT_MAPPING = {"s3://autogluon/chronos-2": "https://d7057vjasule5.cloudfront.net"}
CHUNK_SIZE = 1024 * 1024  # 1MB


def download_model_files_from_cloudfront(
    cloudfront_url: str,
    bucket: str,
    prefix: str,
    local_path: Path,
    force_download: bool = False,
) -> None:
    for filename in MODEL_FILENAMES:
        key = f"{prefix}/{filename}"
        dest = local_path / bucket / key
        url = f"{cloudfront_url}/{filename}"

        if key not in ALWAYS_DOWNLOAD and not force_download and dest.exists():
            logger.info(f"skipping (already exists): {key}")
            continue

        if not dest.parent.exists():
            logger.info(f"creating directory: {dest.parent}")
            dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"downloading from CloudFront: {url} -> {dest}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)


def download_model_files_from_s3(
    bucket: str,
    prefix: str,
    local_path: Path,
    force_download: bool = False,
    boto3_session: boto3.Session | None = None,
) -> None:
    boto3_session = boto3_session or boto3.Session()
    s3_client = boto3_session.client("s3")

    for filename in MODEL_FILENAMES:
        key = f"{prefix}/{filename}"
        dest = local_path / bucket / key

        if key not in ALWAYS_DOWNLOAD and not force_download and dest.exists():
            logger.info(f"skipping (already exists): {key}")
            continue

        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except (ClientError, NoCredentialsError) as e:
            if isinstance(e, ClientError) and e.response["Error"]["Code"] == "404":
                logger.warning(f"object not found on S3: s3://{bucket}/{key}")
                continue
            elif isinstance(e, NoCredentialsError):
                logger.warning("credentials error, falling back to unsigned")
                # Fallback to UNSIGNED for public buckets
                s3_client = boto3_session.client("s3", config=Config(signature_version=UNSIGNED))
                s3_client.head_object(Bucket=bucket, Key=key)
            else:
                raise

        if not dest.parent.exists():
            logger.info(f"creating directory: {dest.parent}")
            dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"downloading: s3://{bucket}/{key} -> {dest}")
        try:
            s3_client.download_file(bucket, key, str(dest))
        except NoCredentialsError:
            # Fallback to UNSIGNED for public buckets
            s3_client = boto3_session.client("s3", config=Config(signature_version=UNSIGNED))
            s3_client.download_file(bucket, key, str(dest))


def cache_model_from_s3(
    s3_uri: str,
    force_download: bool = False,
    boto3_session: boto3.Session | None = None,
):
    assert re.match("^s3://([^/]+)/(.*?([^/]+)/?)$", s3_uri) is not None, f"Not a valid S3 URI: {s3_uri}"
    cache_home = Path(os.environ.get("XGD_CACHE_HOME", os.path.expanduser("~/.cache")))
    cache_dir = cache_home / "chronos-s3"
    s3_uri = s3_uri.rstrip("/")
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)

    # Check if S3 URI is in CloudFront mapping
    cloudfront_url = CLOUDFRONT_MAPPING.get(s3_uri)

    if cloudfront_url:
        try:
            download_model_files_from_cloudfront(
                cloudfront_url=cloudfront_url,
                bucket=bucket,
                prefix=prefix,
                local_path=cache_dir,
                force_download=force_download,
            )
        except Exception as e:
            logger.warning(f"CloudFront download failed, falling back to S3: {e}")
            download_model_files_from_s3(
                bucket=bucket,
                prefix=prefix,
                local_path=cache_dir,
                force_download=force_download,
                boto3_session=boto3_session,
            )
    else:
        download_model_files_from_s3(
            bucket=bucket,
            prefix=prefix,
            local_path=cache_dir,
            force_download=force_download,
            boto3_session=boto3_session,
        )

    return cache_dir / bucket / prefix
