# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import logging
import os
import re
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

MODEL_FILENAMES = ["config.json", "model.safetensors"]


def download_model_files(
    bucket: str,
    prefix: str,
    local_path: Path,
    force_download: bool = False,
    boto3_session: boto3.Session | None = None,
) -> None:
    if boto3_session is not None:
        s3_client = boto3_session.client("s3")
    else:
        boto3_session = boto3.Session()
        # We use UNSIGNED here because we are downloading from a public S3 prefix
        s3_client = boto3_session.client("s3", config=Config(signature_version=UNSIGNED))

    for filename in MODEL_FILENAMES:
        key = f"{prefix.rstrip('/')}/{filename}"
        dest = local_path / bucket / key

        if not force_download and dest.exists():
            logger.info(f"skipping (already exists): {key}")
            continue

        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"object not found on S3: s3://{bucket}/{key}")
                continue
            raise

        if not dest.parent.exists():
            logger.info(f"creating directory: {dest.parent}")
            dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"downloading: s3://{bucket}/{key} -> {dest}")
        s3_client.download_file(bucket, key, str(dest))


def cache_model_from_s3(
    s3_uri: str,
    force_download: bool = False,
    boto3_session: boto3.Session | None = None,
):
    assert re.match("^s3://([^/]+)/(.*?([^/]+)/?)$", s3_uri) is not None, f"Not a valid S3 URI: {s3_uri}"
    cache_home = Path(os.environ.get("XGD_CACHE_HOME", os.path.expanduser("~/.cache")))
    cache_dir = cache_home / "chronos-s3"
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    download_model_files(
        bucket=bucket, prefix=prefix, local_path=cache_dir, force_download=force_download, boto3_session=boto3_session
    )
    return cache_dir / bucket / prefix
