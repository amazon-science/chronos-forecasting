# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import logging
import os
import re
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

ALLOWED_FILE_SUFFIXES = [".json", ".safetensors", ".md", ".txt"]


def download_prefix(bucket, prefix, local_path, force_download: bool = False, boto3_session=None) -> None:
    boto3_session = boto3_session or boto3.Session()
    s3_resource = boto3_session.resource("s3")
    bucket = s3_resource.Bucket(bucket)

    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue

        if not any(obj.key.endswith(suffix) for suffix in ALLOWED_FILE_SUFFIXES):
            logger.info(f"skipping (not in allowed file types): {obj.key}")
            continue

        dest = local_path / bucket.name / obj.key

        if not force_download and dest.exists():
            logger.info(f"skipping (already exists): {obj.key}")
            continue

        if not dest.parent.exists():
            logger.info(f"creating directory: {dest.parent}")
            dest.parent.mkdir(exist_ok=True, parents=True)

        logger.info(f"downloading: s3://{bucket.name}/{obj.key} -> {dest}")
        bucket.download_file(obj.key, str(dest))


def cache_model_from_s3(s3_uri: str, force_download=False):
    assert re.match("^s3://([^/]+)/(.*?([^/]+)/?)$", s3_uri) is not None, f"Not a valid S3 URI: {s3_uri}"
    cache_home = Path(os.environ.get("XGD_CACHE_HOME", os.path.expanduser("~/.cache")))
    cache_dir = cache_home / "chronos-s3"
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    download_prefix(bucket=bucket, prefix=prefix, local_path=cache_dir, force_download=force_download)
    return cache_dir / bucket / prefix
