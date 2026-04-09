"""Upload the current model artifact and registry to Supabase Storage.

Run once locally before deploying to Koyeb:

    python deployment/scripts/upload_model.py

Requires SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and optionally
SUPABASE_STORAGE_BUCKET (default: "models") to be set in the environment
or .env file.

The script uploads two objects to the bucket:
  - latest.json                           (registry metadata)
  - <artifact_filename>.joblib            (serialised pipeline)

Existing objects are overwritten (upsert).
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add repo root to path so config/ and src/ are importable without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from supabase import create_client  # noqa: E402

from config.settings import settings  # noqa: E402


def _upload(bucket, key: str, data: bytes, content_type: str) -> None:
    try:
        bucket.upload(
            path=key,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        logger.info("uploaded  %s", key)
    except Exception as exc:
        logger.error("failed to upload %s: %s", key, exc)
        raise


def main() -> None:
    registry_path = settings.models_registry_path / "latest.json"
    if not registry_path.exists():
        logger.error("Registry not found at %s — run 'make train' first.", registry_path)
        sys.exit(1)

    entry = json.loads(registry_path.read_text())
    artifact_path = Path(entry["path"])
    if not artifact_path.exists():
        logger.error("Artifact not found at %s", artifact_path)
        sys.exit(1)

    client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    bucket_name = settings.supabase_storage_bucket

    # Create the bucket if it doesn't exist yet.
    try:
        client.storage.create_bucket(bucket_name, options={"public": False})
        logger.info("created bucket '%s'", bucket_name)
    except Exception as exc:
        # Supabase raises if it already exists — that's fine.
        if "already exists" in str(exc).lower() or "Duplicate" in str(exc):
            logger.info("bucket '%s' already exists", bucket_name)
        else:
            logger.error("could not create bucket '%s': %s", bucket_name, exc)
            raise

    bucket = client.storage.from_(bucket_name)

    logger.info(
        "uploading to bucket '%s' on %s",
        bucket_name,
        settings.supabase_url,
    )

    _upload(bucket, "latest.json", registry_path.read_bytes(), "application/json")
    _upload(
        bucket,
        artifact_path.name,
        artifact_path.read_bytes(),
        "application/octet-stream",
    )

    # Upload the raw training data so Streamlit Cloud can load it.
    csv_path = settings.data_raw_path
    if csv_path.exists():
        _upload(bucket, "ds_salaries.csv", csv_path.read_bytes(), "text/csv")
    else:
        logger.warning("raw data not found at %s — skipping CSV upload", csv_path)

    logger.info("done — model '%s' (trained %s) is live in Supabase Storage.",
                entry["name"], entry["timestamp"])


if __name__ == "__main__":
    main()
