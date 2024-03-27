# not using this currently

from google.cloud import storage, aiplatform
from commons.constants import GCS_BUCKET, PROJECT, LOCATION, DEPLOY_IMAGE

from utils.logger_details import get_logger
log = get_logger('__main__')


def download_blob(source_blob, destination_file, gcs_client=None):
    """Downloads a blob from bucket."""
    if gcs_client is None:
        gcs_client = storage.Client(project=PROJECT)
    bucket = gcs_client.bucket(GCS_BUCKET)
    blob = bucket.blob(source_blob)
    if blob.exists():
        blob.download_to_filename(destination_file, checksum=None)
    else:
        raise FileNotFoundError(f"Blob {source_blob} doesn't exist in specified bucket")
    if gcs_client is not None:
        gcs_client.close()
    log.debug(f"Downloaded storage object {source_blob} from bucket {GCS_BUCKET} to {destination_file}.")


def upload_blob(source_file_name, destination_blob_name, gcs_client=None):
    """Uploads a file to the bucket."""
    if gcs_client is None:
        gcs_client = storage.Client(project=PROJECT)
    bucket = gcs_client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    if gcs_client is not None:
        gcs_client.close()
    log.debug(f"Uploaded local file {source_file_name} to blob {destination_blob_name} in bucket {GCS_BUCKET}.")


def move_blob(source_blob_name, destination_blob_name, gcs_client=None):
    """Moves a blob from one location to another within GCS."""
    if gcs_client is None:
        gcs_client = storage.Client(project=PROJECT)

    bucket = gcs_client.bucket(GCS_BUCKET)
    blob = bucket.blob(source_blob_name)

    if not blob.exists():
        log.debug(f"Blob {source_blob_name} doesn't exist in specified bucket")
        return

    destination_blob = bucket.copy_blob(blob, bucket, new_name=destination_blob_name)

    if not destination_blob.exists():
        raise Exception(f"Failed to copy {source_blob_name} to {destination_blob_name}")

    blob.delete()

    if gcs_client is not None:
        gcs_client.close()

    log.debug(f"Moved {source_blob_name} to {destination_blob_name}.")


def upload_to_model_registry(artifact_uri: str, display_name: str):
    aiplatform.init(project=PROJECT, location=LOCATION)
    models = aiplatform.Model.list(filter=f"display_name={display_name}")

    if len(models) == 0:
        model_v = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=DEPLOY_IMAGE,
            is_default_version=True,
        )
    else:
        parent_model = models[0].resource_name
        model_v = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=DEPLOY_IMAGE,
            parent_model=parent_model,
            is_default_version=True,
        )

    model_v.wait()
    log.debug(f"model version saved in {artifact_uri} uploaded to model registry")
    log.debug(f"model version display name: {model_v.display_name}")
    log.debug(f"model version resource name: {model_v.resource_name}")
    return model_v
