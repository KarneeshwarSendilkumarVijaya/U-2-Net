import io

from google.cloud import storage
from commons.constants import PROJECT, GCS_BUCKET, TRAINING_FOLDER, MODEL_CHECKPOINT, FINAL_MODEL
import torch


def get_training_files_from_gcs(dataset, image_ext='.jpg', label_ext='.png'):
    gcs_client = storage.Client(project=PROJECT)
    bucket = gcs_client.bucket(GCS_BUCKET)

    tra_img_name_list = []
    tra_lbl_name_list = []

    prefix = f'{TRAINING_FOLDER}/{dataset}/image/'

    blobs = bucket.list_blobs(prefix=prefix)
    tra_img_name_list = [f"{blob.name}" for blob in blobs]
    tra_img_name_list = tra_img_name_list[1:]

    for img_path in tra_img_name_list:
        img_name_extension = img_path.split('/')[-1]

        img_name_extension_list = img_name_extension.split(".")
        img_name_list = img_name_extension_list[0:-1]
        img_name = img_name_list[0]
        for i in range(1, len(img_name_list)):
            img_name = img_name + "." + img_name_list[i]

        tra_lbl_name_list.append(f"{TRAINING_FOLDER}/{dataset}/label/{img_name}{label_ext}")

    gcs_client.close()

    return tra_img_name_list, tra_lbl_name_list


def save_model_to_gcs(model, model_to_save, model_to_delete):
    gcs_client = storage.Client(project=PROJECT)
    bucket = gcs_client.bucket(GCS_BUCKET)

    save_model_location = f'{MODEL_CHECKPOINT}/{model_to_save}'

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    blob = bucket.blob(save_model_location)
    blob.upload_from_string(buffer.getvalue())

    buffer.close()
    del buffer

    if model_to_delete is not None:
        delete_model_location = f'{MODEL_CHECKPOINT}/{model_to_delete}'
        blob_to_delete = bucket.blob(delete_model_location)
        blob_to_delete.delete()

    # blobs = bucket.list_blobs(prefix=f'{MODEL_CHECKPOINT}/')
    #
    #
    # file_info = [(blob.name, blob.time_created) for blob in blobs]
    # file_info.sort(key=lambda x: x[1])
    #
    # num_to_delete = len(file_info) - MODEL_CONVERGENCE_COUNT
    # for i in range(num_to_delete):
    #     blob_to_delete = bucket.blob(file_info[i][0])
    #     blob_to_delete.delete()

    gcs_client.close()
