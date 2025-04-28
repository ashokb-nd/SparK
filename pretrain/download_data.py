
import os
import zipfile
import mlflow

FULL_DATASET = True

if not FULL_DATASET:
    zip_s3_path = "s3://netradyne-sharing/analytics/ashok/spark_data_dummy.zip"
    local_dir = 'tmp'
    print(f"Downloading data from {zip_s3_path} to {local_dir}...")
    zip_path = mlflow.artifacts.download_artifacts(zip_s3_path, dst_path=local_dir)

    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_dir)
else:
    image_s3_path = "s3://netradyne-sharing/analytics/advaith/midgard/ds/fe_large_scale_reprocessing_200k_equally_sampled_alerts_july_1/"
    temp_dir = 'tmp/spark_data_dummy'

    print(f"Downloading data from {image_s3_path} to {temp_dir}...")
    os.makedirs(temp_dir, exist_ok=True)
    mlflow.artifacts.download_artifacts(image_s3_path, dst_path=temp_dir)


# remove any subdirectories without having any files in it. as it causes error.
cmd = "rmdir tmp/spark_data_dummy/* "
os.system(cmd)

print("Data downloaded and extracted successfully.")
# extracted_dir = 'data/spark_dummy_data'

