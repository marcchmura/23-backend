import os
import boto3
from botocore.exceptions import NoCredentialsError

# === CONFIGURE THESE ===
R2_ACCESS_KEY = 'f9dc489256184a5b86a7c4552d845e4a'
R2_SECRET_KEY = '97bab7537687d7984b9c080b649356932aa55190b2adcd5a28951ea1e8c88e53'
R2_ENDPOINT = 'https://eb485412a282a7a1f97b760fa3f00107.r2.cloudflarestorage.com'  # e.g., 'https://abc123.r2.cloudflarestorage.com'
R2_BUCKET_NAME = 'images'
LOCAL_FOLDER_PATH = './downloaded_images'  # e.g., './images'

# Optional: prefix to organize files in R2 (can be empty)
R2_FOLDER_PREFIX = ''  # e.g., 'my-images/'

# === SETUP CLIENT ===
session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    endpoint_url=R2_ENDPOINT
)
print('good here')

def upload_folder_to_r2(local_folder, bucket_name, prefix=''):
    print('trigger')
    for root, _, files in os.walk(local_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")

            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f'Uploaded: {s3_key}')
            except NoCredentialsError:
                print('Credentials not available')
            except Exception as e:
                print(f'Failed to upload {local_path}: {e}')

# === RUN ===
if __name__ == '__main__':
    upload_folder_to_r2(LOCAL_FOLDER_PATH, R2_BUCKET_NAME, R2_FOLDER_PREFIX)
