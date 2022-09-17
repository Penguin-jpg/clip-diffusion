import firebase_admin
from firebase_admin import storage


def get_credential(key_path):
    """回傳firebase credentials"""
    return firebase_admin.credentials.Certificate(key_path)


def initialize_firebase_app(credential, options={}):
    """初始化firebase app"""
    firebase_admin.initialize_app(credential, options=options)


def create_storage_bucket():
    """建立storage bucket"""
    return storage.bucket()


def upload_file_to_storage(bucket, file_path):
    """上傳檔案至firebase storage，並回傳url"""
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    blob.make_public()
    return blob.public_url


def delete_file_from_storage(bucket, file_path):
    """刪除firebase storage的檔案"""
    blob = bucket.blob(file_path)
    blob.delete()
