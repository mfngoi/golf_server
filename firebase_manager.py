import firebase_admin
from firebase_admin import credentials, storage


class FireBaseManager:

    def __init__(self):
        if not firebase_admin._apps:
            self.bucket_link = 'golf-storage.appspot.com'
            self.cred = credentials.Certificate("firebase_private_key.json")
            firebase_admin.initialize_app(self.cred, {'storageBucket': self.bucket_link})

    def upload_file(self, destination_path, origin_path):
        # the bucket object in firebase storage
        bucket = storage.bucket()

        # create blob object to get ready to upload file to destination
        blob = bucket.blob(destination_path)

        # specify which file to upload from local computer
        blob.upload_from_filename(origin_path)

        blob.make_public()

        return blob.public_url

