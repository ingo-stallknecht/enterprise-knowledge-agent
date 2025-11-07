# app/storage.py
import os
import pathlib

USE_AZURE = os.getenv("VARIANT", "local").lower() == "dbx"

if USE_AZURE:
    # pip install azure-storage-blob
    from azure.storage.blob import BlobServiceClient
    AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "eka-data")
    _bsc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
    _container = _bsc.get_container_client(AZURE_BLOB_CONTAINER)

def write_text_doc(rel_path: str, text: str) -> str:
    """
    Writes a Markdown/text doc to local disk (local) or Azure Blob (dbx).
    Returns a path/URI string.
    """
    if not USE_AZURE:
        fp = pathlib.Path(rel_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8")
        return str(fp)

    blob = _container.get_blob_client(rel_path)
    blob.upload_blob(text.encode("utf-8"), overwrite=True)
    return f"az://{os.getenv('AZURE_BLOB_CONTAINER','eka-data')}/{rel_path}"

def write_binary(rel_path: str, data: bytes) -> str:
    if not USE_AZURE:
        fp = pathlib.Path(rel_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(data)
        return str(fp)

    blob = _container.get_blob_client(rel_path)
    blob.upload_blob(data, overwrite=True)
    return f"az://{os.getenv('AZURE_BLOB_CONTAINER','eka-data')}/{rel_path}"
