import pandas as pd
import numpy as np
import os
import requests

# Google Drive File ID from the shared link
GDRIVE_FILE_ID = "1U3Xy2-DtddgTFTuG4SnjBJ1k7w42R_Fj"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "raw_data.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "raw_data.parquet")

# Standard path used by the rest of the application
DATA_PATH = CSV_PATH 

FEATURES = ["pm25", "pm10", "no2", "o3", "temperature", "humidity"]
CHUNK_SIZE = 100_000


def download_from_gdrive(file_id, destination):
    """
    Downloads a file from Google Drive, handling the 'confirm' token for large files.
    """
    print(f"Downloading dataset from Google Drive (ID: {file_id})...")
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(url, params={'id': file_id}, stream=True)
    
    # Check for confirmation token in cookies (for large files)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        response = session.get(url, params={'id': file_id, 'confirm': token}, stream=True)
        
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"Download complete: {destination}")


def ensure_data_exists():
    """
    Ensures the data exists locally, downloading if necessary.
    Converts CSV to Parquet for performance if not already done.
    """
    if not os.path.exists(CSV_PATH) and not os.path.exists(PARQUET_PATH):
        download_from_gdrive(GDRIVE_FILE_ID, CSV_PATH)
    
    if os.path.exists(CSV_PATH) and not os.path.exists(PARQUET_PATH):
        print("Optimizing dataset: Converting CSV to Parquet...")
        df = pd.read_csv(CSV_PATH)
        df.to_parquet(PARQUET_PATH, index=False, engine='pyarrow')
        print(f"Optimization complete: {PARQUET_PATH}")


def load_data_chunked(path=None, usecols=None):
    """
    Loads data. If Parquet exists, uses it (much faster). Otherwise falls back to CSV.
    """
    ensure_data_exists()
    
    if os.path.exists(PARQUET_PATH):
        # Parquet loading is fast enough that we don't usually need chunking here,
        # but we respect usecols for performance.
        return pd.read_parquet(PARQUET_PATH, columns=usecols, engine='pyarrow')
    
    # Fallback to CSV chunked loading
    path = CSV_PATH
    chunks = []
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, usecols=usecols):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def load_and_prepare(path=None):
    df = load_data_chunked()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.dropna(inplace=True)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    return df


def load_station_means(path=None):
    """
    Computes station-level means. Uses Parquet if available.
    """
    ensure_data_exists()
    
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH, columns=["station_id", "zone"] + FEATURES, engine='pyarrow')
        means = df.groupby(["station_id", "zone"])[FEATURES].mean().reset_index()
        return means
        
    # Fallback to CSV chunked approach
    agg = None
    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
        grp = chunk.groupby(["station_id", "zone"])[FEATURES].agg(["sum", "count"])
        if agg is None:
            agg = grp
        else:
            agg = agg.add(grp, fill_value=0)
    
    means = pd.DataFrame(index=agg.index)
    for f in FEATURES:
        means[f] = agg[(f, "sum")] / agg[(f, "count")]
    means.reset_index(inplace=True)
    return means