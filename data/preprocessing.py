import os
import pandas as pd
import json

class LocalDatasetLoader:
    def __init__(self, data_path="data/cnn_dailymail"):
        self.data_path = data_path

    def load_from_csv(self, filename="train.csv", limit=None):
        """Loads dataset from a local CSV file."""
        full_path = os.path.join(self.data_path, filename)
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} not found.")
            return None
        
        df = pd.read_csv(full_path)
        if limit:
            df = df.head(limit)
        return df

    def preprocess(self, df):
        """
        Simple preprocessing: cleaning whitespace, removing nulls.
        Expects 'article' and 'highlights' columns for CNN/DM.
        """
        if df is None: return None
        
        df = df.dropna(subset=['article', 'highlights'])
        df['article'] = df['article'].apply(lambda x: x.strip())
        df['highlights'] = df['highlights'].apply(lambda x: x.strip())
        
        return df

    def get_sample(self, df, index=0):
        """Fetches a single sample from the dataframe."""
        if df is None or index >= len(df):
            return None, None
        
        return df.iloc[index]['article'], df.iloc[index]['highlights']
