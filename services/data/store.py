import sqlite3
from sqlite3 import Error
import hnswlib
import numpy as np


def dataframe_to_sqlite(table_name, dataframe, db_file="data_store"):
    """
    Creates a table in SQLite from a pandas DataFrame if it doesn't exist and inserts the data.

    Parameters:
    db_file (str): The path to the SQLite database file.
    table_name (str): The name of the table to create and insert data into.
    dataframe (pd.DataFrame): The pandas DataFrame to insert into the SQLite table.

    Returns:
    bool: True if the operation was successful, False otherwise.
    """
    # Establish a connection to the database
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite database: {db_file}")

        # Use the pandas function to_sql to insert the data
        dataframe.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Data inserted into table '{table_name}' successfully.")
        return True
    except Error as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        if conn:
            conn.close()


class IndexingService:
    def __init__(self, space='cosine', dim=768, max_elements=1000):
        self.dim = dim
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(1800)  # Set higher for more accurate but slower search

    def add_items(self, embeddings, ids):
        self.index.add_items(embeddings, ids)

    def save_index(self, path='hnsw_index.bin'):
        self.index.save_index(path)

    def load_index(self, path='hnsw_index.bin'):
        self.index.load_index(path)

    def query(self, queries, k=5):
        labels, distances = self.index.knn_query(queries, k=k)
        return labels, distances

    def get_item(self, chunk_id):
        embedding = self.index.get_items([chunk_id])
        if embedding.size > 0:
            return embedding[0]
        else:
            return None
