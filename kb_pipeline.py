"""
kb_pipeline.py

Description:
    This script serves as the main orchestration module for a knowledge base processing pipeline. It is designed to
    automate and streamline several critical operations including file fetching, data cleaning, file storage,
    data chunking, data embedding, and saving the processed data into a vector index for efficient querying
    and retrieval.

Author: [Your Name]
Date: [Date of Creation]
Version: [Version of the Script]
"""
from services.data.oper import read_csv_to_dataframe, read_table_to_dataframe
from services.data.store import dataframe_to_sqlite

from services.embeddings.embed import generate_embedding

import os
import pandas as pd

directory = "data"

if __name__ == '__main__':

    file_paths = []
    file_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]

            file_paths.append(file_path)
            file_names.append(file_name)

    for file_path, file_name in zip(file_paths, file_names):
        df = read_csv_to_dataframe(file_path)
        dataframe_to_sqlite(file_name, df)

    table_names = file_names

    entities_df = pd.DataFrame({
        'id': [None],
        'entity_type': [""],
        'entity_name': [""],
        'entity_description': [""]
    })
    entities_df = entities_df.iloc[0:0]
    for table_name in table_names:
        df = read_table_to_dataframe(table_name)
        column_names = []
        for column_name, column_data in df.iteritems():
            column_names.append(column_name)
            if entities_df[column_name].dtype == 'object' or pd.api.types.is_string_dtype(
                    entities_df[column_name]):
                for item in column_data:
                    entities_df = entities_df.append({
                        "id": None,
                        "entity_type": "sqlite field",
                        "entity_name": item,
                        "entity_description": f"""is a field in "{column_name}" column in {table_name} table."""
                    })

            entities_df = entities_df.append({
                "id": None,
                "entity_type": "sqlite column",
                "entity_name": column_name,
                "entity_description": f"""is a column in "{table_name}" table and contains {entities_df["column_name"].dtype} data."""
            })

        entities_df = entities_df.append({
            "id": None,
            "entity_type": "sqlite table",
            "entity_name": table_name,
            "entity_description": f"""is a table name. It contains the following columns: {column_names}"""
        })
    # process/clean
    # data_processed = clean_data(data)

    # save processed data to disk

    # chunk and save

    # for each text chunk it and return chunks

    # save chunks to disk

    # for each chunk run
    print(generate_embedding("sample text"))

    # write embeddings to vector store

    # in chatizard.py interface, perform vector searches

    print("hello")
