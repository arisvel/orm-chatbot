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
import sqlite3

from services.data.oper import read_csv_to_dataframe, read_table_to_dataframe, fetch_entity_by_id, summarize_schema
from services.data.store import dataframe_to_sqlite, IndexingService

from services.embeddings.embed import generate_embedding

import os
import pandas as pd
import numpy as np

directory = "data"


def parse_csv_and_save_to_db():
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

    return file_names


def create_knowledge_base(table_names):
    entities_df = pd.DataFrame({
        'id': [None],
        'entity_type': [""],
        'entity_name': [""],
        'entity_description': [""]
    })
    entities_df = entities_df.iloc[0:0]
    i = 1
    for table_name in table_names:
        df = read_table_to_dataframe(table_name)
        column_names = []
        for column_name, column_data in df.items():
            column_names.append(column_name)
            if df[column_name].dtype == 'object' or pd.api.types.is_string_dtype(
                    df[column_name]):
                for item in column_data:
                    entities_df = entities_df._append({
                        "id": i,
                        "entity_type": "sqlite field",
                        "entity_name": item,
                        "entity_description": f"""is a field in "{column_name}" column in {table_name} table."""
                    }, ignore_index=True)
                    i += 1

            entities_df = entities_df._append({
                "id": i,
                "entity_type": "sqlite column",
                "entity_name": column_name,
                "entity_description": f"""is a column in "{table_name}" table and contains {df[column_name].dtype} data."""
            }, ignore_index=True)
            i += 1

        entities_df = entities_df._append({
            "id": i,
            "entity_type": "sqlite table",
            "entity_name": table_name,
            "entity_description": f"""is a table name. It contains the following columns: {column_names}"""
        }, ignore_index=True)
        i += 1

    dataframe_to_sqlite("Entities", entities_df, "entities.db")


def generate_embeddings(database_path, index_path):
    """
    Generate embeddings for entities in the entities.db SQLite database and save to disk.
    """
    # Connect to the database
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        # Fetch the id, entity_type, entity_name, and entity_description from the relevant table
        cursor.execute("SELECT id, entity_type, entity_name, entity_description FROM entities")
        rows = cursor.fetchall()

    # Generate embeddings
    entity_ids = []
    embeddings = []
    count = 1
    for entity_id, entity_type, entity_name, entity_description in rows:
        # Concatenate the entity_type, entity_name, and entity_description
        full_text = f"{entity_type} {entity_name} {entity_description}"
        print(f"Generating embedding for entity {entity_id} ({count}/{len(rows)}) with full text: {full_text}")
        if full_text.strip():  # Ensure there is text to process
            embedding = generate_embedding(full_text)  # Assuming generate_embedding is defined elsewhere
            embeddings.append(embedding)
            entity_ids.append(entity_id)

        count += 1

    # Initialize the indexing service with the correct dimension for the embeddings
    service = IndexingService(max_elements=1000)
    service.add_items(np.array(embeddings), entity_ids)

    # Save the index to disk
    service.save_index(index_path)


if __name__ == '__main__':
    # table_names = parse_csv_and_save_to_db()

    # create_knowledge_base(table_names)

    # generate_embeddings('entities.db', 'vector_index.bin')

    user_prompt = "Are there any understocked items, without expected orders?"

    service = IndexingService()

    service.load_index("vector_index.bin")

    prompt_embedding = generate_embedding(user_prompt)
    prompt_for_sql_query_request = (f"User has asked the following: {user_prompt}, and we have the following database "
                                    f"schema:\n")
    prompt_for_sql_query_request += summarize_schema("data_store")
    prompt_for_sql_query_request += ("\nAlso we have fetched the following information that may or may not be relevant "
                                     "to the user's question:\n")

    labels, distances = service.query(np.array([prompt_embedding]), k=10)
    print("Query results:", labels, distances)

    for i in range(len(labels[0])):
        label = labels[0][i]
        info = fetch_entity_by_id("entities.db", int(label))
        prompt_for_sql_query_request += (str(info) + "\n")

    prompt_for_sql_query_request += ("Your task is to utilize all the above information that have been given to you, "
                                     "to construct a SQLite query that fetches from the database the answer that the "
                                     "user requests. Give ONLY the SQL query and nothing else.")

    print(prompt_for_sql_query_request)
