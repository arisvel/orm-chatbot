import pandas as pd
import sqlite3


def read_csv_to_dataframe(file_path):
    """
    Reads a CSV file from the specified file path and returns a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, sep=";")
        return df
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None


def read_table_to_dataframe(table_name, db_file="data_store"):
    """
    Reads a specified table from a SQLite database into a pandas DataFrame.

    Parameters:
    table_name (str): The name of the table to read.
    db_file (str): The path to the SQLite database file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the specified table, or None if an error occurs.
    """
    # Establish a connection to the database
    try:
        with sqlite3.connect(db_file) as conn:
            # Use pandas to read the table into a DataFrame
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            return df
    except Exception as e:
        print(f"An error occurred while reading the table: {e}")
        return None
