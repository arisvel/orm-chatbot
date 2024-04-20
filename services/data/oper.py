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


def fetch_entity_by_id(database_path, entity_id):
    """
    Fetch a single entity by ID from the entities table in the entities.db SQLite database.

    Args:
    database_path (str): The path to the SQLite database file.
    entity_id (int): The ID of the entity to retrieve.

    Returns:
    tuple: The row corresponding to the entity_id or None if no such entity exists.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        # Prepare the SELECT query
        query = "SELECT * FROM entities WHERE id = ?"
        # Execute the query with the provided entity_id
        cursor.execute(query, (entity_id,))
        # Fetch the result
        result = cursor.fetchone()
        return result
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()


def summarize_schema(database_path):
    """
    Summarizes the schema of an SQLite database by listing tables and their columns with types.

    Args:
    database_path (str): The path to the SQLite database file.

    Returns:
    str: A string summarizing the schema of the database.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    # Initialize an empty list to store schema descriptions for each table
    schema_descriptions = []

    # Iterate over each table and get column details
    for table_name, in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        # Format the column details into a string
        column_descriptions = ", ".join(f"{col[1]}:{col[2]}" for col in columns)
        table_description = f"{table_name} has columns ({column_descriptions})"
        schema_descriptions.append(table_description)

    # Close the cursor and connection
    cursor.close()
    conn.close()

    # Join all table descriptions into a single string
    schema_summary = "\n".join(schema_descriptions)
    return schema_summary


def execute_sqlite_query(db_path, query):
    """
    Executes an SQLite query on the specified database and returns the results.

    Args:
    db_path (str): The path to the SQLite database file.
    query (str): The SQL query to execute.

    Returns:
    list: A list of tuples representing the rows returned by the query.

    Raises:
    Exception: If an error occurs during database connection or query execution.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Create a cursor object using the cursor() method
        cursor = conn.cursor()

        # Executing the query
        cursor.execute(query)

        # Fetch all rows from the last executed statement using fetchall()
        results = cursor.fetchall()

        # Committing the changes (important if the query modifies the database)
        conn.commit()

        # Closing the connection
        conn.close()

        return results

    except sqlite3.Error as e:
        # Handle the SQLite error
        raise Exception(f"An error occurred while executing the SQL query: {e}")

    finally:
        # Ensure the connection is closed even if an error occurs
        if conn:
            conn.close()
