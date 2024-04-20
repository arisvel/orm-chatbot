import sqlite3
from sqlite3 import Error


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
