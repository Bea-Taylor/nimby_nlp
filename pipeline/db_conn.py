import psycopg2
from psycopg2 import sql

class DBConn:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        """Initialize connection parameters"""
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            # Establish connection using psycopg2
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.conn.cursor()  # Create a cursor for executing queries
        except Exception as e:
            print(f"Error: Unable to connect to the database\n{e}")
            # If connection fails, set conn and cursor to None
            self.conn = None
            self.cursor = None
        return self.conn, self.cursor

    def close(self):
        """Close the cursor and the connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            print("Connection closed.")
        except Exception as e:
            print(f"Error closing connection: {e}")

    def execute_query(self, query, params=None):
        """Execute a single query (e.g., CREATE, INSERT, UPDATE, etc.)."""
        try:
            if self.conn is None:
                print("Error: No active connection to the database.")
                return

            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            self.conn.commit()  # Commit the transaction
            print("Query executed successfully.")
        except Exception as e:
            print(f"Error executing query: {e}")
            if self.conn:
                self.conn.rollback()  # Rollback in case of error

    def fetch_all(self, query, params=None):
        """Fetch all results from a SELECT query."""
        try:
            if self.conn is None:
                print("Error: No active connection to the database.")
                return None

            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            return self.cursor.fetchall()  # Return all results
        except Exception as e:
            print(f"Error fetching results: {e}")
            return None
