from db_conn import DBConn
import psycopg2
import pandas as pd
import time
import uuid
import base64

class CommentsSaver:
    def __init__(self, env='dev'):
        """Initialize database connection with environment-specific settings."""
        valid_envs = {'dev', 'prod'}
        if env not in valid_envs:
            raise ValueError(f"Invalid environment: {env}. Must be one of {valid_envs}.")

        # Database configuration
        self.env = env
        self.dbname = f"ai4ci-db-{self.env}" if self.env == "dev" else "ai4ci-db"
        self.dbuser = "postgres"
        self.dbpassword = "ci4ai"
        self.dbhost = "128.40.193.10"
        self.dbport = "5433"

        # Table schema
        self.table_name = "comments"
        self.columns = [
            "council", "application_id", "address",
            "stance", "date", "comment_text", "add_date"
        ]

        # Initialize database connection
        self.db_connector = DBConn(
            dbname=self.dbname,
            user=self.dbuser,
            password=self.dbpassword,
            host=self.dbhost,
            port=self.dbport
        )
        self.connection = None
        self.cursor = None

        try:
            print(f"Connecting to the {self.dbname} database...")
            self.connection, self.cursor = self.db_connector.connect()
            print(f"Successfully connected to {self.dbname}.")
        except Exception as e:
            print(f"Error connecting to {self.dbname}: {e}")
            raise ConnectionError(f"Failed to connect to {self.dbname}: {e}")

    def drop_table(self):
        """Drop the comments table."""
        drop_table_statement = f"DROP TABLE IF EXISTS {self.table_name};"
        try:
            if self.connection and self.cursor:
                self.cursor.execute(drop_table_statement)
                self.connection.commit()
                print(f"Table {self.table_name} dropped successfully.")
            else:
                print("Database connection not established.")
        except Exception as e:
            print(f"Error dropping table: {e}")

    # ' stance VARCHAR(255) CHECK (stance IN ('Supports', 'Objects', 'Neutral')),'
    def create_table(self):
        """Create the comments table if it doesn't exist."""
        create_table_statement = """
        CREATE TABLE IF NOT EXISTS comments (
            id SERIAL PRIMARY KEY,
            council VARCHAR(255) NOT NULL,
            comment_id VARCHAR(155) NOT NULL,
            application_id VARCHAR(155) NOT NULL,
            address VARCHAR(512),
            stance VARCHAR(255),
            date DATE NOT NULL,
            comment_text TEXT NOT NULL,
            add_date DATE DEFAULT CURRENT_DATE,
            CONSTRAINT unique_comment_application UNIQUE (comment_id, application_id)
        );
        """
        try:
            if self.connection and self.cursor:
                self.cursor.execute(create_table_statement)
                self.connection.commit()
                print(f"Table {self.table_name} created successfully.")
            else:
                print("Database connection not established.")
        except Exception as e:
            print(f"Error creating table: {e}")

    def insert_comment(self, council, comment_id, application_id, address, stance, date, comment_text):
        """Insert a single comment into the table."""

        temp_id = comment_id if comment_id not in [None, ""] else self.generate_comment_uuid()

        # Convert "None" (string) or empty string to Python's None (which becomes SQL NULL)
        date = None if date in [None, "", "None"] else date

        insert_statement = """
        INSERT INTO comments (council, comment_id, application_id, address, stance, date, comment_text, add_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_DATE);
        """

        retry_attempts = 3

        for attempt in range(1, retry_attempts + 1):
            try:
                if self.connection and self.cursor:
                    with self.connection:
                        with self.connection.cursor() as cursor:
                            cursor.execute(insert_statement, (council, temp_id, application_id, address, stance, date, comment_text))
                            print(f"Inserted comment on attempt {attempt}.")
                            return  # Exit if successful
            except psycopg2.Error as e:
                if e.pgcode == "23505":  # Unique constraint violation
                    print(f"Skipping duplicate comment: (comment_id={temp_id}, application_id={application_id})")
                    return  # Immediately skip further attempts
                else:
                    print(f"Error inserting comment on attempt {attempt}: {e}")
                    if attempt == retry_attempts:
                        print("Max retry attempts reached. Could not insert comment.")
                    else:
                        print("Retrying in 2 seconds...")
                        time.sleep(2)

    def insert_sample(self):
        """Insert hardcoded comments into the table."""
        sample_data = [
            ("Lambeth", "111", "21/00343/FUL", "None", "Objects", "2021-11-08",
             "This site is variously known as 75 Knollys Rd or 73-79 Knollys Rd or The Marziale. It has been subject to multiple applications since 2013..."),
            ("Southwark", "2", "22/00412/FUL", "123 High Street", "Supports", "2024-01-15",
             "A well-thought-out development that enhances the local community...")
        ]

        try:
            if self.connection and self.cursor:
                for data in sample_data:
                    # Call the insert_comment method for each sample
                    council, comment_id, application_id, address, stance, date, comment_text = data
                    self.insert_comment(council, comment_id, application_id, address, stance, date, comment_text)
                print("Sample comments inserted successfully.")
            else:
                print("Database connection not established.")
        except Exception as e:
            print(f"Error inserting sample comments: {e}")

    def read_by_council(self, council, output_format='DataFrame'):
        """Fetch and return all comments for a given council in the specified format."""
        select_statement = "SELECT * FROM comments WHERE council = %s;"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(select_statement, (council,))
                rows = cursor.fetchall()
                # columns = [desc for desc in cursor.description]  # Get column names
                columns = [desc[0] for desc in cursor.description]

                if output_format == 'DataFrame':
                    return pd.DataFrame(rows, columns=columns)  # Convert to DataFrame
                elif output_format == 'List':
                    return rows  # Return as a list of tuples
                else:
                    raise ValueError("Invalid output_format. Choose 'DataFrame' or 'List'.")
        except Exception as e:
            print(f"Error fetching comments for council {council}: {e}")
            return None

    def read_all(self, output_format='DataFrame'):
        """Fetch and return all comments from the table in the specified format."""
        select_statement = "SELECT * FROM comments;"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(select_statement)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                if output_format == 'DataFrame':
                    return pd.DataFrame(rows, columns=columns)
                elif output_format == 'List':
                    return rows
                else:
                    raise ValueError("Invalid output_format. Choose 'DataFrame' or 'List'.")
        except Exception as e:
            print(f"Error retrieving comments: {e}")
            return None


    def delete_all(self):
        """Delete all comments from the table with a bit of fun!"""
        user_input = self.delete_confirmation(type=" ALL")
        if user_input == "yes":
            print("Deleting all the records...")

            delete_statement = "DELETE FROM comments;"

            try:
                if self.connection and self.cursor:
                    # Execute the deletion query
                    self.cursor.execute(delete_statement)
                    self.connection.commit()

                    print("All comments have been deleted from the table!")
                else:
                    print("Error: No database connection established. Can't delete anything!")
            except Exception as e:
                print(f"Something went wrong while deleting comments: {e}")
        else:
            print("No comments were deleted.")

    def delete_by_council(self, council):
        """Delete all comments associated with a given council."""
        user_input = self.delete_confirmation(type="")
        if user_input == "yes":
            print("Deleting the record...")

            delete_statement = "DELETE FROM comments WHERE council = %s;"
            try:
                if self.connection and self.cursor:
                    self.cursor.execute(delete_statement, (council,))
                    self.connection.commit()
                    print(f"All comments for council '{council}' have been deleted.")
                else:
                    print("Database connection not established.")
            except Exception as e:
                print(f"Error deleting comments for council {council}: {e}")
        else:
            print("No comment was deleted.")

    def delete_by_id(self, comment_id):
        """Delete a comment by its ID."""
        user_input = self.delete_confirmation(type="")
        if user_input == "yes":
            delete_statement = "DELETE FROM comments WHERE id = %s;"
            try:
                if self.connection and self.cursor:
                    self.cursor.execute(delete_statement, (comment_id,))
                    self.connection.commit()
                    print(f"Comment with ID {comment_id} has been deleted.")
                else:
                    print("Database connection not established.")
            except Exception as e:
                print(f"Error deleting comment with ID {comment_id}: {e}")
        else:
            print("No comment was deleted.")

    def finish(self):
        """Close the database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
                print("Database connection closed.")
        except Exception as e:
            print(f"Error closing the database connection: {e}")

    def delete_confirmation(self, type=""):
        while True:
            user_input = input(f"😱 Are you sure you want to DELETE{type}? This action cannot be undone! (yes/no) 🤔: ").strip().lower()
            if user_input in ['yes', 'no']:
                return user_input
            else:
                print("Oops! That's not a valid answer. Please type 'yes' or 'no' only!")

    def generate_comment_uuid(self):
        # Generate a UUID
        unique_id = uuid.uuid4()

        # Encode the UUID into base64 (without padding to make it shorter)
        short_id = base64.urlsafe_b64encode(unique_id.bytes).decode('utf-8').rstrip("=")

        return short_id

    def check_table_exists(self):
        """Check if the comments table exists in the database."""
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(check_query, (self.table_name,))
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return False


    # Sync with the production
    def sync_prod(self):
        """Sync the dev database with the production database."""
        cs_prod = CommentsSaver("prod")

        # Check if the table exists in the production database
        if not cs_prod.check_table_exists():
            print("Table does not exist in the production database. Creating it...")
            cs_prod.create_table()

        # Fetch all data from dev
        dev_data = self.read_all(output_format='List')

        if dev_data:
            print(f"Syncing {len(dev_data)} records to production...")

            for row in dev_data:
                try:
                    # Unpack row based on the table schema
                    _, council, comment_id, application_id, address, stance, date, comment_text, add_date = row
                    cs_prod.insert_comment(council, comment_id, application_id, address, stance, date, comment_text)
                except Exception as e:
                    print(f"Error syncing comment {comment_id} in application {application_id}): {e}")

            print("Sync complete!")
        else:
            print("No data found in the dev database.")

        # Close the production connection
        cs_prod.finish()


    def check_exists(self, council, application_id):
        """Check if a specific council and application_id combination already exists in the database."""
        
        query = """
        SELECT EXISTS(
            SELECT 1 FROM comments WHERE council = %s AND application_id = %s
        );
        """
        
        try:
            if self.connection and self.cursor:
                with self.connection.cursor() as cursor:
                    cursor.execute(query, (council, application_id))
                    exists = cursor.fetchone()[0]  # Fetch result (True/False)
                    return exists
        except Exception as e:
            print(f"Error checking comment existence: {e}")
            return False  # Assume it doesn't exist if an error occurs
        
    
    def get_comment_count(self, council, application_id):
        """Return the number of comments for a given council and application_id."""
        
        query = """
        SELECT COUNT(*) FROM comments WHERE council = %s AND application_id = %s;
        """
        
        try:
            if self.connection and self.cursor:
                with self.connection.cursor() as cursor:
                    cursor.execute(query, (council, application_id))
                    count = cursor.fetchone()[0]  # Fetch the count value
                    return count
            else:
                print("Database connection not established.")
                return 0  # Return 0 if connection is not available

        except Exception as e:
            print(f"Error retrieving comment count for {application_id}: {e}")
            return 0  # Return 0 in case of an error

        

    def remove_duplicate_comments(self):
        """Removes duplicate comments from the database, keeping only one unique entry per set of values."""
        
        query = """
        DELETE FROM comments
        WHERE comment_id IN (
            SELECT comment_id FROM (
                SELECT 
                    comment_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY council, application_id, address, stance, date, comment_text 
                        ORDER BY comment_id ASC
                    ) AS row_num
                FROM comments
            ) AS duplicates
            WHERE row_num > 1
        );
        """
        
        try:
            if self.connection and self.cursor:
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    deleted_rows = cursor.rowcount  # Get number of deleted rows
                    self.connection.commit()
                    print(f"Removed {deleted_rows} duplicate comments.")
                    return deleted_rows
            else:
                print("Database connection not established.")
                return 0
        except Exception as e:
            print(f"Error removing duplicate comments: {e}")
            return 0  # Return 0 if there was an error
        


    def count_duplicate_comments_by_council(self):
        """Returns a dictionary with the number of duplicated comments grouped by council."""
        
        query = """
        SELECT council, COUNT(*) AS duplicate_count
        FROM (
            SELECT 
                council,
                ROW_NUMBER() OVER (
                    PARTITION BY council, application_id, address, stance, date, comment_text 
                    ORDER BY comment_id ASC
                ) AS row_num
            FROM comments
        ) AS numbered
        WHERE row_num > 1
        GROUP BY council
        ORDER BY duplicate_count DESC;
        """

        try:
            if self.connection and self.cursor:
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    # Convert to dictionary or any desired format
                    duplicates_by_council = {row[0]: row[1] for row in results}
                    return duplicates_by_council
            else:
                print("Database connection not established.")
                return {}
        except Exception as e:
            print(f"Error counting duplicate comments: {e}")
            return {}
    

    
    def update_cleaned_comment_text(self, comment_id, cleaned_comment_text):
        """
        Updates the remote database with the cleaned comment text for a single comment_id.
        """
        # SAFETY: check that column exists only once, cached for future
        if not hasattr(self, "_column_checked"):
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT * FROM comments LIMIT 1;")
                colnames = [desc[0] for desc in cursor.description]
                if 'cleaned_comment_text' not in colnames:
                    raise Exception("Column 'cleaned_comment_text' does not exist in the 'comments' table.")
            self._column_checked = True

        update_sql = """
        UPDATE comments
        SET cleaned_comment_text = %s
        WHERE comment_id = %s;
        """

        retry_attempts = 3

        for attempt in range(1, retry_attempts + 1):
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(update_sql, (cleaned_comment_text, comment_id))
                self.connection.commit()
                break
            except Exception as e:
                print(f"[Attempt {attempt}] Error updating comment_id {comment_id}: {e}")
                if attempt == retry_attempts:
                    print(f"[Failed] Could not update comment_id {comment_id} after {retry_attempts} attempts.")
                else:
                    time.sleep(1)


    def add_new_column(self, table_name, column_name, column_type= "TEXT"):
        """
        Adds a new column to the specified table in the remote database if it doesn't already exist.

        Args:
            table_name (str): Name of the table to modify.
            column_name (str): Name of the new column to add.
            column_type (str, optional): SQL data type of the new column (default is "TEXT").
        """

        table_name = self.table_name if table_name is None else table_name
        
        try:
            with self.connection.cursor() as cursor:
                # Check if the column already exists
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s;
                """, (table_name, column_name))
                column_exists = cursor.fetchone() is not None

                if not column_exists:
                    print(f"Adding column '{column_name}' to table '{table_name}'...")
                    alter_query = f"""ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};"""
                    cursor.execute(alter_query)
                    self.connection.commit()
                    print(f"Column '{column_name}' added successfully.")
                else:
                    print(f"Column '{column_name}' already exists in table '{table_name}'.")
        except Exception as e:
            print(f"Error adding column '{column_name}' to table '{table_name}': {e}")
            raise


