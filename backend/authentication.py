import bcrypt
import sqlite3


class AuthenticationDB:
    def __init__(self, db_name="cleanspeak.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS authentication (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                fullname TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
        """
        )
        self.conn.commit()

    def add_user(self, username, fullname, email, phone, password):
        cursor = self.conn.cursor()
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        cursor.execute(
            "INSERT INTO authentication (username, password_hash, fullname, email, phone) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, fullname, email, phone),
        )
        self.conn.commit()

    def update_password(self, username, new_password):
        cursor = self.conn.cursor()
        new_password_hash = bcrypt.hashpw(
            new_password.encode("utf-8"), bcrypt.gensalt()
        )

        cursor.execute(
            "UPDATE authentication SET password_hash = ? WHERE username = ?",
            (new_password_hash, username),
        )
        self.conn.commit()

    def verify_user(self, username, password):
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT password_hash FROM authentication WHERE username = ?", (username,)
        )
        result = cursor.fetchone()

        if result:
            stored_password_hash = result[0]

            if bcrypt.checkpw(password.encode("utf-8"), stored_password_hash):
                return True
            else:
                return False

        else:
            return None
