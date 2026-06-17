import sqlite3

try:
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE camera ADD COLUMN is_primary BOOLEAN DEFAULT 1;")
    conn.commit()
    print("Column is_primary added successfully.")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")
finally:
    if conn:
        conn.close()
