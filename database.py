import sqlite3

DB_FILE = "app.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        category TEXT,
        word_count INTEGER
    )
    """)

    # Insert sample data
    cursor.execute("DELETE FROM documents")

    cursor.executemany("""
    INSERT INTO documents (title, category, word_count)
    VALUES (?, ?, ?)
    """, [
        ("LangChain RAG Guide", "RAG", 2500),
        ("Vector DB Guide", "Database", 1800),
        ("LangGraph Agent Guide", "Agent", 2000),
    ])

    conn.commit()
    conn.close()


def run_query(sql):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return rows