#db.py

import sqlite3
import os
import uuid

DB_PATH = "data/chat_memory.db"


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Conversations table
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Messages table
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Add summary column if it doesn't exist
    c.execute("""
        PRAGMA table_info(conversations)
    """)
    columns = [col[1] for col in c.fetchall()]

    if "summary" not in columns:
        c.execute("""
            ALTER TABLE conversations
            ADD COLUMN summary TEXT
        """)

    conn.commit()
    conn.close()


def create_conversation(title: str = "New Chat") -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    session_id = str(uuid.uuid4())

    c.execute(
        "INSERT INTO conversations (id, title) VALUES (?, ?)",
        (session_id, title)
    )

    conn.commit()
    conn.close()

    return session_id


def get_conversations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT id, title, last_updated
        FROM conversations
        ORDER BY last_updated DESC
    """)

    rows = c.fetchall()
    conn.close()

    return rows


def update_conversation_activity(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        UPDATE conversations
        SET last_updated = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (session_id,))

    conn.commit()
    conn.close()


def update_conversation_title(session_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        UPDATE conversations
        SET title = ?
        WHERE id = ?
    """, (title, session_id))

    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )

    conn.commit()
    conn.close()

    update_conversation_activity(session_id)


def load_messages(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    )

    rows = c.fetchall()
    conn.close()

    return rows


def delete_conversation(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Delete messages first
    c.execute(
        "DELETE FROM messages WHERE session_id = ?",
        (session_id,)
    )

    # Delete conversation
    c.execute(
        "DELETE FROM conversations WHERE id = ?",
        (session_id,)
    )

    conn.commit()
    conn.close()



def search_conversations(query: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT id, title, last_updated
        FROM conversations
        WHERE title LIKE ?
        ORDER BY last_updated DESC
    """, (f"%{query}%",))

    rows = c.fetchall()
    conn.close()

    return rows



def get_conversation_summary(session_id: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT summary FROM conversations WHERE id=?",
        (session_id,)
    )

    row = c.fetchone()
    conn.close()

    if row:
        return row[0]
    return None


def update_conversation_summary(session_id: str, summary: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "UPDATE conversations SET summary=? WHERE id=?",
        (summary, session_id)
    )

    conn.commit()
    conn.close()