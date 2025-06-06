import json
import sqlite3
import os
from tqdm import tqdm

def convert_jsonl_to_sqlite(input_file, db_path = "intents.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        url TEXT,
        page_title TEXT,
        action_text TEXT,
        bm25_score REAL,
        confidence_score REAL,
        label_source TEXT,
        inferred_intent TEXT
    );
    """
    )

    with open(input_file, "r") as f:
        for line in tqdm(f, desc="Inserting Rows"):
            item = json.loads(line)

            for action in item.get("visible_actions", []):
                cursor.execute("""
                INSERT INTO intents (domain, url, page_title, action_text, bm25_score, confidence_score, label_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                item.get("domain"), item.get("url"), item.get("page_title"), action, item.get("bm25_score", None), item.get("confidence_score", None), "raw"
                )
                )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    convert_jsonl_to_sqlite("../engenium/intents.jsonl")
