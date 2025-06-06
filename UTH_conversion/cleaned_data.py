import sqlite3
import json

def build_training_jsonl(output_path="training_data.jsonl", db_path="intents.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT domain, action_text, inferred_intent FROM intents
        WHERE inferred_intent IS NOT NULL AND label_source = 'pseudo'
    """).fetchall()

    with open(output_path, 'w') as f:
        for domain, action, label in rows:
            json.dump({
                "input": f"{domain} - {action}",
                "label": label
            }, f)
            f.write("\n")

    conn.close()