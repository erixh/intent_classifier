import sqlite3
from tqdm import tqdm


def build_prompt(domain, action_text):
    return f"what is the user's likely intent if they are on {domain} and click {action_text}?"


def mock_label_with_mistral(prompt):
    # Dummy classifier
    if "search" in prompt.lower():
        return "search_product"
    elif "add" in prompt.lower():
        return "add_to_cart"
    else:
        return "navigate"


def pseudo_label_all(db_path="intents.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, domain, action_text FROM intents WHERE inferred_intent IS NULL
    """).fetchall()

    for row in tqdm(rows, desc="Labeling intents"):
        id, domain, action = row
        prompt = build_prompt(domain, action)
        inferred_intent = mock_label_with_mistral(prompt)

        cur.execute("""
            UPDATE intents SET inferred_intent = ?, label_source = ? WHERE id = ?
        """, (inferred_intent, "pseudo", id))

    conn.commit()
    conn.close()

