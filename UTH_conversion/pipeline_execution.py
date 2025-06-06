import os
from file_converter import convert_jsonl_to_sqlite
from self_labeling import pseudo_label_all
from cleaned_data import build_training_jsonl
from bm25_filter import bm25_filter
import os

def main():

    input_path = "../engenium/intents.jsonl"
    print("Starting pipeline execution...")

    if not os.path.exists(input_path):
        print("Crawled data not found. Run the crawler first.")
        return
    
    print("Converting JSONL to SQLite...")
    convert_jsonl_to_sqlite("../engenium/intents.jsonl")

    print("applying bm25")
    bm25_filter()

    print("applying self-labeling")
    pseudo_label_all()

    print("building training data")
    build_training_jsonl()

    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
