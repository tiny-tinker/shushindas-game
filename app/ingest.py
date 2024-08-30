import argparse
import uuid
from utils import *

emb_stuff = EmbeddingsDB()

def ingest(filename=None, text=None, title=None):
    """Ingest a document or text into the embeddings database.

    This function handles the ingestion of a file or text into the embeddings database.
    It retrieves text from the file or directly from input, chunks the text, generates embeddings,
    and stores the resulting data in the vector database.

    Args:
        filename (str, optional): The filename and path of the file to ingest.
        text (str, optional): The actual text content to ingest.
        title (str, optional): The title of the document. If not provided, it defaults to the filename or a UUID.

    Returns:
        None
    """
    # Determine document name
    doc_name = title or (filename.split("/")[-1] if filename else str(uuid.uuid4()))
    print(f"Ingesting '{doc_name}'")

    # Retrieve text
    all_text = emb_stuff.get_text(filename, text)
    if not all_text:
        print(f"   No text found! Exiting.")
        return

    print(f"   Got text! '{all_text[0][:50]}'")

    # Insert text (currently not storing entire docs; code commented out)
    # TODO: Check length of `all_text` < 10 MB
    # TODO: Update instead of always insert? Or check for duplicate `doc_name`?
    rows_to_insert = [
        {u"id": str(uuid.uuid4()), u"doc_name": doc_name, u"text": "".join(all_text)}
    ]

    # Uncomment to store the entire documents in the vector database
    # emb_stuff.insert_recs(rows_to_insert)
    # print(f"   Inserted all text!")

    # Chunk text
    all_chunks = emb_stuff.get_chunks(all_text)
    print(f"   Chunked all text! '{all_chunks[0][:30]}'...")

    # Insert chunks
    rows_to_insert = []
    doc_id = str(uuid.uuid4())
    for chunk_id, chunk in enumerate(all_chunks):
        embeddings = emb_stuff.get_embeddings(chunk)
        rows_to_insert.append(
            {
                u"id": doc_id + str(chunk_id),
                u"chunk_id": chunk_id,
                u"doc_name": doc_name,
                u"chunk_text": chunk,
                u"chunk_vector": embeddings
            }
        )

    emb_stuff.insert_recs(rows_to_insert)

    print("   Finished ingesting document!")

def main(args):
    """Main function to parse arguments and start the ingestion process.

    Args:
        args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        None
    """
    print(f'The filename: {args.file}!')
    ingest(args.file, args.text, args.title)

# Command-line entry point
# Example: python ingest.py --file=../docs/some_words_about_dragons.txt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest some docs')
    parser.add_argument('-f', '--file', help='The filename and path of the file to ingest')
    parser.add_argument('-t', '--text', help='The actual text of the file to ingest. Optional')
    parser.add_argument('-ti', '--title', help='The title of the document. If not set, the doc name will be the filename or a UUID. Optional')
    args = parser.parse_args()

    main(args)