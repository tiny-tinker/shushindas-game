import argparse
import uuid
from utils import *

emb_stuff = EmbeddingsDB()


def ingest(filename=None, text=None, title=None):
    doc_name = title 
    if doc_name is None:
        doc_name = filename.split("/")[-1] if filename is not None else None
    if doc_name is None:
        doc_name = str(uuid.uuid4())
    
    print( f"Ingesting '{doc_name}'")

    ##############
    # Retrieve text
    all_text = []
    all_text = emb_stuff.get_text(filename, text)

    if all_text is None or len(all_text) == 0:
        print( f"   No text found! Exiting.")
        return
    
    print( f"   Got text! '{all_text[0][:50]}'")

    ##############
    # Insert text

    # TODO: check length of `all_text` < 10 MB
    # TODO: update instead of always insert? Or just check for duplicate `doc_name`?
    rows_to_insert = [
        {u"id": str(uuid.uuid4()), u"doc_name": doc_name, u"text": "".join(all_text)}
    ]

    # Do we want to store the entire docs? 
    # emb_stuff.insert_recs( rows_to_insert )
    # print( f"   Inserted all text!")


    ##############
    # Chunk text
    all_chunks = emb_stuff.get_chunks(all_text)
    print( f"   Chunked all text! '{all_chunks[0][:30]}'...")

    ##############
    # Insert chunks
    # table_id = f"{DATASET}.chunks"
    rows_to_insert = []
    chunk_id = 0
    doc_id = str(uuid.uuid4())
    for chunk in all_chunks:
        embeddings = emb_stuff.get_embeddings(chunk)
        rows_to_insert.append( 
            {u"id": doc_id + str(chunk_id), u"chunk_id": chunk_id, u"doc_name": doc_name, u"chunk_text": chunk, u"chunk_vector": embeddings}
        )
        chunk_id = chunk_id + 1
    
    emb_stuff.insert_recs( rows_to_insert )

    ##############
    # Fin

def main(args):
    print('The filename:, %s!' % args.file)
    ingest(args.file, args.text, args.title)

# python ingest.py --file=../docs/some_words_about_dragons.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Ingest some docs')
    parser.add_argument('-f', '--file', help='The filename and path of the file to ingest')
    parser.add_argument('-t', '--text', help='The actual text of the file to ingest. Optional')
    parser.add_argument('-ti', '--title', help='The title of the document. If not set, the doc name will be the filename or a UUID. Optional')
    args = parser.parse_args()

    main(args)