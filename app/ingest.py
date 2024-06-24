import argparse
import time
import PyPDF2
import hashlib
import uuid
import spacy
import os

import vertexai
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel


from google.cloud import bigquery

DATASET = "shushindas_stuff"


PROJECT_ID = os.getenv("GCP_PROJECT")

# Construct a BigQuery client object.
bq_client = bigquery.Client()
nlp = spacy.load("en_core_web_md")

vertexai.init(project=PROJECT_ID, location="us-central1") 
emb_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@002")
txt_model = TextGenerationModel("gemini-1.0-pro-001")
                 
def get_text(the_filename=None, the_text=None):
    '''
    Returns an array of text.

            Parameters:
                    the_filename (str): The filename and path of the file to ingest Optional.
                    the_text (str): The actual text of the file to ingest. Optional.
            Returns:
                    all_text (array[str]): An array of text
    '''
    all_text = []

    if the_text is not None:
        all_text = [the_text]
        return all_text

    if the_filename.endswith( '.pdf' or '.PDF'):
        all_text = []
        with open(the_filename, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                all_text.append(page_text)
        return all_text
    
    elif the_filename.endswith( '.txt' or '.TXT'):
        with open(the_filename, 'r') as txt_file:
            all_text = [txt_file.read()]
            return all_text

    return all_text

def get_chunks(text):
    '''
    Returns an array of chunks from a large body of text.

            Parameters:
                    text (array[str]): A large body of text
            Returns:
                    chunks (array[str]): An array of chunks
    '''
    #SpaCy
    doc = nlp("".join(text))
    chunks = [chunk.text for chunk in doc.sents]
    return chunks

def get_embeddings( chunk ):
    '''
    Returns an array of embedding vectors using `emb_model`.

            Parameters:
                    chunk (str): A chunk of text to generate embeddings for
            Returns:
                    embs (array): An array of embedding vectors
    '''
    print('      Getting embeddings...')
    embs = []
    # time.sleep(1)  # to avoid the quota error
    result = emb_model.get_embeddings([chunk])
    embs = result[0].values
    return embs


def insert_bq(table_id, rows_to_insert):
    '''
    Inserts rows into the given BigQuery table.

            Parameters:
                    table_id (str): The BigQuery table to insert rows into
                    rows_to_insert (array[dict]): The rows to insert
            Returns:
                    errors (array): The errors encountered while inserting rows (if any)
    '''
    batch_size = 100

    errors = []
    for i in range(0, len(rows_to_insert), batch_size):
        batch = rows_to_insert[i:i+batch_size]
        errors = bq_client.insert_rows_json(table_id, batch)

    
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))
    
    return errors

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
    all_text = get_text(filename, text)

    if all_text is None or len(all_text) == 0:
        print( f"   No text found! Exiting.")
        return
    
    print( f"   Got text! '{all_text[0][:50]}'")

    ##############
    # Insert text
    table_id = f"{DATASET}.docs"

    # TODO: check length of `all_text` < 10 MB
    # TODO: update instead of always insert? Or just check for duplicate `doc_name`?
    rows_to_insert = [
        {u"id": str(uuid.uuid4()), u"doc_name": doc_name, u"text": "".join(all_text)}
    ]
    # insert_bq( table_id, rows_to_insert)
    print( f"   Inserted all text!")


    ##############
    # Chunk text
    all_chunks = get_chunks(all_text)
    print( f"   Chunked all text! '{all_chunks[0][:30]}'...")

    ##############
    # Insert chunks
    table_id = f"{DATASET}.chunks"
    rows_to_insert = []
    chunk_id = 0
    for chunk in all_chunks:
        embeddings = get_embeddings(chunk)
        rows_to_insert.append(
            {u"id": chunk_id, u"doc_name": doc_name, u"chunk_text": chunk, u"chunk_vector": embeddings}
        )
        chunk_id = chunk_id + 1

    insert_bq( table_id, rows_to_insert)

    ##############
    # Fin

def main(args):
    print('The filename:, %s!' % args.file)
    ingest(args.file, args.text, args.title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Ingest some docs')
    parser.add_argument('-f', '--file', help='The filename and path of the file to ingest')
    parser.add_argument('-t', '--text', help='The actual text of the file to ingest. Optional')
    parser.add_argument('-ti', '--title', help='The title of the document. If not set, the doc name will be the filename or a UUID. Optional')
    args = parser.parse_args()

    main(args)