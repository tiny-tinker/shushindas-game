

import os
import random
import yaml
import numpy as np

import PyPDF2

from google.cloud import aiplatform
from google.cloud import bigquery
import vertexai

from vertexai.language_models import TextGenerationModel, TextEmbeddingModel

from vertexai.generative_models import GenerativeModel, Part, FinishReason, Candidate
from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec

import spacy

import weave

# Helper function that reads from the config file. 
def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    """
    try:
        return config[section][key]
    except:
        return default

# Open the config file (config.yaml)
with open('./config.yaml') as f:
    config = yaml.safe_load(f)

# Read application variables from the config fle
TITLE = get_config_value(config, 'app', 'title', 'The Desk of Shushinda')
SUBTITLE = get_config_value(config, 'app', 'subtitle', 'Beware of librarian, she\'s a real cat-astrophy!')
CONTEXT = get_config_value(config, 'palm', 'context',
                           'You are Shushinda Hushwisper, the infamous fictional cat wizard who lives in Discworld.')
BOTNAME = get_config_value(config, 'palm', 'botname', 'Shushinda')
TEMPERATURE = get_config_value(config, 'palm', 'temperature', 0.8)
MAX_OUTPUT_TOKENS = get_config_value(config, 'palm', 'max_output_tokens', 256)
TOP_P = get_config_value(config, 'palm', 'top_p', 0.8)
TOP_K = get_config_value(config, 'palm', 'top_k', 40)

PREAMBLE = get_config_value(config, 'shushinda', 'preamble')
MY_BIO = get_config_value(config, 'shushinda', 'my_bio')
PERSONALITY_TRAITS = get_config_value(config, 'shushinda', 'personality_traits')
SOME_FACTS = get_config_value(config, 'shushinda', 'some_facts')

SHUSHINDA = PREAMBLE + MY_BIO + PERSONALITY_TRAITS + SOME_FACTS

GREETINGS = get_config_value(config, 'shushinda', 'greetings' )

ALL_SAMPLE_QUESTIONS = get_config_value(config, 'shushinda', 'sample_questions')

COULD_NOT_ANSWER = get_config_value( config, 'shushinda', 'say_what')

UNLOCK_QUESTION = "What is the Trial?"


LLMS = [
    {"model": "gpt-4o-mini", "family": "openai"},
    {"model": "gemini-1.5-flash-001", "family": "gemini" }
    
]

class SystemPrompt(weave.Object):
    prompt: str

class LanguageModel:
    def __init__(self, llm_name):
    
        llm = next( llm for llm in LLMS if llm["model"] == llm_name )
    
        print( f"Model: {llm['family']}")
        self.llm_fam = llm["family"]
        self.llm_model_name = llm["model"]
    
        self.SYSTEM_PROMPT = SystemPrompt( prompt=SHUSHINDA )
        weave.publish(self.SYSTEM_PROMPT)


        if self.llm_fam == "gemini":

            # TODO: Should probably get this from config.gcp.project_id or something
            PROJECT_ID = os.getenv("PROJECT")
            REGION     = os.getenv( "REGION" )
            vertexai.init( project=PROJECT_ID, location=REGION ) 
            
            self.txt_model = GenerativeModel(self.llm_name)

        elif self.llm_fam == "openai":
            # self.model_name = "gpt-4o-mini"

            # Is this really useful? ¯\_(ツ)_/¯
            client = OpenAI()
            self.txt_model = client

            # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # @weave.op() # Having this here will only log this block. 
    def predict(self, question, context ):

        if self.llm_fam == "gemini":
            return self.ask_gemini( question, context)
        if self.llm_fam == "openai":
            return self.ask_openai( question, context)

    @weave.op()
    def ask_openai( self, question, data ):

        # PROMPT = f"""
        # {SHUSHINDA}
        # CONTEXT: {data}
        # QUESTION: {question}
        # """
        messages=[
                {"role": "system", "content":  self.SYSTEM_PROMPT.prompt },
                {"role": "user", "content": question },
                {"role": "assistant", "content": data },
            ]
        
        response = self.txt_model.chat.completions.create(
            model=self.llm_model_name,
            messages=messages
            )
        # print( f"openai response: {response}" )
        if response.choices[0].finish_reason != "stop":
            return random.choice(COULD_NOT_ANSWER)
        else:
            return response.choices[0].message.content
            
    @weave.op()
    def ask_gemini(self, question, data):
        # You will need to change the code below to ask Gemni to
        # answer the user's question based on the data retrieved
        # from their search

        PROMPT = f"""
        {self.SYSTEM_PROMPT.prompt}
        CONTEXT: {data}
        QUESTION: {question}
        """
        
        response = self.txt_model.generate_content(
            [PROMPT],
            stream=False,
        )
        # print( f"PROMPT: {PROMPT[:400]}...")
        # print( f"Full response: {response}")

        if response.candidates[0].finish_reason != FinishReason.STOP:
            return random.choice(COULD_NOT_ANSWER)
        else:
            return response.text

# TODO: Finish building this out. 
class EmbeddingsDB:
    def __init__(self, emb_model_fam="openai", vector_db="pinecone"):
        
        self.emb_model = None
        self.model_fam = emb_model_fam
        self.dimensions = 768

        self.vector_db = vector_db

        self.nlp = spacy.load("en_core_web_md")

        if emb_model_fam == "gecko":
            self.emb_model_name = "text-embedding-004"
            
            self.emb_model = TextEmbeddingModel.from_pretrained( self.emb_model_name )

        elif emb_model_fam == "openai":
            self.emb_model_name = "text-embedding-3-small"
            client = OpenAI()
            self.emb_model = client.embeddings
            self.dimensions = 1536
    
        if vector_db == "pinecone":
            self.vector_db = "pinecone"

            pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
            self.pc = pc
            self.pc_index_name = "shushinda-index"
            self.namespace = "the_library"

            if self.pc_index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=self.pc_index_name,
                    dimension=self.dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws', 
                        region='us-east-1'
                    ) 
                )
            

        elif vector_db == "bigquery":
            self.vector_db = "bigquery"
            self.bq_client = bigquery.Client()

        print( f"Emb Model: {emb_model_fam}")

    def get_text(self, the_filename=None, the_text=None):
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
        
        elif the_filename.endswith( '.txt' ) or \
            the_filename.endswith('.TXT' ) or \
            the_filename.endswith('.html' ) or \
            the_filename.endswith( '.htm'):
            with open(the_filename, 'r') as txt_file:
                all_text = [txt_file.read()]
                return all_text

        return all_text

    def get_chunks(self, text):
        '''
        Returns an array of chunks from a large body of text.

                Parameters:
                        text (array[str]): A large body of text
                Returns:
                        chunks (array[str]): An array of chunks
        '''
        #SpaCy
        doc = self.nlp("".join(text))
        chunks = [chunk.text for chunk in doc.sents]
        return chunks

    def get_embeddings(self, chunk ):
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
        if self.model_fam == "gecko":
            result = self.emb_model.get_embeddings([chunk])
            embs = result[0].values

        elif self.model_fam == "openai":

            chunk = chunk.replace("\n", " ")
            embs = self.emb_model.create(input = [chunk], model=self.emb_model_name).data[0].embedding

        return embs

    def search_vector_database(self, question):

        # 1. Convert the question into an embedding
        # 2. Search the Vector database for the 5 closest embeddings to the user's question
        # 3. Get the IDs for the five embeddings that are returned
        # 4. Get the five documents from Firestore that match the IDs
        # 5. Concatenate the documents into a single string and return it

        the_embeddings = self.get_embeddings(question)
        
        if self.vector_db == "bq":
            data = self.search_bq( the_embeddings)
        if self.vector_db == "pinecone":
            data = self.search_pinecone( the_embeddings)

        return data
    @weave.op()
    def search_bq(self, the_embeddings):
                    # Search BQ. Something like this:
            query = f"""
            SELECT base.id, base.doc_name, base.chunk_text, distance 
            FROM VECTOR_SEARCH(
                TABLE sushindas_stuff.chunks, 
                'chunk_vector',
                (select @search_embedding as embedding),
                'embedding',
                top_k => 2,
                distance_type => 'EUCLIDEAN' -- change to COSINE or EUCLIDEAN
                )
            ORDER BY distance ASC;
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("search_embedding", "FLOAT", the_embeddings),
                ]
            )
            query_job = bq_client.query(query, job_config=job_config)  # Make an API request.

            for row in query_job:
                data = data + row.chunk_text
                # print( f"id: {row.id}, doc_name: {row.doc_name}, chunk_text: {row.chunk_text}, distance: {row.distance}")

            return data

    @weave.op()
    def search_pinecone( self, the_embeddings ):
        # https://docs.pinecone.io/guides/get-started/quickstart#8-run-a-similarity-search
        index = self.pc.Index(self.pc_index_name)

        results = index.query(
            namespace=self.namespace,
            vector=the_embeddings,
            top_k=3,
            include_values=True,
            include_metadata=True
        )

        for r in results['matches']:
            print( f"results: {round( r['score'], 2 )}: {r['metadata']}" )
            # print( f"stuff: {r['score']}")
        return ""
    

    def insert_recs( self, rows_to_insert ):
        if self.vector_db == "pinecone":
            self.insert_pinecone( rows_to_insert )
        elif self.vector_db == "bigquery":
            self.insert_bq( rows_to_insert )


    def insert_pinecone(self, rows_to_insert ):
        index = self.pc.Index(self.pc_index_name)

        pc_rows = [ 
            { "id": str(r["id"]), 
              "values": r["chunk_vector"], 
              "metadata": {
                  "doc_name": r["doc_name"], 
                  "chunk_text": r["chunk_text"],
                  "chunk_id": r["chunk_id"]
              }
            } for r in rows_to_insert
        ]

        # This is to break up the insert into 20 row chunks
        list_of_rows = np.array_split( pc_rows, 20 )

        for l in list_of_rows:
            
            pc_response = index.upsert(
                vectors=l,
                namespace=self.namespace
            )

        print( f" {pc_response} ")
    def insert_bq( self, rows_to_insert):
        '''
        Inserts rows into the given BigQuery table.

                Parameters:
                        table_id (str): The BigQuery table to insert rows into
                        rows_to_insert (array[dict]): The rows to insert
                Returns:
                        errors (array): The errors encountered while inserting rows (if any)
        '''
        batch_size = 100
        dataset = "shushindas_stuff"
        table_id = f"{dataset}.docs"

        errors = []
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i+batch_size]
            errors = self.bq_client.insert_rows_json(table_id, batch)

        
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
        
        return errors
    
    def get_doc_names(self):

        index = self.pc.Index(self.pc_index_name)

        the_ids = []
        for ids in index.list(namespace=self.namespace):
            the_ids.extend( ids)

        # Assume `the_ids` is a list of all your IDs
        list_of_ids = np.array_split(the_ids, 1000)  # Split IDs into chunks of 1000

        # Initialize a set to collect unique doc_name values
        unique_doc_names = set()

        for l in list_of_ids:
            # Fetch data for the current chunk
            resp = index.fetch(ids=l.tolist(), namespace=self.namespace)

            # Iterate over the fetched items and extract doc_name values
            for item_id, item_data in resp['vectors'].items():
                # Check if the item contains metadata and a 'doc_name' field
                if 'metadata' in item_data and 'doc_name' in item_data['metadata']:
                    # Add the doc_name to the set
                    unique_doc_names.add(item_data['metadata']['doc_name'])

        # Convert the set to a sorted list (optional)
        unique_doc_names = sorted(list(unique_doc_names))

        # Print or use the unique_doc_names list
        return unique_doc_names
