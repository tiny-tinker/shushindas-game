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
from pydantic import BaseModel, Field
from typing import Union
import spacy
import weave

# Helper function that reads from the config file.
def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    
    Args:
        config (dict): The loaded configuration from a YAML file.
        section (str): The section in the configuration to search.
        key (str): The key to retrieve the value for.
        default (Optional): The default value to return if the key is not found.
        
    Returns:
        The value from the configuration or the default if not found.
    """
    try:
        return config[section][key]
    except:
        return default

# Open the config file (config.yaml)
with open('./config.yaml') as f:
    config = yaml.safe_load(f)

# Read application variables from the config file
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

GREETINGS = get_config_value(config, 'shushinda', 'greetings')

ALL_SAMPLE_QUESTIONS = get_config_value(config, 'shushinda', 'sample_questions')

COULD_NOT_ANSWER = get_config_value(config, 'shushinda', 'say_what')

UNLOCK_QUESTION = "What is the Trial?"

LLMS = [
    {"model": "gpt-4o-mini", "family": "openai"},
    {"model": "gpt-4o", "family": "openai"},
    {"model": "gpt-4-turbo", "family": "openai"},
    {"model": "gemini-1.5-flash-001", "family": "gemini"}
]

class SystemPrompt(weave.Object):
    """Class to define a system prompt in Weave."""
    prompt: str

class LanguageModel(weave.Model):
    """Class representing a Language Model with various configurations and operations."""

    name: str = Field("gpt-4o-mini", description="The name of the model. Used for nice display in Weave UI.")
    llm_fam: str = Field("openai", description="The family of the language model")
    llm_model_name: str = Field("gpt-4o-mini", description="The specific model name of the LLM")
    system_prompt: str = Field("Shushinda Hushwisper", description="The system prompt used by the LLM")
    # txt_model: Union[GenerativeModel, OpenAI] = Field(None, description="The model client to then .predict")

    def __init__(self, name: str, llm_name: str, prompt: str = SHUSHINDA):
        """Initialize the LanguageModel with configurations."""
        super().__init__()

        # Find the LLM configuration based on the provided llm_name
        llm = next(llm for llm in LLMS if llm["model"] == llm_name)

        print(f"Model: {llm['family']}")
        self.llm_fam = llm["family"]
        self.llm_model_name = llm["model"]
        self.name = name

        self.system_prompt = SystemPrompt(prompt=prompt)
        weave.publish(self.system_prompt)

    @weave.op
    def predict(self, question: str, context: str = None):
        """Predict the response based on the input question and context.

        Args:
            question (str): The input question.
            context (str, optional): Additional context for the model.

        Returns:
            dict: Response containing the model's output and call ID.
        """
        resp: str = ""
        current_call = weave.get_current_call()

        if context is None:
            emb_stuff = EmbeddingsDB()
            # Get context data to answer the question based on embeddings
            context = emb_stuff.search_vector_database(question)

        if self.llm_fam == "gemini":
            resp = self.ask_gemini(question, context)
        elif self.llm_fam == "openai":
            resp = self.ask_openai(question, context)
        return {"response": resp, "call_id": current_call.id}

    @weave.op
    def ask_openai(self, question: str, data: str):
        """Request a response from the OpenAI model.

        Args:
            question (str): The input question.
            data (str): Additional data or context.

        Returns:
            str: The model's response or a fallback if the response is incomplete.
        """
        client = OpenAI()
        txt_model = client

        messages = [
            {"role": "system", "content": self.system_prompt.prompt},
            {"role": "user", "content": question}
        ]

        if data is not None:
            messages.extend([{"role": "assistant", "content": data}])

        response = txt_model.chat.completions.create(
            model=self.llm_model_name,
            messages=messages
        )

        if response.choices[0].finish_reason != "stop":
            return random.choice(COULD_NOT_ANSWER)
        else:
            return response.choices[0].message.content

    @weave.op
    def ask_gemini(self, question: str, data: str):
        """Request a response from the Gemini model.

        Args:
            question (str): The input question.
            data (str): Additional data or context.

        Returns:
            str: The model's response or a fallback if the response is incomplete.
        """
        PROJECT_ID = os.getenv("PROJECT")
        REGION = os.getenv("REGION")
        vertexai.init(project=PROJECT_ID, location=REGION)
        txt_model = GenerativeModel(self.llm_model_name)
        
        PROMPT = f"""
        {self.system_prompt.prompt}
        CONTEXT: {data}
        QUESTION: {question}
        """

        response = txt_model.generate_content(
            [PROMPT],
            stream=False,
        )

        if response.candidates[0].finish_reason != FinishReason.STOP:
            return random.choice(COULD_NOT_ANSWER)
        else:
            return response.text
class EmbeddingsDB:
    """Class to manage embedding models and vector databases for text processing."""

    def __init__(self, emb_model_fam="openai", vector_db="pinecone"):
        """Initialize the EmbeddingsDB with model family and vector database configurations.

        Args:
            emb_model_fam (str): The embedding model family to use ("openai" or "gecko").
            vector_db (str): The vector database to use ("pinecone" or "bigquery").
        """
        self.emb_model = None
        self.model_fam = emb_model_fam
        self.dimensions = 768
        self.vector_db = vector_db
        self.nlp = spacy.load("en_core_web_md")

        if emb_model_fam == "gecko":
            self.emb_model_name = "text-embedding-004"
            self.emb_model = TextEmbeddingModel.from_pretrained(self.emb_model_name)
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

        print(f"Emb Model: {emb_model_fam}")

    def get_text(self, the_filename=None, the_text=None):
        """Returns an array of text from a file or directly provided text.

        Args:
            the_filename (str, optional): The filename and path of the file to ingest.
            the_text (str, optional): The actual text to ingest.

        Returns:
            list[str]: An array of text.
        """
        all_text = []

        if the_text is not None:
            all_text = [the_text]
            return all_text

        if the_filename and the_filename.endswith(('.pdf', '.PDF')):
            with open(the_filename, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    all_text.append(page_text)
            return all_text

        elif the_filename and any(the_filename.endswith(ext) for ext in ['.txt', '.TXT', '.html', '.htm']):
            with open(the_filename, 'r') as txt_file:
                all_text = [txt_file.read()]
            return all_text

        return all_text

    def get_chunks(self, text):
        """Returns an array of chunks from a large body of text.

        Args:
            text (list[str]): A large body of text.

        Returns:
            list[str]: An array of chunks.
        """
        # SpaCy for sentence segmentation
        doc = self.nlp("".join(text))
        chunks = [chunk.text for chunk in doc.sents]
        return chunks

    def get_embeddings(self, chunk):
        """Returns an array of embedding vectors using the selected embedding model.

        Args:
            chunk (str): A chunk of text to generate embeddings for.

        Returns:
            list: An array of embedding vectors.
        """
        print('      Getting embeddings...')
        embs = []

        # time.sleep(1)  # Uncomment if rate limiting is needed
        if self.model_fam == "gecko":
            result = self.emb_model.get_embeddings([chunk])
            embs = result[0].values
        elif self.model_fam == "openai":
            chunk = chunk.replace("\n", " ")
            embs = self.emb_model.create(input=[chunk], model=self.emb_model_name).data[0].embedding

        return embs

    @weave.op
    def search_vector_database(self, question):
        """Search the vector database for the closest embeddings to the user's question.

        Args:
            question (str): The user's question.

        Returns:
            str: The concatenated documents that match the query.
        """
        the_embeddings = self.get_embeddings(question)

        if self.vector_db == "bq":
            data = self.search_bq(the_embeddings)
        elif self.vector_db == "pinecone":
            data = self.search_pinecone(the_embeddings)

        return data

    @weave.op
    def search_bq(self, the_embeddings):
        """Searches BigQuery for the closest matching documents.

        Args:
            the_embeddings (list): The embeddings of the query.

        Returns:
            str: Concatenated results from the search.
        """
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
        query_job = self.bq_client.query(query, job_config=job_config)  # Make an API request.

        data = ""
        for row in query_job:
            data += row.chunk_text
            # print(f"id: {row.id}, doc_name: {row.doc_name}, chunk_text: {row.chunk_text}, distance: {row.distance}")

        return data

    @weave.op
    def search_pinecone(self, the_embeddings):
        """Searches Pinecone for the closest matching documents.

        Args:
            the_embeddings (list): The embeddings of the query.

        Returns:
            str: Concatenated results from the search.
        """
        # Reference: https://docs.pinecone.io/guides/get-started/quickstart#8-run-a-similarity-search
        index = self.pc.Index(self.pc_index_name)

        results = index.query(
            namespace=self.namespace,
            vector=the_embeddings,
            top_k=3,
            include_values=True,
            include_metadata=True
        )
        
        # Initialize a list to store the text of top 3 vectors
        top_texts = []

        # Extract the 'text' from metadata of the top 3 matches
        for match in results['matches']:
            if 'metadata' in match and 'chunk_text' in match['metadata']:
                top_texts.append(match['metadata']['chunk_text'])

        return ''.join(top_texts)

    def insert_recs(self, rows_to_insert):
        """Insert records into the vector database based on the current configuration.

        Args:
            rows_to_insert (list[dict]): The records to insert.
        """
        if self.vector_db == "pinecone":
            self.insert_pinecone(rows_to_insert)
        elif self.vector_db == "bigquery":
            self.insert_bq(rows_to_insert)

    def insert_pinecone(self, rows_to_insert):
        """Insert records into the Pinecone vector database.

        Args:
            rows_to_insert (list[dict]): The records to insert.
        """
        index = self.pc.Index(self.pc_index_name)

        pc_rows = [
            {"id": str(r["id"]),
             "values": r["chunk_vector"],
             "metadata": {
                 "doc_name": r["doc_name"],
                 "chunk_text": r["chunk_text"],
                 "chunk_id": r["chunk_id"]
             }
             } for r in rows_to_insert
        ]

        # This breaks up the insert into 20-row chunks
        list_of_rows = np.array_split(pc_rows, 20)

        for l in list_of_rows:
            pc_response = index.upsert(
                vectors=l,
                namespace=self.namespace
            )

        print(f" {pc_response} ")

    def insert_bq(self, rows_to_insert):
        """Inserts rows into the given BigQuery table.

        Args:
            rows_to_insert (list[dict]): The rows to insert.

        Returns:
            list: Errors encountered while inserting rows, if any.
        """
        batch_size = 100
        dataset = "shushindas_stuff"
        table_id = f"{dataset}.docs"

        errors = []
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i + batch_size]
            errors = self.bq_client.insert_rows_json(table_id, batch)

        if not errors:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))

        return errors

    def get_doc_names(self):
        """Retrieve unique document names from the Pinecone index.

        This function fetches document names stored in the Pinecone vector index.
        It first lists all vector IDs, then retrieves the metadata for each vector 
        to extract document names. The document names are stored in a set to ensure 
        uniqueness and then sorted before returning.

        Returns:
            list: A sorted list of unique document names.
        """
        # Initialize the index
        index = self.pc.Index(self.pc_index_name)

        # List all IDs in the namespace
        the_ids = []
        for ids in index.list(namespace=self.namespace):
            the_ids.extend(ids)

        # Split the list of IDs into chunks of 1000 to manage API limitations
        list_of_ids = np.array_split(the_ids, 1000)

        # Initialize a set to collect unique doc_name values
        unique_doc_names = set()

        for chunk_ids in list_of_ids:
            # Fetch data for the current chunk
            response = index.fetch(ids=chunk_ids.tolist(), namespace=self.namespace)

            # Iterate over the fetched items and extract doc_name values
            for item_id, item_data in response['vectors'].items():
                # Check if the item contains metadata and a 'doc_name' field
                if 'metadata' in item_data and 'doc_name' in item_data['metadata']:
                    # Add the doc_name to the set for uniqueness
                    unique_doc_names.add(item_data['metadata']['doc_name'])

        # Convert the set to a sorted list for consistency
        return sorted(unique_doc_names)