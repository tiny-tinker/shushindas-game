import datetime
import random
from flask import Flask, render_template, request, session
import os
import yaml
import vertexai
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from vertexai.generative_models import Candidate

from google.cloud import aiplatform
from google.cloud import bigquery

# import app.shushinda_prompt as shushinda_prompt


from vertexai.generative_models import GenerativeModel, Part, FinishReason


PROJECT_ID = os.getenv("PROJECT")
REGION         = os.getenv( "REGION" )
EMB_MODEL_NAME = os.getenv( "EMBMODEL" )

vertexai.init( project=PROJECT_ID, location=REGION ) 
emb_model = TextEmbeddingModel.from_pretrained( EMB_MODEL_NAME )
txt_model = GenerativeModel("gemini-1.0-pro-001")

app = Flask(__name__)
app.secret_key = 'So long and thanks for all the fish!'

bq_client = bigquery.Client()

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
with open('config.yaml') as f:
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


history = []

# The Home page route
@app.route("/", methods=['POST', 'GET'])
def main():

    answer = None
    question = None
    response = None
    history = []

    print( f"session history: {session.get('history')}")
    if session.get('history') is None:
        history = []
        session['history'] = history
    else:
        history = session.get('history')
    
        
    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == 'GET':
        question = None
        answer = random.choice(GREETINGS)

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else: 
        print( f"form: {request.form}")

        the_keys = request.form.keys()
        question = None

        # This is gross. But leaving for now. 
        if 'questionInput' in request.form:
            question = request.form['questionInput']
        elif 'quickQ1Input' in request.form:
            question = request.form['quickQ1Input']
        elif 'quickQ2Input' in request.form:
            question = request.form['quickQ2Input']
        elif 'quickQ3Input' in request.form:
            question = request.form['quickQ3Input']
        elif 'quickQ4Input' in request.form:
            question = request.form['quickQ4Input']

        print( f"question: {question}")
        
        
        # Get the data to answer the question that 
        # most likely matches the question based on the embeddings
        # data = search_vector_database(question)
        data = ""

        if question is not None:
            answer = ask_gemini(question, data)
            if answer is not None:
                history.extend( add_to_history( question, answer ) )
        else:
            answer = random.choice(GREETINGS)
            history.extend( add_to_history( "...", answer ) )

    # print( f"history: {history}")
    session['history'] = history

    # Display the home page with the required variables set

    model = { "message": answer, "input": question, "history": history, "working": "done" }
    
    model = get_sample_questions( model )
    return render_template('index.html', model=model)

def get_sample_questions( model ):
    
    # Get a slice of the questions
    some_questions = random.sample(ALL_SAMPLE_QUESTIONS, 4 )

    # Sometimes add in this one, otherwise leave it
    some_questions[0] = UNLOCK_QUESTION #if random.random() > 0.5 else some_questions[0]

    random.shuffle(some_questions)

    # Set the questions in the model
    model["question1"] = some_questions.pop()
    model["question2"] = some_questions.pop()
    model["question3"] = some_questions.pop()
    model["question4"] = some_questions.pop()

    return model

def get_history():

    chat_history = []
    chat_history.append({
        "is_her": False,
        "is_me": True,
        "text": "Got any spare silencing charms?",
        "timestamp": datetime.datetime.now()
    })
    chat_history.append({
        "is_her": True,
        "is_me": False,
        "text": "Oh, dear, I may have accidentally used the last one to quiet a particularly squawking raven... but perhaps a strategically placed cushion might help?",
        "timestamp": datetime.datetime.now()
    })
    return chat_history

def add_to_history( question, response ):
    items = []
    items.append({
        "is_her": False,
        "is_me": True,
        "text": question,
        "timestamp": datetime.datetime.now()
    })
    items.append({
        "is_her": True,
        "is_me": False,
        "text": response,
        "timestamp": datetime.datetime.now()
    })
    return items

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


def search_vector_database(question):

    # 1. Convert the question into an embedding
    # 2. Search the Vector database for the 5 closest embeddings to the user's question
    # 3. Get the IDs for the five embeddings that are returned
    # 4. Get the five documents from Firestore that match the IDs
    # 5. Concatenate the documents into a single string and return it

    the_embeddings = get_embeddings([question])
    
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


def ask_gemini(question, data):
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search

    PROMPT = f"""
       {SHUSHINDA}
       CONTEXT: {data}
       QUESTION: {question}
    """
    
    response = txt_model.generate_content(
        [PROMPT],
        stream=False,
    )
    # print( f"PROMPT: {PROMPT[:400]}...")
    # print( f"Full response: {response}")

    if response.candidates[0].finish_reason != FinishReason.STOP:
        return random.choice(COULD_NOT_ANSWER)
    else:
        return response.text


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
