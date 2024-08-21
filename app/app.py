import datetime
import random
from flask import Flask, jsonify, render_template, request, session
import os

from utils import *


# import app.shushinda_prompt as shushinda_prompt


app = Flask(__name__)
app.secret_key = 'So long and thanks for all the fish!'


WEAVE_PROJECT = "sushindas-game"
weave.init( WEAVE_PROJECT )


# VECTOR_DB = "bq"
VECTOR_DB = "pinecone"

LLM = "openai"

# EMB_MODEL_FAM = "gecko"
# EMB_MODEL_FAM = "openai"


history = []

@app.route("/clear-history", methods=['POST'])
def clear_hist():
    print( 'clear!' )
    history = []
    session['history'] = history
    return jsonify(history=session.get('history',[]))

@app.route("/ask", methods=['POST'])
def ask():
    data = request.json
    print( f"data: {data}")
    question = data.get( "question" )
    print( f"question: {question}")
    model = data.get("model")
    sample_question_num = data.get("q_num")

    emb_stuff = EmbeddingsDB()
    
    # # Get the data to answer the question that 
    # # most likely matches the question based on the embeddings
    context = emb_stuff.search_vector_database(question)
    llm = LanguageModel(llm_name=model)
    history = session['history']

    if question is not None:
        with weave.attributes({'sample_question_num': sample_question_num}):
            answer = llm.predict(question, context)
        if answer is not None:
            history.extend( add_to_history( question, answer ) )
            answer = random.choice(GREETINGS)
    session['history'] = history 
    sample_questions = get_sample_questions()

    return jsonify(answer=answer, history=history, sample_questions=sample_questions)


# The Home page route
@app.route("/", methods=['POST', 'GET'])
def main():

    answer = None
    question = None
    response = None
    history = []

    print( f"session history: {str( session.get('history') )[:40]}...")
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

    # print( f"history: {history}")
    session['history'] = history

    # Display the home page with the required variables set
    llms = [ l['model'] for l in LLMS ]
    model = { "message": answer, "input": question, "history": history, "working": "done", "llms":llms }
    
    qs = get_sample_questions()
    model = model | qs

    return render_template('index.html', model=model)

def get_sample_questions():
    
    # Get a slice of the questions
    some_questions = random.sample(ALL_SAMPLE_QUESTIONS, 4 )

    # Sometimes add in this one, otherwise leave it
    some_questions[0] = UNLOCK_QUESTION #if random.random() > 0.5 else some_questions[0]

    random.shuffle(some_questions)

    model = {}
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



if __name__ == '__main__':
    if os.getenv("PINECONE_KEY") is None:
        print( "PINECONE_KEY not set!" )
    if os.getenv( "PROJECT" ) is None:
        print( "PROJECT not set!" )
    if os.getenv( "REGION" ) is None:
        print( "REGION not set!" )
    if os.getenv("OPENAI_API_KEY") is None:
        print( "OPENAI_API_KEY not set!" )
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
