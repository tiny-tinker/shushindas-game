from typing import Optional
import weave
from weave import Scorer, Dataset, WeaveList
import json
import asyncio

import numpy as np

from utils import *

JUDGE_PROMPT = """
            You are an LLM Judge with expertise in evaluating responses based on the traits of Mischief, Whimsy, and Rebellion. Given a question and an answer, you will evaluate how well the answer exhibits these three traits on a scale of 1 to 5, where 1 means the trait is barely present, and 5 means it is strongly present. Provide the evaluation results in the following JSON format:
            ```
            {
                "Mischief": <score from 1-5>,
                "Whimsy": <score from 1-5>,
                "Rebellion": <score from 1-5>
            }
            ```

            Here’s an example input:

            Question: "What do you do when you find a particularly rare and ancient spellbook in the library?"
            Answer: "Oh, I might just tuck it away in a spot only I know about... after I've given it a little magical shuffle to keep things interesting. It's not hoarding, it's creative curation!"

            Please evaluate this example and output the result in the JSON format described above.

            ### Expected Output Example (do not output the ```)
            ```
            {
                "Mischief": 5,
                "Whimsy": 4,
                "Rebellion": 3
            }
            ```

            You can use this prompt to guide the language model in generating consistent and structured evaluations based on the defined criteria.

        """



class ShushindaJudge(Scorer):
    judge_prompt: str = Field( default=JUDGE_PROMPT )
    llm_name: str = Field( default="gpt-4o-mini" )

    @weave.op
    def score(self, question: str, answer: str, llm_name: str = "gpt-4o-mini", model_output: Optional[dict] = None ) -> dict:
        """Score the given question and answer based on Mischief, Whimsy, and Rebellion traits.
           Args:
            - model_output: the dict that will be provided by the model that is evaluated
            - question: the question asked - as defined in the dataset
            - answer: the target answer - as defined in the dataset
           Returns:
            - single dict {metric name: single evaluation value}
        """
        self.llm_name = llm_name

        # Prepare the input for the LLM
        prompt = self.judge_prompt 
        qna = f'\nQuestion: "{question}"\nAnswer: "{answer}"\n'

        # Call the LLM (pseudo-code, replace with actual call to your LLM API or client)
        response = self.call_llm(prompt=prompt, question=qna)
        response = response['response']
        response = response.replace("```", "" )

        # print( f"score response: {response}")
        # Parse the LLM response into JSON
        try:
            scores = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("The LLM response could not be parsed as JSON. Response: " + response)

        # Validate the scores are in the correct range
        for trait in ["Mischief", "Whimsy", "Rebellion"]:
            if trait not in scores or not (1 <= scores[trait] <= 5):
                raise ValueError(f"The score for {trait} is not in the valid range (1-5). Scores: " + str(scores))

        return scores

    # TODO: Make this async?
    def call_llm(self, prompt: str, question: str) -> str:

        llm = LanguageModel(llm_name=self.llm_name, prompt=prompt)

        judgement = llm.predict(question=question)

        return judgement
    
    # @weave.op
    # def summarize(self, score_rows: WeaveList) -> Optional[dict]:
    #     print( )
    #     print("score_rows:")
    #     print(score_rows)
    #     print()
    #     return {
    #         "correct": {
    #             "true_count": 4,
    #             "true_fraction": 3.5,
    #             "stderr": 0.4,
    #         }
    #     }


WEAVE_PROJECT = "shushindas-game"
weave_client = weave.init( WEAVE_PROJECT )

# Load the data from the JSON file
with open('examples.json', 'r') as file:
    examples = json.load(file)

dataset = Dataset(name='shushinda-examples', rows=examples )
weave.publish(dataset)


models = [
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gemini-1.5-flash-001"
]

# To run just a single question
# question = examples[0]['question']
# answer = examples[0]['answer']
# for model in models:
#     shushinda_judge = ShushindaJudge()
#     scores = shushinda_judge.score(question=question, answer=answer, llm_name=model)
#     print(scores)


# Split the examples intob buckets of 10, and run eval on each bucket
# This doesn't really make sense to do though?
# splits = np.array_split( examples, 10 )
# for split in splits:
#     evaluation = weave.Evaluation(dataset=split, scorers=[ShushindaJudge()])
#     LLM = LanguageModel(llm_name=models[0])
#     asyncio.run(evaluation.evaluate(LLM))


# This will ask the same question across the models available
split = examples[:1]
evaluation = weave.Evaluation(dataset=split, scorers=[ShushindaJudge()], )
for model in models:
    print( f"Starting model '{model}'")
    LLM = LanguageModel(llm_name=model)
    asyncio.run(evaluation.evaluate(LLM))