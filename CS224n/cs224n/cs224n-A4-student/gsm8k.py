import re
import json
from typing import Dict
import os

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

INVALID_ANS = "[invalid]"

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in 
# ------------------------------------------- #



def eval_model_on_gsm8k() -> None:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.
    
    See example_usage.py for how to query models and handle responses.
    The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer' fields.
    
    Think about: What metric will you use to evaluate performance? How will you 
    handle cases where the model's output cannot be parsed?
    """
    

    examples = []
    with open("data/gsm8k_first_100.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[: 3]

    for model_id in ["A", "B"]:
        correct = 0
        wrong = 0
        invalid = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        for example in tqdm(examples,desc=f"Evaluating model {model_id}"):
            question = example["question"]
            gold_answer = str(example["numerical_answer"])

            prompt = standard_prompt_template(question)
            query = Query(turns=[{"user":prompt}])
            response = query_model(
                model_id = model_id,
                query = query,
            )
            prediction = standard_output_extractor(response.text)

            if prediction == INVALID_ANS:
                invalid+=1
            elif prediction == gold_answer:
                correct+=1
            else: 
                wrong+=1
            total_cost += response.cost
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
        total = len(examples)
        accuracy = correct / total

        print(f"Model {model_id}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {correct}")
        print(f"Wrong: {wrong}")
        print(f"Invalid: {invalid}")
        print(f"Input tokens: {total_input_tokens}")
        print(f"Output tokens: {total_output_tokens}")
        print(f"Cost: ${total_cost:.8f}")        



def superior_prompt_template(question: str) -> str:
    """
    Design your own prompt template that outperforms standard_prompt_template on model A.
    
    Args:
        question: gsm8k question.
    Returns:
        Your improved prompt for the model.
    
    Look at standard_prompt_template() to understand the baseline approach. What 
    aspects of how you prompt the model might affect its reasoning or accuracy?
    
    NOTE: Your prompt must still produce output in the "#### <answer>" format
    so that standard_output_extractor() can parse the response.
    """
    
    prompt = f"""
Solve the following math word problem carefully.

Instructions:
1. Identify the known quantities and what the problem asks for.
2. Determine the required operations and calculate step by step.
3. Check that the final result is consistent with the question.
4. End your response with exactly one line in this format:
   #### <numerical_answer>

Problem:
{question}
""".strip()

    return prompt
    

def eval_model_on_gsm8k_with_improved_prompt() -> None:
    examples = []

    with open("data/gsm8k_first_100.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line))

    examples = examples[:3]

    correct = 0
    wrong = 0
    invalid = 0

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    model_id = "A"

    for example in tqdm(
        examples,
        desc="Evaluating model A with improved prompt",
    ):
        question = example["question"]
        gold_answer = str(example["numerical_answer"])

        
        prompt = superior_prompt_template(question)

        
        query = Query(turns=[
            {"user": prompt}
        ])

        
        response = query_model(
            model_id=
            model_id,
            query=query,
        )

       
        prediction = standard_output_extractor(
            response.text
        )

        
        if prediction == INVALID_ANS:
            invalid+=1
        elif prediction == gold_answer:
            correct+=1
        else:
            wrong+=1

        
        total_cost += response.cost
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

    
    assert correct + wrong + invalid == len(examples)

    
    total = len(examples)
    accuracy = correct / total

    print("Model A with improved prompt")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Invalid: {invalid}")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Cost: ${total_cost:.8f}")

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    #eval_model_on_gsm8k()
    #eval_model_on_gsm8k_with_improved_prompt()