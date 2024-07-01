import json
from typing import List
from openai import OpenAI
import re


def chat_oai(system_prompt, text):
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": system_prompt
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": text
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].message.content

import numpy as np
def get_embeddings_oai(text, model="text-embedding-3-small"):
   client = OpenAI()

   assert type(text) == list
   text = [t.replace('\n', ' ') for t in text]
   vectors = client.embeddings.create(input = text, model=model).data
   vectors = [
       v.embedding for v in vectors
   ]
   return np.array(vectors)

def extract_block(text):
    pattern = r'```(?:json)?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return json.dumps({"entities": []})
    return matches[0].strip()



def evaluate(predicted_labels: List[List[str]], true_labels: List[List[str]]):
    """
    Calculate precision, recall, and F1-score for Named Entity Recognition where
    entities are simple labels like 'AMAZON', 'GOOGLE', etc.

    :param predicted_labels: List of lists, where each sublist contains predicted entity labels for a sentence.
    :param true_labels: List of lists, where each sublist contains true entity labels for a sentence.
    :return: A dictionary containing the precision, recall, and F1-score.
    """
    # Initialize counters for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate over each sentence's predicted and true entities
    for predicted, true in zip(predicted_labels, true_labels):
        # Convert lists to sets for easy comparison
        predicted_set = set(predicted)
        true_set = set(true)

        # Calculate metrics based on sets
        true_positives += len(predicted_set & true_set)
        false_positives += len(predicted_set - true_set)
        false_negatives += len(true_set - predicted_set)

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
