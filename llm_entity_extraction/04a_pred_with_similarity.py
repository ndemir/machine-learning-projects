
from tqdm import tqdm
import helper
import json
import multiprocessing as mp
import pickle
import faiss


list_of_orgs = open('list_of_orgs.txt', 'r').read()
list_of_orgs = [org.strip() for org in list_of_orgs.split(",")]
def extract_organizations(text, examples):

    examples = "\n".join([f"Text: {example[0]}\nExtracted Organizations: {example[1]}" for example in examples])

    system_prompt = f"""
    You are tasked with extracting named entities from a given text. Named entities will be organizations.
    Possible entities are:
    {list_of_orgs}
    The output should be in the following JSON format, between 3 backticks. Extract the entities if they are in the given list.
    ```
    {{"entities": ["ENTITY_A", "ENTITY_B"]}}
    ```

    Consider the following examples:
    {examples}
    """   


    response = helper.chat_oai(system_prompt, text)
    # print(response)
    response = helper.extract_block(response)
    # print(response)
    entities = json.loads(response)['entities']
    entities = set(list_of_orgs).intersection(entities)
    return list(set(entities))



def main():

    test_data = json.load(open("test_data.json", "r"))
    train_data = json.load(open("train_data.json", "r"))

    # load faiss
    index = faiss.read_index("training_data_vectors.faiss")
 
    predicted_orgs_with_similarity = []
    for text in tqdm(test_data['text']):
        test_vector = helper.get_embeddings_oai([text])
        D, I = index.search(test_vector, 10)
        examples = []
        for i in I[0]:
            examples.append([
                    train_data['text'][i], train_data['orgs'][i]
            ])
        orgs = extract_organizations(text, examples)
        predicted_orgs_with_similarity.append(orgs)
        
    
    pickle.dump(predicted_orgs_with_similarity, open("predicted_orgs_with_similarity.pkl", "wb"))

if __name__ == "__main__":
    main()