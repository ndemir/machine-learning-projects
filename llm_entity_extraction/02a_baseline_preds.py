
from tqdm import tqdm
import helper
import json

list_of_orgs = open('list_of_orgs.txt', 'r').read()
list_of_orgs = [org.strip() for org in list_of_orgs.split(",")]

def extract_organizations(text):
    system_prompt = f"""
    You are tasked with extracting named entities from a given text. Named entities will be organizations.
    Possible entities are:
    {list_of_orgs}
    The output should be in the following JSON format, between 3 backticks. Extract the entities if they are in the given list.
    ```
    {{"entities": ["ENTITY_A", "ENTITY_B"]}}
    ```
    """   

    try:
        response = helper.chat_oai(system_prompt, text)
        # print(response)
        response = helper.extract_block(response)
        # print(response)
        entities = json.loads(response)['entities']
        entities = set(list_of_orgs).intersection(entities)
        return list(set(entities))
    except Exception as e:
        import traceback
        traceback.print_exc()
        # print("Error in text: ", text)
        return extract_organizations(text)


def main():

    test_data = json.load(open("test_data.json", "r"))
 
    from math import sqrt

    from joblib import Parallel, delayed


    baseline_predicted_orgs= []
    for text in tqdm(test_data['text']):
        organizations = extract_organizations(text)
        print(organizations)
        baseline_predicted_orgs.append(organizations)
    

    print(baseline_predicted_orgs)
    import pickle
    pickle.dump(baseline_predicted_orgs, open("baseline_predicted_orgs.pkl", "wb"))

if __name__ == "__main__":
    main()