
from tqdm import tqdm
import helper
import json
import pickle

list_of_orgs = open('list_of_orgs.txt', 'r').read()
test_data = json.load(open("test_data.json", "r"))


predicted_orgs_with_similarity = pickle.load(open("predicted_orgs_with_similarity.pkl", "rb"))
metrics = helper.evaluate(predicted_orgs_with_similarity, test_data['orgs'])

print(metrics)