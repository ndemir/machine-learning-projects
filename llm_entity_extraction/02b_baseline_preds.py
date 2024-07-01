
from tqdm import tqdm
import helper
import json
import pickle

list_of_orgs = open('list_of_orgs.txt', 'r').read()
test_data = json.load(open("test_data.json", "r"))


baseline_predicted_orgs = pickle.load(open("baseline_predicted_orgs.pkl", "rb"))
metrics = helper.evaluate(baseline_predicted_orgs, test_data['orgs'])

print(metrics)