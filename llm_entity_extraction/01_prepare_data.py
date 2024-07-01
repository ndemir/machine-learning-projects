from datasets import load_dataset
import json
# Load the dataset
dataset = load_dataset("conll2003")



def convert_to_text_entities(tokens, ner_tags):
    text = " ".join(tokens)
    entities = []
    current_entity = None
    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": token, "type": tag[2:], "start": text.index(token)}
        elif tag.startswith("I-"):
            if current_entity and current_entity["type"] == tag[2:]:
                current_entity["text"] += " " + token
        else:  # "O" tag
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return {"text": text, "entities": entities}

def extract_organizations(examples):
    texts = []
    orgs = []
    for example in examples:
        tokens = example['tokens']
        ner_tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        ner_tags_str = [ner_tag_names[tag] for tag in example['ner_tags']]
        texts.append(" ".join(tokens))
        entities = convert_to_text_entities(tokens, ner_tags_str)
        orgs.append([entity['text'] for entity in entities['entities'] if entity['type'] == 'ORG'])
    return texts, orgs

# Extract organizations from the dataset
train_text, train_orgs = extract_organizations(dataset['train'])
test_text, test_orgs = extract_organizations(dataset['test'])
print("Sample Train:")
print(train_text[0])
print(train_orgs[0])

print("Sample Test:")
print(test_text[0])
print(test_orgs[0])

filtered_test_text = []
filtered_test_orgs = []
for a, b in zip(test_text, test_orgs):
  if len(b)>0:
    filtered_test_text.append(a)
    filtered_test_orgs.append(b)

train_orgs_flat_list = set([item for sublist in train_orgs for item in sublist])
train_orgs_flat_list_str = ",".join(train_orgs_flat_list)

open('list_of_orgs.txt', 'w').write(train_orgs_flat_list_str)
test_data = {
    "text": filtered_test_text,
    "orgs": filtered_test_orgs
}
train_data = {
    "text": train_text,
    "orgs": train_orgs
}
json.dump(train_data, open("train_data.json", "w"))
json.dump(test_data, open("test_data.json", "w"))

