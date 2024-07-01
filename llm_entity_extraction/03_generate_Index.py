
import helper
import json
import faiss



def main():
    train_data = json.load(open("train_data.json", "r"))
    dim = helper.get_embeddings_oai(["test"]).shape[1]
    print(dim)
    index = faiss.IndexFlatL2(dim)

    for i in range(0, len(train_data['text']), 100):
        text = train_data['text'][i:i+100]
        train_vectors = helper.get_embeddings_oai(text)
        index.add(train_vectors)

    # save the index
    faiss.write_index(index, "training_data_vectors.faiss")


if __name__ == "__main__":
    main()