from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch

# Define the model name using a pre-trained model from the Sentence Transformers library
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load the tokenizer for the specified model from Hugging Face's transformers library
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model for masked language modeling based on the specified model
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Determine if a GPU is available and set the device accordingly; use CPU if GPU is not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the appropriate device (GPU or CPU)
model.to(device)

# Define a generator function to create a dataset; this should be replaced with actual data loading logic
def dataset_generator():
    # Example dataset composed of individual sentences; replace with your actual dataset sentences
    dataset = ["sentence1", "sentence2", "sentence3"]
    # Yield each sentence as a dictionary with the key 'text'
    for sentence in dataset:
        yield {"text": sentence}

# Create a dataset object using Hugging Face's Dataset class from the generator function
dataset = Dataset.from_generator(dataset_generator)

# Define a function to tokenize the text data
def tokenize_function(example):
    # Tokenize the input text and truncate it to the maximum length the model can handle
    return tokenizer(example["text"], truncation=True)

# Apply the tokenization function to all items in the dataset, batch processing them for efficiency
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Initialize a data collator for masked language modeling which randomly masks tokens
# This is used for training the model in a self-supervised manner
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Define training arguments to configure the training session
training_args = TrainingArguments(
    output_dir="output",  # Directory where the outputs (like checkpoints) will be saved
    num_train_epochs=3,  # Total number of training epochs to perform
    per_device_train_batch_size=16,  # Batch size per device during training
    learning_rate=2e-5,  # Learning rate for the optimizer
)

# Initialize the Trainer, which handles the training loop and evaluation
trainer = Trainer(
    model=model,  # The model to be trained, already loaded and configured
    args=training_args,  # The training arguments defining the training setup
    train_dataset=tokenized_datasets,  # The dataset to train on, already tokenized and prepared
    data_collator=data_collator,  # The data collator that handles input formatting and masking
)

# Start the training process
trainer.train()

# Define the paths where the fine-tuned model and tokenizer will be saved
model_path = "./model"
tokenizer_path = "./tokenizer"

# Save the fine-tuned model to the specified path
model.save_pretrained(model_path)

# Save the tokenizer used in training to the specified path
tokenizer.save_pretrained(tokenizer_path)

# Load the tokenizer and model from saved paths, ensuring the model is allocated to the appropriate device (GPU or CPU)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)

# Define a function to tokenize input sentences, configuring padding and truncation to handle variable sentence lengths
def tokenize_function_embedding(example):
    return tokenizer(example["text"], padding=True, truncation=True)

# List of example sentences to generate embeddings for
sentences = ["This is the first sentence.", "This is the second sentence."]

# Create a Dataset object directly from these sentences
dataset_embedding = Dataset.from_dict({"text": sentences})

# Apply the tokenization function to the dataset, preparing it for embedding generation
tokenized_dataset_embedding = dataset_embedding.map(tokenize_function_embedding, batched=True, batch_size=None)

# Extract 'input_ids' and 'attention_mask' needed for the model to understand which parts of the input are padding and which are actual content
input_ids = tokenized_dataset_embedding["input_ids"]
attention_mask = tokenized_dataset_embedding["attention_mask"]

# Convert these lists into tensors and ensure they are on the correct device (GPU or CPU) for processing
input_ids = torch.tensor(input_ids).to(device)
attention_mask = torch.tensor(attention_mask).to(device)

# Generate embeddings using the model without updating gradients to save computational resources
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # Extract the last layer's hidden states as embeddings, specifically the first token (typically used in BERT-type models for representing sentence embeddings)
    embeddings = outputs.hidden_states[-1][:, 0, :]

# Move the embeddings from the GPU back to CPU for easy manipulation or saving
embeddings = embeddings.cpu()

# Print each sentence with its corresponding embedding vector
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}\n")
