# Importing necessary libraries and modules.
import os
from transformers import (
    pipeline,  # For creating a pipeline for text generation tasks.
    AutoModelForCausalLM,  # Auto class for causal language models.
    AutoTokenizer,  # Auto class for tokenizer.
    BitsAndBytesConfig,  # Configuration class for model quantization.
)
import json  # For handling JSON data.

def chat(text):
    """
    Function to generate text using a pre-trained model.
    
    Args:
        text (str): The input text for the model to process.
        
    Returns:
        str: The generated text output.
    
    Note:
        This function is not optimized for repeated model loading.
        It is simplified for demonstration purposes.
    """
    # Define the model name to be used for the chat function.
    model_name = "Trelis/Llama-2-7b-chat-hf-function-calling-v2"

    # Retrieve the authentication token for Hugging Face API from environment variables.
    token = os.environ['HUGGINGFACE_TOKEN']

    # Configure the model to load in a quantized 8-bit format for improved efficiency.
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Define the device map to specify which GPU to use for model loading.
    device_map = {"": 0}

    # Load the model from Hugging Face with the specified configuration.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=token
    )

    # Load the tokenizer associated with the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # Create a text-generation pipeline with the loaded model and tokenizer.
    llama_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Generate sequences using the pipeline with specified parameters.
    sequences = llama_pipeline(
        text,
        do_sample=True,  # Enable sampling for diverse generation.
        top_k=10,  # Top-k sampling for sequence generation.
        num_return_sequences=1,  # Number of sequences to generate.
        eos_token_id=tokenizer.eos_token_id,  # End-of-sequence token ID.
        max_length=32000  # Maximum length of the generated sequence.
    )

    # Extract the generated text from the sequences.
    generated_text = sequences[0]["generated_text"]
    
    # Trim the generated text to remove the instruction part.
    generated_text = generated_text[generated_text.find('[/INST]')+len('[/INST]'):]

    # Return the processed generated text.
    return generated_text

# Metadata about the 'add' function.
function_metadata = {
    "function": "add",
    "description": "This function adds n numbers and returns the sum. Use this function if customer asks for the sum of numbers.",
    "arguments": [
        {
            "name": "numbers",
            "type": "string",
            "description": "The numbers to be added in a comma-separated string.",
        }
    ]
}

def add(numbers):
    """
    Function to add a list of numbers.

    Args:
        numbers (str): Comma-separated string of numbers to be added.
    
    Returns:
        int: Sum of the numbers.
    """
    # Split the input string into a list of numbers and convert them to integers.
    numbers = [int(n) for n in numbers.split(',')]
    # Return the sum of the numbers.
    return sum(numbers)

# Template strings for formatting the prompt.
B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
B_INST, E_INST = "[INST] ", " [/INST]"

# Convert function metadata to JSON and format it.
function_list = json.dumps(function_metadata, indent=4)

# Example question for the model.
question = "What is the sum of numbers between 187592 and 558429?"

# Construct the prompt for the model.
prompt = f"{B_FUNC}{function_list.strip()}{E_FUNC}{B_INST}{question.strip()}{E_INST}\n\n"

# Generate a response using the chat function.
data = chat(prompt)
# Parse the response as JSON.
data = json.loads(data)

# Check if the function to be called is 'add'.
if data["function"] == "add":
    # Call the add function with the provided arguments and print the result.
    result = add(data["arguments"]["numbers"])
    print(result)
