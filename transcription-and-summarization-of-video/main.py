from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import os
import random

def yt2mp3(url, outputMp3F):
    tmpVideoF=random.random()
    os.system(f"./bin/youtube-dl -o /tmp/{tmpVideoF} --verbose " + url)
    os.system(f"ffmpeg -y -i /tmp/{tmpVideoF}.* -vn -ar 44100 -ac 2 -b:a 192k {outputMp3F}")


def speech2text(mp3_file):
    # Set the computation device to GPU (if available) or CPU
    device = 'cuda:0'

    # Choose data type based on CUDA availability (float16 for GPU, float32 for CPU)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model identifier for the speech-to-text model
    model_id = "distil-whisper/distil-large-v2"

    # Load the model with specified configurations for efficient processing
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        use_flash_attention_2=True
    )

    # Move the model to the specified device (GPU/CPU)
    model.to(device)

    # Load the processor for the model (handling tokenization and feature extraction)
    processor = AutoProcessor.from_pretrained(model_id)

    # Set up a speech recognition pipeline with the model and processor
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Process the MP3 file through the pipeline to get the speech recognition result
    result = pipe(mp3_file)

    # Extract the text from the recognition result
    text_from_video = result["text"]

    # Return the extracted text
    return text_from_video


def chat(system_prompt, text):
    """
    It is not a good practice to load the model again and again,
    but for the sake of simlicity for demo, let's keep as it is
    """

    # Define the model name to be used for the chat function
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    # Authentication token for Hugging Face API
    token = os.environ['HUGGINGFACE_TOKEN']

    # Configure the model to load in a quantized 8-bit format for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # Set the device map to load the model on GPU 0
    device_map = {"": 0}
    # Load the model from Hugging Face with the specified configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=token
    )

    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # Create a text-generation pipeline with the loaded model and tokenizer
    llama_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Format the input text with special tokens for the model
    text = f"""
    <s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>
    {text}[/INST]
    """

    # Generate sequences using the pipeline with specified parameters
    sequences = llama_pipeline(
        text,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=32000
    )

    # Extract the generated text from the sequences
    generated_text = sequences[0]["generated_text"]
    # Trim the generated text to remove the instruction part
    generated_text = generated_text[generated_text.find('[/INST]')+len('[/INST]'):]

    # Return the processed generated text
    return generated_text

def summarize(text):
    # Define the maximum input length for each iteration of summarization
    input_len = 10000

    # Start an infinite loop to repeatedly summarize the text
    while True:
        # Print the current length of the text
        print(len(text))
        # Call the chat function to summarize the text. Only the first 'input_len' characters are considered for summarization
        summary = chat("", "Summarize the following: " + text[0:input_len])

        if len(text) < input_len:
            return summary
        
        # Concatenate the current summary with the remaining part of the text for the next iteration
        text = summary + " " + text[input_len:]

outputMp3F="./files/audio.mp3"
yt2mp3(url="https://www.youtube.com/watch?v=SEkGLj0bwAU", outputMp3F=outputMp3F)
transcribed=speech2text(mp3_file=outputMp3F)
print(summarize(transcribed))