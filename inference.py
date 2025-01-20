from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the Mistral v0.3 instruct model and tokenizer
model_name = "mistral/v03-instruct"  # Replace with the correct model path on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")  # Move model to GPU

# Set up the text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Ensure GPU is used (device=0 for first GPU)
)

# Define the prompt
prompt = "Explain the importance of teamwork in online multiplayer games."

# Generate a response
output = generator(
    prompt,
    max_length=150,  # Max tokens; allows room for a 100-word response
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7  # Adjust for more randomness
)

# Extract the response without the prompt
response = output[0]["generated_text"].replace(prompt, "").strip()

# Post-process to limit the response to 100 words
response_words = response.split()
if len(response_words) > 100:
    response = " ".join(response_words[:100])

print("Generated Response:")
print(response)
