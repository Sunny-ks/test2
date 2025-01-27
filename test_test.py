from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(model_name, system_prompt, user_query, max_new_tokens=100, device='cpu'):
    """
    Generate a response using the Mistral model with a system and user prompt.

    Args:
        model_name (str): The name of the Mistral model on Hugging Face.
        system_prompt (str): The system prompt for the conversation.
        user_query (str): The user's query.
        max_new_tokens (int): Maximum number of tokens to generate.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        str: The model's response without the prompt part.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to the specified device
    model = model.to(device)

    # Construct the full prompt
    full_prompt = f"{system_prompt}\nUser: {user_query}\nAssistant:"

    # Tokenize the input prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7  # Adjust sampling temperature as needed
    )

    # Decode the full generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    response_start = len(full_prompt)  # Calculate the start of the response
    response = generated_text[response_start:].strip()

    return response

# Example usage
if __name__ == "__main__":
    model_name = "mistral-7b"  # Replace with the actual model path or name
    system_prompt = "You are a helpful assistant."
    user_query = "What is the capital of France?"

    response = generate_response(model_name, system_prompt, user_query, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Response:", response)
