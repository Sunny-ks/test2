from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
import torch

# Load your fine-tuned Phi-4 model and tokenizer
model_name = "path_to_your_fine_tuned_phi_4_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Initialize the GPTQ quantizer
quantizer = GPTQQuantizer(
    bits=4,  # Number of bits for quantization
    dataset="your_calibration_dataset",  # Dataset for calibration
    model_seqlen=2048  # Sequence length used during model training
)

# Apply quantization
quantized_model = quantizer.quantize_model(model, tokenizer)

# Save the quantized model
save_folder = "/path/to/save_quantized_model"
quantizer.save(quantized_model, save_folder)
