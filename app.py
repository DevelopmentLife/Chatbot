from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Base path where checkpoints are stored
base_path = "C:\\Users\\Zacha\\Desktop\\zack\\Final_MSAAI"

# List of checkpoints
checkpoints = [
    "checkpoint-500", "checkpoint-1000", "checkpoint-1500",
    "checkpoint-2000", "checkpoint-2500", "checkpoint-3000",
    "checkpoint-3500", "checkpoint-4000", "checkpoint-4500",
    "checkpoint-5000", "checkpoint-5500", "checkpoint-6000",
    "checkpoint-6500", "checkpoint-7000", "checkpoint-7500",
    "checkpoint-8000", "checkpoint-8500", "checkpoint-9000",
    "checkpoint-9500", "checkpoint-10000", "checkpoint-10500"
]

models = {}
tokenizers = {}

for checkpoint in checkpoints:
    try:
        checkpoint_path = os.path.join(base_path, checkpoint)
        models[checkpoint] = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        tokenizers[checkpoint] = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Could not load model or tokenizer from checkpoint {checkpoint}: {e}")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    example_checkpoint = "checkpoint-500"  # Replace with any loaded checkpoint
    if example_checkpoint not in models or example_checkpoint not in tokenizers:
        return jsonify({'response': 'Model or tokenizer for this checkpoint could not be loaded.'})
        
    input_ids = tokenizers[example_checkpoint].encode(user_input, return_tensors='pt')
    with torch.no_grad():
        output = models[example_checkpoint].generate(input_ids)
    generated_response = tokenizers[example_checkpoint].decode(output[0], skip_special_tokens=True)
    
    return jsonify({'response': generated_response})

if __name__ == '__main__':
    app.run(port=5000)