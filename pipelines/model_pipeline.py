import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    def __init__(self, model_name, save_dir="models"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.model_path = os.path.join(save_dir, model_name)

    def load_pipeline(self):
        # Check if the model is already downloaded
        if os.path.exists(self.model_path):
            """
            print(f"Loading model from local path: {self.model_path}")
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            """
            print(f"Downloading model from HuggingFace: {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="cuda")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            """
            # Save the model locally
            os.makedirs(self.model_path, exist_ok=True)
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)
            print(f"Model saved to: {self.model_path}")
            """

        # Create the pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        return pipeline

# Example usage
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    loader = ModelLoader(model_name)
    pipeline = loader.load_pipeline()

    # Use the pipeline
    output = pipeline("Once upon a time,")[0]["generated_text"]
    print(output)
