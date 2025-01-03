import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

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
            """
            
        else:
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

        generation_config = GenerationConfig(
        min_new_tokens=100,
        max_new_tokens=2000,
        repetition_penalty=1.2,
        return_full_text = False)

        pipeline = transformers.pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="cuda",
            generation_config=generation_config)

        return pipeline

# Example usage
if __name__ == "__main__":
    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    loader = ModelLoader(model_name)
    pipeline = loader.load_pipeline()

    # Use the pipeline
    output = pipeline("Que peux-tu dire sur le RAG ?")[0]["generated_text"]
    print(output)
