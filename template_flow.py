import os
from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    step,
    environment,
    kubernetes,
    pypi,
    retry,
)
from metaflow.cards import Markdown

GCS_PROJECT_NAME = "project-name-here"
GCS_BUCKET_NAME = "bucket-name-here"

class TemplateFlow(FlowSpec):
    """
    This flow uses a Hugging Face model to generate text messages.
    """

    example_config = IncludeFile("example_config", default="./example_config.json")
    prompt = Parameter("prompt", help="Prompt for the model to generate a message", type=str, default="Hello!")

    offline_wandb = Parameter(
        "offline",
        help="Do not connect to W&B servers when training",
        type=bool,
        default=True,
    )

    @card(type="default")
    @step
    def start(self):
        """
        Start step for setup tasks.
        """
        self.next(self.generate_message)

    
    @card
    @retry(times=4)
    @pypi(python='3.10.8',
        packages={
            'torch': '2.5.1',
            'wandb': '0.17.9',
            'transformers': '4.46.1',
            'mozmlops': '0.1.4',
            'sentencepiece': '0.2.0',
            'protobuf': '4.25.5',
            'causal-conv1d': '1.4.0',
        })
    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "OTEL_TRACES_EXPORTER": "none",
        }
    )
    @kubernetes(cpu=1)
    @step
    def generate_message(self):
        """
        Generate a message using a Hugging Face model.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "mistralai/Mamba-Codestral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))


        inputs = tokenizer(self.prompt, return_tensors="pt")
        
        outputs = model.generate(**inputs, max_length=50)
        generated_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Generated Message: {generated_message}")

        self.generated_message = generated_message
        self.next(self.end)

    @step
    def end(self):
        """
        End step for final output.
        """
        print("Flow complete.")
        print(f"Generated Message: {self.generated_message}")

if __name__ == "__main__":
    TemplateFlow()
