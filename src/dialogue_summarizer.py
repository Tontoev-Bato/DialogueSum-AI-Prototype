"""
Dialogue Summarization Module
Main AI component for summarizing dialogues using FLAN-T5
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

class DialogueSummarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self.is_loaded = True
            print(f"✅ Model {self.model_name} loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def summarize(self, dialogue, method="few-shot", max_length=50):
        """
        Summarize dialogue using specified method
        
        Args:
            dialogue (str): Input dialogue text
            method (str): Prompting method - 'zero-shot', 'one-shot', 'few-shot'
            max_length (int): Maximum summary length
            
        Returns:
            str: Generated summary
        """
        if not self.is_loaded:
            self.load_model()
            
        prompt = self._create_prompt(dialogue, method)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def _create_prompt(self, dialogue, method):
        """Create prompt based on the selected method"""
        if method == "zero-shot":
            return f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:"
        
        elif method == "one-shot":
            example = self._get_one_shot_example()
            return f"{example}\n\nDialogue:\n{dialogue}\n\nSummary:"
        
        elif method == "few-shot":
            examples = self._get_few_shot_examples()
            return f"{examples}\n\nDialogue:\n{dialogue}\n\nSummary:"
        
        else:
            return f"Summarize this dialogue: {dialogue}"
    
    def _get_one_shot_example(self):
        """Return one example for one-shot learning"""
        return """Dialogue:
#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close.

Summary: #Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time."""
    
    def _get_few_shot_examples(self):
        """Return multiple examples for few-shot learning"""
        examples = [
            self._get_one_shot_example(),
            """Dialogue:
#Person1#: May, do you mind helping me prepare for the picnic?
#Person2#: Sure. Have you checked the weather report?
#Person1#: Yes. It says it will be sunny all day.

Summary: Mom asks May to help prepare for the picnic and May agrees."""
        ]
        return "\n\n".join(examples)

# Example usage
if __name__ == "__main__":
    summarizer = DialogueSummarizer()
    
    test_dialogue = """
    #Person1#: Have you considered upgrading your system?
    #Person2#: Yes, but I'm not sure what exactly I would need.
    #Person1#: You could consider adding a painting program to your software.
    """
    
    summary = summarizer.summarize(test_dialogue, method="few-shot")
    print(f"Generated Summary: {summary}")