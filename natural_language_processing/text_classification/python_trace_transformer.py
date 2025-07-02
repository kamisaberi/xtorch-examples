# python_trace_transformer.py
import torch
from transformers import AutoModel, AutoTokenizer # Generic way to load models

# model_name = 'bert-base-uncased'
model_name = 'distilbert-base-uncased' # Smaller, faster alternative

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval() # Set to evaluation mode for tracing

    # Create dummy inputs
    dummy_text = "This is a dummy sentence for tracing."
    # Tokenizer output includes input_ids, attention_mask.
    # DistilBERT typically doesn't use token_type_ids for single sentences.
    inputs = tokenizer(dummy_text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # The base AutoModel typically returns an object (e.g., BaseModelOutput) or a tuple.
    # We often want the `last_hidden_state` and potentially `pooler_output` (if available, BERT has it, DistilBERT doesn't have a separate pooler by default).
    # For classification with DistilBERT, we usually take the hidden state of the [CLS] token from last_hidden_state.

    # Wrapper to ensure consistent output (e.g., just last_hidden_state)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state # Shape: [batch_size, seq_len, hidden_dim]

    wrapped_model = ModelWrapper(base_model)
    traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))

    traced_model_path = "transformer_traced.pt"
    traced_model.save(traced_model_path)
    print(f"Transformer model traced and saved to {traced_model_path}")

    # Save tokenizer vocabulary (this will create a directory, e.g., 'tokenizer_vocab/vocab.txt')
    tokenizer.save_vocabulary("./transformer_vocab")
    print("Tokenizer vocabulary saved to ./transformer_vocab/vocab.txt")

    # Verify traced model output
    # python_output = traced_model(input_ids, attention_mask)
    # print(f"Traced model output shape (last_hidden_state): {python_output.shape}")

except Exception as e:
    print(f"Error during tracing or saving: {e}")
    print("Check model compatibility with tracing and Hugging Face documentation for specific model outputs.")