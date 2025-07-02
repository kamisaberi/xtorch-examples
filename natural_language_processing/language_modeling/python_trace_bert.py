# python_trace_bert.py
import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.eval() # Set to evaluation mode for tracing

# Create dummy inputs
dummy_text = "This is a dummy sentence."
inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
# token_type_ids = inputs['token_type_ids'] # For sentence pairs, not strictly needed for single sentences for base BERT outputs

# Trace the BERT model (only the base BERT part, not the classification head yet)
# Input to base BertModel is input_ids and attention_mask (and token_type_ids if needed)
# It typically outputs (last_hidden_state, pooler_output)
try:
    # If your BertModel returns a tuple:
    traced_model = torch.jit.trace(bert_model, (input_ids, attention_mask))
    # If your BertModel is configured to return a BaseModelOutputWithPoolingAndCrossAttentions object:
    # You might need to wrap it or select specific outputs for tracing.
    # For simplicity, assume it returns a tuple or you adapt the model.
    # Example for specific output selection if it's an object:
    # class BertWrapper(torch.nn.Module):
    #     def __init__(self, bert):
    #         super().__init__()
    #         self.bert = bert
    #     def forward(self, input_ids, attention_mask):
    #         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #         return outputs.last_hidden_state, outputs.pooler_output # Or just pooler_output
    # wrapped_bert = BertWrapper(bert_model)
    # traced_model = torch.jit.trace(wrapped_bert, (input_ids, attention_mask))

    traced_model.save("bert_traced.pt")
    print("BERT model traced and saved to bert_traced.pt")
    print(f"Dummy input_ids shape: {input_ids.shape}")
    print(f"Dummy attention_mask shape: {attention_mask.shape}")
    # Example output from traced model (Python)
    # last_hidden_state, pooler_output = traced_model(input_ids, attention_mask)
    # print(f"Traced model last_hidden_state shape: {last_hidden_state.shape}") # e.g., [1, 128, 768]
    # print(f"Traced model pooler_output shape: {pooler_output.shape}")       # e.g., [1, 768]

except Exception as e:
    print(f"Error during tracing: {e}")
    print("Ensure the BERT model's forward method returns Tensors or a tuple of Tensors.")
    print("If it returns a custom object (like BaseModelOutputWithPoolingAndCrossAttentions),")
    print("you might need to wrap the model to extract tensor outputs before tracing.")

# Also save the tokenizer's vocab for C++ (though a real C++ WordPiece tokenizer is needed)
# tokenizer.save_vocabulary("./bert_vocab") # This saves vocab.txt