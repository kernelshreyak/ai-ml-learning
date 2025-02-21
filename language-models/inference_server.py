from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
import re

app = FastAPI()

# Define request model
class PromptRequest(BaseModel):
    prompt: str
    top_k: int = 5  # Default to top 5 predictions

# Tokenizer function
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load the vocabulary
with open("vocab_v3.pkl", "rb") as f:
    vocab = pickle.load(f)

special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
for idx, token in enumerate(special_tokens):
    vocab[token] = idx

inv_vocab = {idx: word for word, idx in vocab.items()}

# Load model parameters
embed_size = 512
hidden_size = 512
num_layers = 4
dropout = 0.5
vocab_size = 10000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Model Class
class RNNLanguageModelv3(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5, tie_weights=True):
        super(RNNLanguageModelv3, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size, padding_idx=vocab['<pad>'])
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)
        
        if tie_weights and hidden_size == embed_size:
            self.fc.weight = self.embedding.weight

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        output = self.layer_norm(output)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(num_layers, batch_size, hidden_size),
                weight.new_zeros(num_layers, batch_size, hidden_size))

# Load the model
model = RNNLanguageModelv3(
    vocab_size=vocab_size, 
    embed_size=embed_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    dropout=dropout,
    tie_weights=True
).to(device)

model.load_state_dict(torch.load("rnn_language_model_v4_1epochs.pth", map_location=device))
model.eval()

# Function to predict next words
def predict_next_word(model, input_text, vocab, inv_vocab, top_k=5):
    model.eval()
    tokens = tokenize(input_text)
    input_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)

    with torch.no_grad():
        output, hidden = model(input_tensor, hidden)

    logits = output[0, -1]
    probabilities = torch.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_words = [inv_vocab[idx.item()] for idx in top_indices]

    return top_words

# FastAPI route
@app.post("/predict/v4/")
def get_predictions(request: PromptRequest):
    predicted_words = predict_next_word(model, request.prompt, vocab, inv_vocab, request.top_k)
    return {"prompt": request.prompt, "predictions": predicted_words}

# Run the FastAPI server with: uvicorn filename:app --reload
