import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    inputs, targets = [], []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            inputs.append(parts[0].lower())
            targets.append(parts[1].lower())
    return inputs, targets

def clean_text(text):
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text

# Load and clean dataset
input_texts, target_texts = load_data('chatbot dataset.txt')  # Replace with your dataset file
input_texts = [clean_text(text) for text in input_texts]
target_texts = ['<start> ' + clean_text(text) + ' <end>' for text in target_texts]

# Initialize tokenizer
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)

# Add special tokens if not already in the vocabulary
if '<start>' not in tokenizer.word_index:
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
if '<end>' not in tokenizer.word_index:
    tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

# Update vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Convert texts to sequences
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_length, padding='post')
# Hyperparameters
embedding_dim = 256
units = 512
batch_size = 64
epochs = 20

# Encoder model
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c

# Decoder model
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, enc_output, state):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=state)
        x = self.fc(output)
        return x, state_h, state_c

# Initialize models
encoder = Encoder(vocab_size, embedding_dim, units)
decoder = Decoder(vocab_size, embedding_dim, units)

# Loss function and optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_state_h, enc_state_c = encoder(inp)
        dec_state = [enc_state_h, enc_state_c]
        
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * inp.shape[0], 1)

        for t in range(1, targ.shape[1]):
            predictions, state_h, state_c = decoder(dec_input, enc_output, dec_state)
            print(f"Predictions shape: {predictions.shape}, Target shape: {targ[:, t].shape}")  # Debug shapes
            loss += loss_object(targ[:, t], predictions[:, -1, :])
            dec_input = tf.expand_dims(targ[:, t], 1)
            dec_state = [state_h, state_c]

    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return loss / targ.shape[1]

# Training loop
# Training loop
for epoch in range(epochs):
    total_loss = 0
    num_batches = len(input_sequences) // batch_size

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        # Handle the last batch if it has fewer samples
        if end_idx > len(input_sequences):
            end_idx = len(input_sequences)

        inp = input_sequences[start_idx:end_idx]
        targ = target_sequences[start_idx:end_idx]

        # Ensure the batch size is consistent
        if inp.shape[0] != batch_size:
            continue  # Skip the last incomplete batch

        batch_loss = train_step(inp, targ)
        total_loss += batch_loss

    print(f'Epoch {epoch + 1}, Loss: {total_loss / num_batches}')
# Prediction function
def predict_with_prompt(prompt):
    cleaned_prompt = clean_text(prompt)
    input_sequence = tokenizer.texts_to_sequences([cleaned_prompt])
    input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    
    enc_output, enc_state_h, enc_state_c = encoder(input_sequence)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    dec_state = [enc_state_h, enc_state_c]
    
    result = []
    for _ in range(max_length):
        predictions, state_h, state_c = decoder(dec_input, enc_output, dec_state)
        predicted_id = tf.argmax(predictions[0, -1, :]).numpy()
        
        if predicted_id == tokenizer.word_index['<end>']:
            break
        
        result.append(tokenizer.index_word.get(predicted_id, ''))
        dec_input = tf.expand_dims([predicted_id], 0)
        dec_state = [state_h, state_c]
    
    return f"You: {prompt}\nAI: {' '.join(result)}"

# Interactive chatbot loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("AI: Goodbye!")
        break
    print(predict_with_prompt(user_input))
