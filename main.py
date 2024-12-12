import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

# Load Dataset
file_path = 'amazon_cells_labelled.txt'
data = pd.read_csv(file_path, sep='\t', header=None, names=["review", "sentiment"])

# Check for unusual characters
data['unusual_chars'] = data['review'].apply(lambda x: re.findall(r'[^\w\s]', x))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['review'])
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
sequences = tokenizer.texts_to_sequences(data['review'])

# Statistical Justification for Sequence Length
sequence_lengths = [len(text.split()) for text in data['review']]
max_sequence_length = int(np.percentile(sequence_lengths, 95))  # 95th percentile for sequence length

# Padding Sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Splitting Data into Training, Validation, and Test Sets
X_train, X_temp, y_train, y_temp = train_test_split(
    padded_sequences, data['sentiment'], test_size=0.3, random_state=42, stratify=data['sentiment']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.build(input_shape=(None, max_sequence_length))  # Explicitly build the model with input shape
model.summary()


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the Model on Test Data
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize Loss and Accuracy
# Loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model in TensorFlow
model.save('sentiment_analysis_model.h5')

loaded_model = load_model('sentiment_analysis_model.h5')
