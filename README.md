## Machine Translation Project: Deutsch to English

This GitHub repository contains code for a machine translation project focused on translating text from Deutsch (German) to English. The following code snippet showcases some key functionalities:

```python
import numpy as np
import tensorflow as tf

# Check TensorFlow version
print(tf.__version__)

# Load the pre-trained model
model = load_model('model.1234')

# Reshape test data
testX = testX.reshape(testX.shape[0], testX.shape[1])
testY = testY.reshape(testY.shape[0], testY.shape[1])

# Perform predictions
preds = model.predict(testY).astype("int32")

# Define helper functions
def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return word

# Tokenization
tok = Tokenizer(char_level=True)

# Reverse word mapping
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Convert predictions to text
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j].all(), eng_tokenizer)
        if j > 0:
            if t == get_word(i[j-1].all(), eng_tokenizer):
                temp.append(t)
            else:
                temp.append('')
    preds_text.append(' '.join(temp))

# Decode sequences
def decode_sequences(tokenizer, length, lines):
    seq = tokenizer.sequences_to_texts(lines)
    return seq

words = decode_sequences(eng_tokenizer, eng_length, preds)

# Create DataFrame with predicted translations
pred_df = pd.DataFrame({'actual': test[:, 0], 'predicted': words})

# Display DataFrame
pd.set_option('display.max_colwidth', 200)
```

This repository aims to provide an end-to-end solution for Deutsch to English translation using machine learning techniques. It includes code examples, documentation, and resources for better understanding and implementation of machine translation models.
