# Text Generation with LSTM Neural Network on Song Lyrics

This project aims to build a text generation model using an LSTM neural network trained on a dataset of song lyrics. The model learns the patterns and structures in the lyrics and generates new lyrics based on what it has learned.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Text generation is a fascinating application of Natural Language Processing (NLP) and deep learning, where a model learns to generate new text that resembles the input data it was trained on. In this project, we focus on generating song lyrics by training an LSTM (Long Short-Term Memory) neural network on a dataset of song lyrics.

## Dataset

The dataset used in this project is `lyrics.csv`, which contains song lyrics along with metadata such as the title, film, year, singer, composer, and lyricist. The lyrics are in a transliterated format.

**Sample Data Structure:**

| Title                                      | Film          | Year | Singer                                      | Composer           | Lyricist       | Lyrics                            |
|--------------------------------------------|---------------|------|---------------------------------------------|--------------------|----------------|------------------------------------|
| tuu ne o rangiile kaisaa jaaduu kiyaa      | kudrat        | -1   | lata                                        | r d burman         | majrooh        | tuu ne o rangiile kaisaa...        |
| main tujhase milane aaii mandir jaane ke...| heeraa        | 1973 | rafi lata                                    | kalyanjianandji    | indeevar       | la main tujhase milane aaii...     |
| ...                                        | ...           | ...  | ...                                         | ...                | ...            | ...                                |

## Data Preprocessing

Data preprocessing is a crucial step in preparing the text data for training the neural network. The following steps were taken:

1. **Data Cleaning**: Removing unwanted characters, tabs, backslashes, specific placeholders like `'threedots'`, numbers, and punctuation.
2. **Normalization**: Converting all text to lowercase and normalizing whitespace.
3. **Tokenization**: Using Keras' `Tokenizer` to convert text to sequences of integers.
4. **Sequence Generation**: Creating input-output pairs where the input is a sequence of words, and the output is the next word.
5. **Padding**: Padding sequences to have the same length using dynamic padding within the dataset.

**Example of Cleaned Text:**

```plaintext
la main tujhase milane aaii mandir jaane ke bahaane baabul se jhuuth bolii sakhiyon se jhuuth bolii main ban gaii bilkul bholii ra are o bholii tuu jhuuth nahiin bolii tuu jhuuth kahaan bolii pyaar ko hii puujaa kahate hain pyaar ke paravaane la main tujhase milane ra ho aankhon men jab terii suurat phir koii muurat kyaa hai pyaar kiyaa hai jisane use puujaa kii zaruurat kyaa hai...
```

## Model Architecture

The model is built using TensorFlow and Keras and consists of the following layers:

- **Embedding Layer**: Converts word indices to dense vectors of fixed size.
- **LSTM Layers**: Four LSTM layers to capture temporal dependencies in the sequence data.
- **Dropout Layers**: Added after each LSTM layer to prevent overfitting by randomly setting input units to 0 with a frequency of 20% during training.
- **Dense Layer**: Outputs a probability distribution over the vocabulary with a softmax activation function.

**Model Summary:**

```plaintext
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 350, 50)           541400
 lstm (LSTM)                 (None, 350, 128)          91648
 dropout (Dropout)           (None, 350, 128)          0
 lstm_1 (LSTM)               (None, 350, 128)          131584
 dropout_1 (Dropout)         (None, 350, 128)          0
 lstm_2 (LSTM)               (None, 350, 256)          394240
 dropout_2 (Dropout)         (None, 350, 256)          0
 lstm_3 (LSTM)               (None, 128)               197120
 dropout_3 (Dropout)         (None, 128)               0
 dense (Dense)               (None, 10828)             1397652
=================================================================
Total params: 2,716,644
Trainable params: 2,716,644
Non-trainable params: 0
_________________________________________________________________
```

## Training

The model is trained using the following configurations:

- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metrics**: `accuracy`
- **Batch Size**: `500`
- **Epochs**: `50`
- **Shuffling**: The dataset is shuffled with a buffer size of `10000` for better training performance.

The model is trained to predict the next word in a sequence given the previous words. After training, the model's weights are saved to `text_generation_model.h5`.

## Installation

To run this project, you'll need to have Python 3.x installed along with the following packages:

- `tensorflow`
- `keras`
- `pandas`
- `numpy`

You can install the required packages using `pip`:

```bash
pip install tensorflow pandas numpy
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vpratap007/your-repo-name.git
   cd your-repo-name
   ```

2. **Place the Dataset:**

   Ensure that you have the `lyrics.csv` file in the root directory of the project or update the path in the script accordingly.

3. **Run the Script:**

   ```bash
   python text_generation.py
   ```

   This will start the training process. Make sure you have enough computational resources, as training can be intensive.

4. **Generating Text:**

   After training, you can use the trained model to generate new song lyrics. Here's an example of how to do that:

   ```python
   import tensorflow as tf
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   import numpy as np

   # Load the trained model
   model = tf.keras.models.load_model('text_generation_model.h5')

   # Load the tokenizer
   tokenizer = ... # Load or initialize the tokenizer used during training

   def generate_text(seed_text, next_words):
       for _ in range(next_words):
           token_list = tokenizer.texts_to_sequences([seed_text])[0]
           token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
           predicted = model.predict(token_list, verbose=0)
           predicted_word_index = np.argmax(predicted, axis=-1)[0]
           output_word = tokenizer.index_word[predicted_word_index]
           seed_text += " " + output_word
       return seed_text

   # Example usage
   print(generate_text("main tujhase", 50))
   ```

## Results

The model generates new sequences of lyrics that mimic the style and content of the training data. The quality of the generated text depends on the diversity and size of the dataset, as well as the training time.

**Sample Generated Text:**

```plaintext
main tujhase milane aaii mandir jaane ke bahaane baabul se jhuuth bolii sakhiyon se jhuuth bolii main ban gaii bilkul bholii ra are o bholii tuu jhuuth nahiin bolii tuu jhuuth kahaan bolii pyaar ko hii puujaa kahate hain pyaar ke paravaane la main tujhase milane ra ho aankhon men jab terii suurat phir koii muurat kyaa hai pyaar kiyaa hai...
```

*Note: The generated text is for illustrative purposes and may vary based on the actual training.*

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgements

- **TensorFlow and Keras**: For providing the deep learning framework.
- **Pandas and NumPy**: For data manipulation and numerical computations.
