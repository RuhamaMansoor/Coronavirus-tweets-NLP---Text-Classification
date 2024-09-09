# Coronavirus Tweets NLP Text Classification

This project uses Natural Language Processing (NLP) and machine learning to classify tweets about coronavirus into different sentiment categories. The tweets are classified as **Positive**, **Negative**, or **Neutral** based on their content.

## Key Features of the Project:
1. **Text Preprocessing**: The tweets are cleaned to remove unnecessary elements like URLs, mentions, hashtags, and numbers. The text is also converted to lowercase, and stopwords (commonly used words like "the", "is", etc.) are removed to focus on meaningful content.
   
2. **Sentiment Labels**: 
   - The original dataset contains 5 categories of sentiment:
     - Positive
     - Negative
     - Neutral
     - Extremely Positive
     - Extremely Negative
   - These labels are consolidated into 3 main categories:
     - **Positive** (includes Extremely Positive and Positive)
     - **Negative** (includes Extremely Negative and Negative)
     - **Neutral**

3. **Modeling**: 
   - The cleaned text data is tokenized (converted into sequences of numbers) and padded to ensure all sequences are of the same length.
   - A machine learning model is built using two architectures:
     1. **Bidirectional GRU** (Gated Recurrent Unit) neural network
     2. **LSTM** (Long Short-Term Memory) neural network

4. **Training the Models**: 
   - The models are trained using the processed data for sentiment classification.
   - The performance is evaluated using metrics such as **accuracy** and **loss**.

## Steps in the Code:

### 1. **Loading the Dataset**:
   - The dataset is loaded from a CSV file and contains tweets and their corresponding sentiment labels.

### 2. **Text Preprocessing**:
   - Tweets are cleaned by:
     - Removing URLs, mentions (@username), hashtags (#), and numbers.
     - Converting text to lowercase.
     - Removing stopwords.

### 3. **Tokenization and Padding**:
   - The tweets are tokenized into sequences of numbers.
   - Padding is applied to make sure all sequences are of equal length.

### 4. **Building the Model**:
   - Two different models are built:
     1. **Bidirectional GRU Model**: This model uses Bidirectional GRU layers to learn the sentiment from both past and future words in the sequence.
     2. **LSTM Model**: Another version uses LSTM layers to capture long-term dependencies in the tweet data.
   - The models include layers for embedding, dropout (to prevent overfitting), and dense (fully connected) layers for classification.

### 5. **Training and Evaluation**:
   - The models are trained using a 70% training and 30% testing split.
   - **Early Stopping** is applied to prevent overfitting by stopping the training process if the model performance does not improve for several epochs.
   - Accuracy and loss metrics are plotted for both training and validation datasets to monitor model performance.

### 6. **Results**:
   - After training, the accuracy and loss are visualized to understand how well the models have learned.
   - Both the Bidirectional GRU and LSTM models can be used to classify future tweets into the three sentiment categories.

## How to Run the Code:

1. Install the necessary libraries:
   ```bash
   pip install numpy pandas nltk keras tensorflow scikit-learn matplotlib plotly
   ```

2. Download the NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

3. Run the code step by step to load, preprocess, train, and evaluate the models.

4. Adjust the parameters (epochs, embedding dimension, model architecture) to further improve the results.

## Conclusion:
This project demonstrates how to use NLP techniques and deep learning models (GRU and LSTM) to classify tweets based on sentiment. You can further fine-tune the models to improve their performance on different datasets.
