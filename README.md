# Coronavirus Tweets NLP Text Classification

## Overview

This project uses natural language processing (NLP) techniques to classify tweets related to the coronavirus pandemic. The goal is to categorize the sentiment of the tweets into three classes: Positive, Negative, and Neutral. This classification is achieved through machine learning models such as Bidirectional GRU and LSTM.

## Dataset

The dataset used is `Corona_NLP_train.csv`, which contains tweets about the coronavirus along with their sentiment labels.

## Requirements

To run this project, install the following dependencies:

```bash
pip install numpy pandas nltk keras tensorflow matplotlib plotly scikit-learn
```

Additionally, download the NLTK stopwords:

```bash
import nltk
nltk.download('stopwords')
```

## Preprocessing

1. **Sentiment Adjustment**: The original dataset has five sentiment labels ("Extremely Positive", "Positive", "Neutral", "Negative", "Extremely Negative"). These are mapped into three categories:
   - Extremely Positive → Positive
   - Positive → Positive
   - Neutral → Neutral
   - Negative → Negative
   - Extremely Negative → Negative

2. **Text Cleaning**: 
   - Removal of URLs, mentions, hashtags, numbers, and special characters.
   - Conversion to lowercase and removal of stopwords.

## Models

Two models are implemented for classification:

1. **Bidirectional GRU Model**:
   - Embedding Layer
   - Bidirectional GRU Layer
   - Global Average Pooling
   - Dropout Layers
   - Dense Layers for classification

2. **LSTM Model**:
   - Embedding Layer
   - LSTM Layer
   - Global Average Pooling
   - Dropout Layers
   - Dense Layers for classification

## Model Training

Both models are trained on the preprocessed tweet data. The dataset is split into training and test sets using an 80/20 ratio. 

- Optimizer: `adam`
- Loss function: `categorical_crossentropy`
- Evaluation metrics: `accuracy`

**Example of training the Bidirectional GRU Model**:

```python
history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
```

**Example of training the LSTM Model** with early stopping:

```python
history = model_1.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[early_stopping])
```

## Results

The model's performance is evaluated using accuracy and loss over the training epochs, with visualizations created using Matplotlib:

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

## Usage

1. Preprocess the dataset using the provided code.
2. Train the model by running the provided model scripts.
3. Visualize the accuracy and loss during the training process.
