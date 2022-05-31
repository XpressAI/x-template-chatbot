# Xircuits Project Template

This template allows you to train a chat model and start a chatbot in a terminal.

It consists of these 6 components:

- Dataset preparation: Load and preprocess dataset.

  - `LoadData` : Load CSV file that provides the 'patterns', 'intents', and 'responses'. To edit the training data, you may add/delete row/pattern/response to the existing 'resource/sample.csv' or provide your own CSV file path. See more at this [section](#training).
  - `Tokenize` : Tokenizer that turns the texts into space-separated sequences of words (by default all punctuations will be removed), and these sequences are then split into lists of tokens.

- Model training: Build and compile the model for training.

  - `CustomModel` : Custom neural network model that takes traininig sentences and pass them through an Embedding layer, a Global Average Pooling 1D layer, a number of Dense layers (nn_layer) with 'relu' activation and lastly a Dense layer with 'softmax' activation.
  - `Train` : Train model with defined epochs and save model at 'model_output_path'. 'training_sentences' and their correspend 'training_labels will be useed to train the model.

- Inference: Load model and do prediction.
  - `SingleInference` : Single sentence inference. Thie component take one text input and predict its intent.
  - `Chat` : Streaming inference. User can give input and chatbot will give response for predicted label based on the input.

## Prerequisites

Python 3.9

## Installation

1. Clone this repository

```
git clone https://github.com/XpressAI/template-chatbot.git
```

2. Create virtual environments and install the required python packages

```
pip install -r requirements.txt
```

3. Run xircuits from the root directory

```
xircuits
```

## Workflow in this Template

### Training

#### [train_chat_model.xircuits](/xircuits-workflows/train_chat_model.xircuits)

Train a chat model that try to predict the intents based on sentences patterns.

<details>
<summary>What are/Why Patterns and Intents?</summary>

In order to answer questions, search from domain knowledge base and perform various other tasks to continue conversations with the user, a chatbot needs to understand what the users say or what they intend to do (identify user’s intention). The strategy here is to define different intents and make training samples for those intents.

Patterns in our case refer to training samples for different intents. Intents in case are the training categories/labels our model will predict. The model would try to match a particular input with its known intent.

See [resource/sample.csv](/resource/sample.csv) as dataset example.

</details>

<details>
<summary>How Can I Change the Training Data?</summary>

To edit the training data, you may add/delete row/pattern/response to the existing `resource/sample.csv`. Or, provide your own CSV file and provide its path at `LoadData` input, `csv_file_path`. The input CSV file should provide these three columns `patterns`, `intents`, and `responses`.

Terms We Use

- Patterns are training samples/possible user inputs for correspond intent.
- Intents are user intentions, also training categories/labels.
- Responses are response texts to send to user after getting the predicted tag from model with user input (only used during inference but not model training). See the workflow [chatbot.xircuits](/xircuits-workflows/chatbot.xircuits)

</details>

Example:
![model_training](/resource/images/model_training.gif)

### Inference

#### [inference.xircuits](/xircuits-workflows/inference.xircuits)

Single prediction on input text.<br>
Example:
![single_inference](/resource/images/single_inference.gif)

#### [chatbot.xircuits](/xircuits-workflows/chatbot.xircuits)

Initiate a chatbot.<br>
Example:
