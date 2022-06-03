from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # current directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# ------------------------------------------------------------------------------
#                                  Training
# ------------------------------------------------------------------------------
@xai_component(type="preprocess")
class LoadData(Component):
    """
    Load CSV file that provides the 'patterns', 'intents', and 'responses'. To edit the training data, you may add/delete row/pattern/response to the existing 'resource/sample.csv' or provide your own CSV file. Pattern is the possible user input and intent is the user intention(label). The model would try to match a particular input with its known intent. Response is the texts to send to user based on their input.
    Parameters:
        csv_file_path (str): Path to the CSV file that has columns 'patterns', 'intents', and 'responses' which provides the training sentences, label and responses to the particalur label respectively.
    Returns:
        sentences (list): List of sentences for training. Extracted from column 'pattern'.
        training_labels (array): Array of encoded label for training. Extracted from column 'intents'.
        responses (dict): Dictionary of label and responses pairs for inference. Extracted from column 'intents' and 'responses'.
    """

    csv_file_path: InCompArg[str]
    sentences: OutArg[list]
    training_labels: OutArg[any]
    responses: OutArg[any]

    def __init__(self):

        self.done = False
        self.csv_file_path = InCompArg(None)
        self.sentences = OutArg(None)
        self.training_labels = OutArg(None)
        self.responses = OutArg(None)

    def execute(self, ctx) -> None:
        import copy
        from sklearn.preprocessing import LabelEncoder

        x = []
        y = []
        z = {}
        df = pd.read_csv(self.csv_file_path.value)

        # convert pattern to list
        patterns = copy.deepcopy(
            df["patterns"]
            .str.replace('\["', "", regex=True)
            .str.replace('"\]', "", regex=True)
            .str.split('", "')
        )
        responses = copy.deepcopy(
            df["responses"]
            .str.replace('\["', "", regex=True)
            .str.replace('"\]', "", regex=True)
            .str.split('", "')
        )

        # data preparation
        for i in range(len(df)):
            for item in patterns[i]:
                x.append(item)
                y.append(df["intents"][i])
            z[df["intents"][i]] = responses[i]

        num_classes = df["intents"].nunique()
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(y)
        y = lbl_encoder.transform(y)

        self.sentences.value = x
        self.training_labels.value = y
        self.responses.value = z
        ctx.update({"num_classes": num_classes, "lbl_encoder": lbl_encoder})

        self.done = True


@xai_component(type="preprocess")
class Tokenize(Component):
    """
    Tokenizer that turns the texts into space-separated sequences of words (by default all punctuations will be removed), and these sequences are then split into lists of tokens.
    Parameters:
        sentences (list): List of sentences (sentences from `load_data`).
        vocab_size (int): Default: 1000. The maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept. Words out of vocabulary will be given "<OOV>"/out of vocabulary token.
        max_len (int): Default: 20. Maximum length of all sequences. If 'None', sequences will be padded to the length of the longest individual sequence.
    Returns:
        training_sentences (array): Array of list of sequences (each sequence is a list of integers).
    """

    sentences: InCompArg[list]
    vocab_size: InArg[int]
    max_len: InArg[int]
    training_sentences: OutArg[any]

    def __init__(self):

        self.done = False
        self.sentences = InCompArg(None)
        self.vocab_size = InArg(1000)
        self.max_len = InArg(20)
        self.training_sentences = OutArg(None)

    def execute(self, ctx) -> None:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # tokenize text and pad sequences
        tokenizer = Tokenizer(num_words=self.vocab_size.value, oov_token="<OOV>")
        tokenizer.fit_on_texts(self.sentences.value)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(self.sentences.value)
        sequences = pad_sequences(
            sequences, truncating="post", maxlen=self.max_len.value
        )

        self.training_sentences.value = sequences
        ctx.update(
            {
                "vocab_size": self.vocab_size.value,
                "max_len": self.max_len.value,
                "tokenizer": tokenizer,
            }
        )
        self.done = True


@xai_component(type="model")
class CustomModel(Component):
    """
    Custom neural network model that takes traininig sentences and pass them through an Embedding layer, a Global Average Pooling 1D layer, a number of Dense layers (nn_layer) with 'relu' activation and lastly a Dense layer with 'softmax' activation.
    Parameters:
        embedding_dim (int): Default: 16. Length of the vector for each word in Embedding layer. Larger value encloses more information but may require higher computation power and may be redundant if dataset is small.
        nn_layer (int): Default: 2. Number of Dense layers to build. Larger value can give better performance but may require higher computation power and may lead to model overfitting.
        optimizer (str): Default: 'adam'. String (name of optimizer) or tf.keras.optimizer instance. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.
        loss (str): Default: 'sparse_categorical_crossentropy'. Loss function. May be a string (name of loss function), or a tf.keras.losses.Loss instance. See https://www.tensorflow.org/api_docs/python/tf/keras/losses.
        metrics (list): Default: ['accuracy']. List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See https://www.tensorflow.org/api_docs/python/tf/keras/metrics.
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """

    embedding_dim: InArg[int]
    nn_layer: InArg[int]
    optimizer: InArg[str]
    loss: InArg[str]
    metrics: InArg[list]
    model: OutArg[any]

    def __init__(self):

        self.done = False
        self.embedding_dim = InArg(16)
        self.nn_layer = InArg(2)
        self.optimizer = InArg("adam")
        self.loss = InArg("sparse_categorical_crossentropy")
        self.metrics = InArg(["accuracy"])
        self.model = OutArg(None)

    def execute(self, ctx) -> None:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

        num_classes = ctx["num_classes"]
        vocab_size = ctx["vocab_size"]
        max_len = ctx["max_len"]

        model = Sequential()
        model.add(Embedding(vocab_size, self.embedding_dim.value, input_length=max_len))
        model.add(GlobalAveragePooling1D())
        for i in range(self.nn_layer.value):
            model.add(Dense(16, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss=self.loss.value,
            optimizer=self.optimizer.value,
            metrics=self.metrics.value,
        )

        model.summary()
        self.model.value = model
        self.done = True


@xai_component(type="model")
class Train(Component):
    """
    Train model with defined epochs and save model at 'model_output_path'. 'training_sentences' and their correspend 'training_labels will be useed to train the model.
    Parameters:
        model (tf.keras.Model): Compiled model. Use `custom_model` to build your model if you have not do so.
        training_sentences (array): Array of list of sentences (user input) for training.
        training_labels (array): Array of labels for each training sentence.
        model_output_path (str): Default: 'chat_model/exp1'. The path to store your trained model.
        epochs (int): Default: 500. Number of training iteration.
        verbose (boolean): Default: True. Print each epoch result if True.
        plot (boolean): Default: True. Plot graph if True.
    """

    model: InCompArg[any]
    training_sentences: InCompArg[any]
    training_labels: InCompArg[any]
    model_output_path: InArg[str]
    epochs: InArg[int]
    verbose: InArg[bool]
    plot: InArg[bool]

    def __init__(self):

        self.done = False
        self.model = InCompArg(None)
        self.training_sentences = InCompArg(None)
        self.training_labels = InCompArg(None)
        self.model_output_path = InArg("chat_model/exp1")
        self.epochs = InArg(500)
        self.verbose = InArg(True)
        self.plot = InArg(True)

    def execute(self, ctx) -> None:
        # training
        model = self.model.value
        history = model.fit(
            self.training_sentences.value,
            self.training_labels.value,
            epochs=self.epochs.value,
            verbose=self.verbose.value,
        )

        if self.plot.value:
            import matplotlib.pyplot as plt

            # summarize history for accuracy
            plt.plot(history.history["accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()

            # summarize history for loss
            plt.plot(history.history["loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()

        # to save the trained model
        output_path = self.model_output_path.value
        if not os.path.exists(output_path):
            # Create a new directory because it does not exist
            os.makedirs(output_path)

        model_path = os.path.join(output_path, "model")
        model.save(model_path)

        import pickle

        # to save the fitted tokenizer
        tokenizer = ctx["tokenizer"]
        with open(os.path.join(output_path, "tokenizer.pickle"), "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # to save the fitted label encoder
        lbl_encoder = ctx["lbl_encoder"]
        with open(os.path.join(output_path, "label_encoder.pickle"), "wb") as ecn_file:
            pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

        self.done = True


# ------------------------------------------------------------------------------
#                                  Inference
# ------------------------------------------------------------------------------
@xai_component(type="inference")
class SingleInference(Component):
    """
    Single sentence inference. This component take one text input and predict its intent. Also, gives a response if responses data is provided.
    Parameters:
        text (str): Input text for prediction.
        responses (dict): Dictionary of label and responses pairs. If 'None', only model result will be returned. Use `load_data` to load responses from csv if you have not do so.
        model_path (str): Default: 'chat_model/exp1'. Path to inference model. Use training workflow to train a model if you have not do so.
    """

    text: InCompArg[str]
    responses: InArg[any]
    model_path: InArg[str]

    def __init__(self):

        self.done = False
        self.text = InCompArg(None)
        self.responses = InArg(None)
        self.model_path = InArg("chat_model/exp1")

    def execute(self, ctx) -> None:
        from tensorflow import keras
        import pickle

        print("Input: ", self.text.value)

        exp_path = self.model_path.value
        model, tokenizer, lbl_encoder, max_len = load_model(exp_path)

        # make prediction
        result = model.predict(
            keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([self.text.value]),
                truncating="post",
                maxlen=max_len,
            )
        )
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        print("Result: ", tag[0])

        if self.responses.value:
            print("ChatBot:", np.random.choice(self.responses.value[tag[0]]))

        self.done = True


@xai_component(type="inference")
class Chat(Component):
    """
    Inference. User can give input and chatbot will give response for predicted label based on the input.
    Parameters:
        responses (dict): Dictionary of label and responses pairs. Use `load_data` to load responses from csv if you have not do so.
        model_path (str): Default: 'chat_model/exp1'. Path to inference model. Use training workflow to train a model if you have not do so.
    """

    responses: InCompArg[any]
    model_path: InArg[str]

    def __init__(self):

        self.done = False
        self.responses = InCompArg(None)
        self.model_path = InArg("chat_model/exp1")

    def execute(self, ctx) -> None:
        import random
        from tensorflow import keras
        import colorama
        import pickle

        colorama.init()
        from colorama import Fore, Style, Back

        exp_path = self.model_path.value
        model, tokenizer, lbl_encoder, max_len = load_model(exp_path)

        # start chatting!
        print(
            Fore.YELLOW
            + "Start messaging with the bot (type quit to stop)!"
            + Style.RESET_ALL
        )
        while True:
            # get user input
            print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
            inp = input()
            if inp.lower() == "quit":
                break

            # make prediction
            result = model.predict(
                keras.preprocessing.sequence.pad_sequences(
                    tokenizer.texts_to_sequences([inp]),
                    truncating="post",
                    maxlen=max_len,
                )
            )
            tag = lbl_encoder.inverse_transform([np.argmax(result)])

            # response
            print(
                Fore.GREEN + "ChatBot:" + Style.RESET_ALL,
                np.random.choice(self.responses.value[tag[0]]),
            )

        self.done = True


def load_model(exp_path):
    from tensorflow import keras
    import pickle

    # load trained model
    model = keras.models.load_model(os.path.join(exp_path, "model"))

    # load tokenizer object
    with open(os.path.join(exp_path, "tokenizer.pickle"), "rb") as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(os.path.join(exp_path, "label_encoder.pickle"), "rb") as enc:
        lbl_encoder = pickle.load(enc)

    # get input length from model
    max_len = model.layers[0].get_output_at(0).get_shape().as_list()[1]

    return model, tokenizer, lbl_encoder, max_len
