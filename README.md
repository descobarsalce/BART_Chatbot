# BART_Chatbot
Chatbot using pertained BART with Amazon/Kaggle interaction data.

## Chatbot Repository

This repository contains code for training and using a Chatbot based on the BART model from the Transformers library. The Chatbot can generate responses to user input in a conversational context. The repository includes the following files:

## Files

### BART_Chatbot.py

This script defines the `ChatbotTrainer` class responsible for training the Chatbot using the BART model. The class provides methods for loading data, training the model, generating responses, and saving/loading the trained model. The file also contains a `main()` function to run the training process and save the trained model to a file.

### ChatDATASET.py

This script contains the `ChatDatasetv2` class, which is a custom PyTorch dataset for preprocessing and managing the chat data. The class tokenizes the input text and formats it into question-answer pairs with optional conversation context. It also includes functions to clean the text data and create batches for training.

### chat_data.csv

This CSV file contains the chat data used for training the Chatbot. Each row represents a message in a conversation, including the conversation ID and the message text.

You can get sample data to train the model here: https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat

### requirements.txt

This file lists the required Python libraries and their versions for running the scripts in this repository. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train the Chatbot and save the trained model, run the `main()` function in `BART_testing.py`. The training parameters and data path can be configured in the script. The model will be saved as a pickle file.

```bash
python BART_testing.py
```

Alternatively, do the following to use step by step:

```python
    parameters = {
        "model_name": "facebook/bart-base",
        "max_length": 128,
        "batch_size": 64,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 100,
        "data_path": "topical_chat.csv",
        "max_data": 10000,
        "num_epochs": 10
    }

    # Create and train the chatbot
    chatbot = ChatbotTrainer(parameters)
    chatbot.load_data()
    chatbot.train_bart_chatbot()

    # Plot loss evolution over epochs and batches
    chatbot.plot_performance()

    # Save the trained model
    path_saving = 'model_trained.pkl'
    chatbot.save_model(path_saving)
```

To use the trained Chatbot for generating responses, load the trained model from the saved pickle file and call the `generate_response()` function.

```python
from BART_testing import ChatbotTrainer

# Load the trained model
chatbot = ChatbotTrainer.load_model('model_trained.pkl')

# Get response for user input
user_input = "Hello, how are you?"
response = chatbot.generate_response(user_input)
print("Chatbot:", response)
```

## Contributions

Contributions to this repository are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
