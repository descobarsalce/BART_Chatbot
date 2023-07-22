    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:42:53 2023

@author: descobarsalce
"""

# import os  # You can uncomment this import if needed

# Define the path to your working folder (optional)
# working_folder = "/Users/descobarsalce/Library/CloudStorage/Dropbox/CV/GITHUB"
# os.chdir(working_folder)

from BART_testing import ChatbotTrainer

def main():
    """Main function to create and train a chatbot using BART.

    This function defines the parameters, creates a chatbot using the provided parameters,
    loads the data, trains the BART-based chatbot, plots the loss evolution over epochs and batches,
    saves the trained model, and allows interaction with the chatbot (commented out).

    Note:
        To enable interaction with the chatbot, uncomment the 'while True' loop and the user_input section.

    """
    # Define the parameters
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

    # This is section to enable interaction with the chatbot
    interaction = False
    if interaction:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = chatbot.generate_response(user_input)
            print("Chatbot:", response)

if __name__ == "__main__":
    main()
