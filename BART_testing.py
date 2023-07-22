
import torch
import os
import pandas as pd
import warnings
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from tqdm import tqdm
from ChatDATASET import ChatDatasetv2
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import pickle
import logging
from typing import List

warnings.filterwarnings("ignore", category=FutureWarning)

class ChatbotTrainer:
    """
    A class to train and use a Chatbot using the BART model from the Transformers library.

    Args:
        model_name (str): Name of the BART model to use.
        max_length (int): Maximum length of input and output sequences after tokenization.
        batch_size (int): Batch size for training and inference.
        learning_rate (float): Learning rate for the AdamW optimizer.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before backpropagation.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.

    Attributes:
        tokenizer (BartTokenizer): The BART tokenizer used for text preprocessing.
        model (BartForConditionalGeneration): The BART model for conditional text generation.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        optimizer (AdamW): The AdamW optimizer used for training.
        dataset (ChatDatasetv2): The dataset for training.
        dataloader (DataLoader): The DataLoader for batch processing.
        epoch_losses (List[float]): List to store the average loss per epoch during training.
        batch_losses (List[float]): List to store the loss per batch during training.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before backpropagation.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.
        max_length (int): Maximum length of input and output sequences after tokenization.
        batch_size (int): Batch size for training and inference.
    """

    def __init__(self, config):
        self.tokenizer = BartTokenizer.from_pretrained(config.get("model_name", "facebook/bart-base"))
        self.model = BartForConditionalGeneration.from_pretrained(config.get("model_name", "facebook/bart-base"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.get("learning_rate", 1e-5))
        self.dataset: ChatDatasetv2 = None
        self.dataloader: DataLoader = None
        self.epoch_losses: List[float] = []
        self.batch_losses: List[float] = []
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
        self.warmup_steps = config.get("warmup_steps", 500)
        self.max_length = config.get("max_length", 128)
        self.batch_size = config.get("batch_size", 32)
        self.data_path = config.get("data_path", "")
        self.max_data = config.get("max_data", 2000)
        self.num_epochs = config.get("num_epochs", 3)
        
    def load_data(self):
        """
        Load and preprocess the chat data for training.

        Args:
            data_path (str): Path to the CSV file containing the chat data.
            max_data (int, optional): Maximum number of data samples to load. Defaults to 2000.
        """
        df = pd.read_csv(self.data_path, delimiter=',', quotechar='"')
        if self.max_data:
            if len(df) > self.max_data:
                df = df.head(self.max_data)
        self.dataset = ChatDatasetv2(self.tokenizer, df, self.max_length)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=os.cpu_count() - 4)

    def train_bart_chatbot(self):
        """
        Train the Chatbot using the BART model.

        Args:
            num_epochs (int, optional): Number of epochs for training. Defaults to 3.
        """
        if self.dataset is None or self.dataloader is None:
            raise ValueError("Data has not been loaded. Use load_data method first.")

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            epoch_progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                      desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)

            effective_warmup_steps = self.warmup_steps // self.gradient_accumulation_steps
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda x: min((x + 1) / effective_warmup_steps, 1.0)
            )

            for step, batch in epoch_progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = {k: v.to(self.device) for k, v in labels.items()}
                outputs = self.model(**inputs, labels=labels["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Learning Rate Warmup (updated outside of gradient_accumulation_steps check)
                warmup_scheduler.step()

                self.batch_losses.append(loss.item() * self.gradient_accumulation_steps)

                batch_progress = (step + 1) / len(epoch_progress_bar) * 100
                epoch_progress_bar.set_postfix({"Loss": total_loss / (step + 1),
                                                "Batch Progress": f"{batch_progress:.2f}%"})

            avg_loss = total_loss / len(self.dataloader)
            self.epoch_losses.append(avg_loss)
            logging.info(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss}")

    def generate_response(self, input_text: str) -> str:
        """
        Generate a response for the input text.

        Args:
            input_text (str): Input text for generating the response.

        Returns:
            str: The generated response by the Chatbot.
        """
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=self.max_length,
                                          truncation=True, padding='max_length')
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(input_ids)

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def plot_performance(self, show_epoch_loss: bool = True, show_batch_loss: bool = True):
        """
        Plot the training performance.

        Args:
            show_epoch_loss (bool, optional): Whether to plot the epoch-wise average loss. Defaults to True.
            show_batch_loss (bool, optional): Whether to plot the batch-wise loss. Defaults to True.
        """
        if show_epoch_loss:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title("Loss Evolution over Epochs")
            plt.grid(True)

    def save_model(self, file_path: str):
        """
        Save the trained ChatbotTrainer to a file.

        Args:
            file_path (str): Path to the file where the model and parameters will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, file_path: str) -> 'ChatbotTrainer':
        """
        Load the trained ChatbotTrainer from a file.

        Args:
            file_path (str): Path to the file containing the saved model and parameters.

        Returns:
            ChatbotTrainer: The loaded ChatbotTrainer instance.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)