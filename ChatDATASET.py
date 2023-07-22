#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:52:58 2023

@author: descobarsalce
"""

from torch.utils.data import Dataset
from dataclasses import dataclass
import regex as re

@dataclass
class ChatExample:
    """A data class to hold chat examples.

    Attributes:
        input_text (str): The input text for the chat example.
        target_text (str): The target text for the chat example.
    """
    input_text: str
    target_text: str

class ChatDatasetv2(Dataset):
    """Custom dataset for chatbot training.

    Args:
        tokenizer: The tokenizer used to tokenize the data.
        df: The DataFrame containing the chat data.
        max_length (int): The maximum length of tokenized input sequences.
    """

    def __init__(self, tokenizer, df, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._process_data(df)

    def _process_data(self, df, include_context=False, max_context_length: int = 2):
        """Preprocess the data and create question-answer pairs.

        Args:
            df: The DataFrame containing the chat data.
            include_context (bool): Whether to include context or not.
            max_context_length (int): Maximum length of context for each example.

        Returns:
            A list of ChatExample objects representing question-answer pairs.
        """
        df['message'] = df['message'].apply(self._clean_text)
        grouped_df = df.groupby('conversation_id')['message'].apply(list).reset_index()

        question_answer_pairs = []

        for index, row in grouped_df.iterrows():
            conversation = row[1]

            if include_context:
                context_pairs = [ChatExample(input_text=' '.join(conversation[max(i - max_context_length, 0):i]),
                                             target_text=conversation[i]) for i in range(1, len(conversation))]
                question_answer_pairs.extend(context_pairs)
            else:
                consecutive_pairs = [ChatExample(input_text=conversation[i], target_text=conversation[i + 1])
                                     for i in range(len(conversation) - 1)]
                question_answer_pairs.extend(consecutive_pairs)

        return question_answer_pairs

    def _clean_text(self, text):
        """Clean the text by removing special characters and converting to lowercase.

        Args:
            text (str): The input text.

        Returns:
            The cleaned text.
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize_example(self, example):
        """Tokenize an example using the provided tokenizer.

        Args:
            example: A ChatExample object representing a question-answer pair.

        Returns:
            The tokenized input and labels as dictionaries of input_ids and attention_mask.
        """
        inputs = self.tokenizer(example.input_text, return_tensors='pt', max_length=self.max_length,
                                truncation=True, padding='max_length')
        labels = self.tokenizer(example.target_text, return_tensors='pt', max_length=self.max_length,
                                truncation=True, padding='max_length')
        return {"input_ids": inputs.input_ids[0], "attention_mask": inputs.attention_mask[0]}, \
               {"input_ids": labels.input_ids[0], "attention_mask": labels.attention_mask[0]}

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx):
        """Return a tokenized example and its label at the given index."""
        example = self.examples[idx]
        inputs, labels = self._tokenize_example(example)
        return inputs, labels
