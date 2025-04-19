
import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import BitsAndBytesConfig
import os
import ollama
import google.generativeai as genai


class VanillaRagProcessor:
    MAX_BUFFER_ENTRIES = 1
    MAX_INPUT_CHARS = 20000-4000

    def __init__(self, embeddings_path):
        self.embeddings_path = embeddings_path
        self.documents = []
        self.embeddings = []
        self.text_to_embed_buffer = []

        if os.path.isfile(self.embeddings_path):
            with open(self.embeddings_path, "r") as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.embeddings = data["embeddings"]

        # Preload the Nomic model
        self.embed_model_name = "nomic-embed-text"
        self.preload_model(self.embed_model_name)
        
        # Configure Gemini
        apikey = open("evaluation/apikey.txt", "r").read()
        genai.configure(api_key=apikey) # Replace with your actual API key
        self.gemini = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro-vision' for multimodal models

    def preload_model(self, model_name):
        # Check if the model exists locally or download it
        try:
            print(f"Checking if model '{model_name}' is available...")
            # Trigger the model to load/download if necessary
            ollama.pull(model=model_name)  # Minimal call to ensure it's preloaded
            print(f"Model '{model_name}' is ready for use.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    def process_single_document(self, file_path):
        """Process a single document and return its content and embedding."""

        with open(file_path, "r", encoding="utf-8") as file:
            filename = os.path.basename(file_path)  # Deduce filename from file path
            title = filename.replace(".txt", "")
            content = file.read().strip()
            if content:
                full_text = f"{title}\n\n{content}"
                embedding = ollama.embeddings(model=self.embed_model_name, prompt=full_text)['embedding']
                self.documents.append(full_text)
                self.embeddings.append(embedding)
        return False

    def ask_question(self, question, top_k=3, max_new_tokens=300):

        try:
            question_embedding = ollama.embeddings(model=self.embed_model_name, prompt=question)['embedding']
        except ollama.ResponseError as e:
            print(f"Ollama embedding error: {e}")
            return


        similarities = cosine_similarity([question_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Generate response from model
        context = "\n\n".join([self.documents[i] for i in top_indices])
        prompt = f"""You are a helpful assistant. Use the following context to answer the question concisely.
            Context:
            {context}

            Question: {question}
            Answer:"""

        response = self.gemini.generate_content(prompt)
        return response.text

    def save(self):
        """Save the embeddings to a file."""
        data = {"documents": self.documents, "embeddings": self.embeddings}

        with open(self.embeddings_path, "w") as f:
            json.dump(data, f)