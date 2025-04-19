from evaluation.rag_system import RagSystem
import os
import logging
from lightrag_app.lightrag import LightRAG, QueryParam
from lightrag_app.lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag_app.lightrag.utils import EmbeddingFunc
from google import genai
from google.genai import types
import numpy as np
from sentence_transformers import SentenceTransformer
import time

class LightRagApp(RagSystem):
    def __init__(self):
        super().__init__("LightRag", "light_rag/work_dir")
        self.rag = LightRAG(
            working_dir=self.base_directory,
            llm_model_func=self.llm_model_func,
            #llm_model_name="bramvanroy/geitje-7b-ultra:Q4_K_M",
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=self.embedding_func,
            )
        )

        # Configure Gemini
        apikey = open("evaluation/apikey.txt", "r").read()
        self.gemini = client = genai.Client(api_key=apikey)

        self.full_texts = []

    def process_document(self, path):
        """Embeds a single file and updates shared data."""
        with open(path, "r", encoding="utf-8") as file:
            filename = os.path.basename(path)  # Deduce filename from file path
            title = filename.replace(".txt", "")
            content = file.read().strip()
            if content:
                full_text = f"{title}\n\n{content}" 
                self.full_texts.append(full_text)
                if len(self.full_texts) >= 10:
                    self.process_document_batch()

    def process_document_batch(self):
        print(f"Processing batch of {len(self.full_texts)}")
        self.rag.insert(self.full_texts)
        self.full_texts = []
    
    def save(self):
        """Done during process document."""
        pass

    def query(self, question) -> str:
        return self.rag.query(question, param=QueryParam(mode= 'mix', top_k=10, stream=False))\
        


        
    async def llm_model_func(self,
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        # 2. Combine prompts: system prompt, history, and user prompt
        if history_messages is None:
            history_messages = []

        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n"

        for msg in history_messages:
            # Each msg is expected to be a dict: {"role": "...", "content": "..."}
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        # Finally, add the new user prompt
        combined_prompt += f"user: {prompt}"

        # 3. Call the Gemini model
        response = self.gemini.models.generate_content(
            model = 'gemini-2.0-flash',
            contents=[combined_prompt]
        )

        # 4. Return the response text
        return response.text


    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        result = self.gemini.models.embed_content(
            model="text-embedding-004",
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=384),
        )
        return np.array([x.values for x in result.embeddings])
