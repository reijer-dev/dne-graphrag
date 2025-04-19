from evaluation.rag_system import RagSystem
from vanilla_rag.vanilla_rag_processor import VanillaRagProcessor
import json

class VanillaRag(RagSystem):
    def __init__(self):
        super().__init__("VanillaRag", "vanilla_rag/work_dir")
        settings = self.load_settings()

        EMBEDDINGS_PATH = settings["EMBEDDINGS_PATH"]

        self.processor = VanillaRagProcessor(
            embeddings_path=EMBEDDINGS_PATH,
        )

        
    def load_settings(self):
        """Load settings from a JSON file."""
        with open("./vanilla_rag/settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    
    def process_document(self, path):
        """Embeds a single file and updates shared data."""
        return self.processor.process_single_document(path)
    
    def save(self):
        """Done during process document."""
        self.processor.save()

    def query(self, question) -> str:
        return self.processor.ask_question(question)