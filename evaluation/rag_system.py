#Class is an interface for a RAG system that can be evaluated by the RagEvaluater class.
class RagSystem():
    def __init__(self, name, base_directory):
        self.name = name
        self.base_directory = base_directory

    def process_document(path):
        """Is called at training time to process a document for future use."""
        pass
    
    def query(question) -> str:
        """Is called at query time to answer a question. Time is managed."""
        pass
    
    def save():
        """Is called at training time to save the system."""
        pass