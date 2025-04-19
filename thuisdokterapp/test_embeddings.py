import unittest
from graph_embeddings import GraphEmbedder
import os
import json

class TestGraphEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = GraphEmbedder()
        # Create test data
        self.test_data = '''{
            "triples": [["hoofdpijn", "is een", "symptoom"], 
                       ["migraine", "veroorzaakt", "hoofdpijn"]]
        }
        {
            "triples": [["koorts", "is een", "symptoom"], 
                       ["griep", "veroorzaakt", "koorts"]]
        }'''
        
        # Write test data to file
        with open('test_triples.jsonl', 'w', encoding='utf-8') as f:
            f.write(self.test_data)

    def tearDown(self):
        # Clean up test files
        files_to_remove = ['test_triples.jsonl', 'test_graph.ttl', 'embeddings_cache.npz']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    def test_create_embedded_graph(self):
        g = self.embedder.create_embedded_graph(
            jsonl_path="test_triples.jsonl",
            ttl_output="test_graph.ttl"
        )
        self.assertTrue(os.path.exists('test_graph.ttl'))
        self.assertGreater(len(list(g)), 0)

    def test_query_similar(self):
        g = self.embedder.create_embedded_graph(
            jsonl_path="test_triples.jsonl",
            ttl_output="test_graph.ttl"
        )
        similar_nodes = self.embedder.query_similar(g, "pijn in het hoofd", top_k=2)
        self.assertEqual(len(similar_nodes), 2)
        self.assertIsInstance(similar_nodes[0][1], float)

    def test_embedding_cache(self):
        self.embedder.get_embedding("test text")
        self.assertIn("test text", self.embedder.embedding_cache)
        self.embedder.save_embeddings()
        self.assertTrue(os.path.exists('embeddings_cache.npz'))

if __name__ == '__main__':
    unittest.main()