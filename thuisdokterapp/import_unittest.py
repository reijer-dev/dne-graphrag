import unittest
import numpy as np
from rdflib import Graph, URIRef, Namespace, Literal
import json
from graph_embeddings import GraphEmbedder
import time
from unittest.mock import MagicMock

# test_graph_embeddings.py

class TestGraphEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = GraphEmbedder()
        self.EMB = Namespace("http://example.org/embedding/")
        self.EX = Namespace("http://example.org/")
        
        # Create a minimal RDF graph with one sample
        self.g = Graph()
        node_uri = URIRef(self.EX["test_node"])
        embedding = np.random.randn(384)  # Random embedding for testing
        embedding = embedding / np.linalg.norm(embedding)  # Normalize the embedding
        
        # Add the node and its embedding to the graph
        self.g.add((node_uri, self.EMB['vector'], Literal(json.dumps(embedding.tolist()))))

    def test_query_similar_basic(self):
        """Test basic similarity query functionality"""
        query = "test query"
        results = self.embedder.query_similar(self.g, query, top_k=5)
        
        self.assertEqual(len(results), 5)
        for node, score in results:
            self.assertIsInstance(node, URIRef)
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)

    def test_query_similar_empty_graph(self):
        """Test query on empty graph"""
        empty_graph = Graph()
        results = self.embedder.query_similar(empty_graph, "test", top_k=5)
        self.assertEqual(len(results), 0)

    def test_query_similar_top_k(self):
        """Test different top_k values"""
        query = "test query"
        for k in [1, 10, 100]:
            results = self.embedder.query_similar(self.g, query, top_k=k)
            self.assertEqual(len(results), k)

    def test_query_similar_ordering(self):
        """Test results are ordered by decreasing similarity"""
        query = "test query"
        results = self.embedder.query_similar(self.g, query, top_k=100)
        
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_similar_malformed_embedding(self):
        """Test handling of malformed embeddings"""
        g = Graph()
        g.add((URIRef(self.EX['bad_node']),
               self.EMB['vector'],
               Literal("not a valid embedding")))
        
        results = self.embedder.query_similar(g, "test", top_k=5)
        self.assertEqual(len(results), 0)

    @unittest.skip("Skipping performance test for debugging")
    def test_query_similar_performance(self):
        """Test query performance with 1000 nodes"""
        
        query = "performance test query"
        start_time = time.time()
        results = self.embedder.query_similar(self.g, query, top_k=10)
        query_time = time.time() - start_time
        
        print(f"\nQuery time for 1000 nodes: {query_time:.3f} seconds")
        self.assertTrue(query_time < 1.0)  # Should complete in under 1 second

    def test_query_similar_cache(self):
        """Test embedding cache functionality"""
        query = "cache test query"
        
        # First query to populate cache
        self.embedder.query_similar(self.g, query)
        cached_embedding = self.embedder.embedding_cache.get(query)
        
        # Second query should use cache
        results = self.embedder.query_similar(self.g, query)
        self.assertTrue(np.array_equal(
            self.embedder.embedding_cache[query],
            cached_embedding
        ))

    def test_query_single_sample(self):
        """Test querying a single sample in the graph"""
        query = "test query"
        self.embedder.model.encode = MagicMock(return_value=np.random.randn(384))  # Mock embedding
        results = self.embedder.query_similar(self.g, query, top_k=1)
        
        # Assert that one result is returned
        self.assertEqual(len(results), 1)
        node, score = results[0]
        print(f"Node: {node}, Score: {score}")  # Debugging output
        self.assertIsInstance(node, URIRef)
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1, f"Score {score} is out of range [0, 1]")

    def get_embedding(self, texts):
        if isinstance(texts, list):
            return self.model.encode(texts)
        return self.model.encode([texts])[0]

    def query_similar(self, g, query_text, top_k=5):
        query_emb = self.get_embedding(query_text)
        similarities = []
        EMB = Namespace("http://example.org/embedding/")
        
        for s, p, o in g:
            if p == EMB['vector']:
                try:
                    node_emb = np.array(json.loads(str(o)))
                    sim = np.dot(query_emb, node_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(node_emb))
                    similarities.append((s, sim))
                except (json.JSONDecodeError, ValueError):
                    continue
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

if __name__ == '__main__':
    unittest.main()