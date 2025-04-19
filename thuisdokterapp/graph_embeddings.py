from sentence_transformers import SentenceTransformer
from rdflib import Graph, URIRef, Namespace, Literal
from urllib.parse import quote
import json
import numpy as np
import os

def sanitize_uri_part(text):
    """Sanitize text for use in URIs by replacing spaces and special characters"""
    if not text:
        return None
    return quote(text.strip().lower().replace(" ", "_").replace("/", "_"))

class GraphEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}
        
    def save_embeddings(self, cache_file="embeddings_cache.npz"):
        """Save embeddings cache to file"""
        if self.embedding_cache:
            np.savez_compressed(
                cache_file,
                embeddings=np.array(list(self.embedding_cache.values())),
                keys=np.array(list(self.embedding_cache.keys()))
            )
            print(f"✅ Embeddings cache saved to: {cache_file}")
            
    def load_embeddings(self, cache_file="embeddings_cache.npz"):
        """Load embeddings from cache file"""
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            self.embedding_cache = dict(zip(data['keys'], data['embeddings']))
            print(f"✅ Loaded {len(self.embedding_cache)} embeddings from cache")
            
    def get_embedding(self, text):
        """Get embedding from cache or compute new one"""
        if text not in self.embedding_cache:
            embedding = self.model.encode(text)
            self.embedding_cache[text] = embedding / np.linalg.norm(embedding)  # Normalize
        return self.embedding_cache[text]

    def create_embedded_graph(self, jsonl_path="processed_triples.jsonl", ttl_output="thuisdokter_graph.ttl"):
        """Create an RDF graph with embedded vectors from JSONL triples.
        
        Args:
            jsonl_path (str): Path to input JSONL file
            ttl_output (str): Path to output TTL file
            
        Returns:
            rdflib.Graph: The created RDF graph
        """
        g = Graph()
        EX = Namespace("http://example.org/")
        EMB = Namespace("http://example.org/embedding/")
        skipped = 0
        processed = 0

        def validate_triple(triple):
            """Validate a triple before processing."""
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                return False
            return all(isinstance(x, str) and x.strip() for x in triple)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    entry = json.loads(line)
                    for triple in entry.get("triples", []):
                        if not validate_triple(triple):
                            skipped += 1
                            continue
                            
                        subj, pred, obj = [x.strip() for x in triple]
                        
                        # Create URIRefs
                        s = URIRef(EX[sanitize_uri_part(subj)])
                        p = URIRef(EX[sanitize_uri_part(pred)])
                        o = URIRef(EX[sanitize_uri_part(obj)])
                        
                        # Generate embeddings using cache
                        subj_emb = self.get_embedding(subj)
                        obj_emb = self.get_embedding(obj)
                        
                        # Add triple to graph
                        g.add((s, p, o))
                        
                        # Add embeddings as literals
                        g.add((s, EMB['vector'], Literal(json.dumps(subj_emb.tolist()))))
                        g.add((o, EMB['vector'], Literal(json.dumps(obj_emb.tolist()))))
                        processed += 1
                        
                except Exception as e:
                    print(f"⚠️ Line {line_num} skipped: {str(e)}")
                    skipped += 1

        g.serialize(destination=ttl_output, format="turtle")
        print(f"✅ Processed {processed} valid triples")
        print(f"⚠️ Skipped {skipped} invalid triples")
        print(f"✅ RDF graph exported to: {ttl_output}")
        
        return g

    def query_similar(self, g, query_text, top_k=5):
        """Find similar nodes based on embedding similarity.
        
        Args:
            g (rdflib.Graph): The RDF graph to search
            query_text (str): Text to find similar nodes for
            top_k (int): Number of results to return
            
        Returns:
            list: Top-k similar nodes with similarity scores
        """
        query_emb = self.get_embedding(query_text)
        similarities = []
        EMB = Namespace("http://example.org/embedding/")
        
        for s, p, o in g:
            if p == EMB['vector']:
                try:
                    node_emb = np.array(json.loads(str(o)))
                    query_norm = np.linalg.norm(query_emb)
                    node_norm = np.linalg.norm(node_emb)
                    if query_norm == 0 or node_norm == 0:
                        print(f"⚠️ Zero norm detected for node: {s}")
                        continue
                    sim = np.dot(query_emb, node_emb) / (query_norm * node_norm)
                    similarities.append((s, sim))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ Skipping malformed embedding: {e}")
                    continue
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
# Create embedded graph
embedder = GraphEmbedder()
g = embedder.create_embedded_graph()

# Query similar concepts
query = "pijn in het gezicht"
similar_nodes = embedder.query_similar(g, query)
for node, similarity in similar_nodes:
    print(f"Node: {node}, Similarity: {similarity:.3f}")