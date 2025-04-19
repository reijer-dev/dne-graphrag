from evaluation.rag_system import RagSystem
from thuisdokterapp.thuisdokter_graph_builder_modular import ThuisdokterGraphBuilder
from sentence_transformers import SentenceTransformer
from rdflib import Graph, URIRef, Namespace, Literal
import streamlit as st
import nltk
import thuisdokterapp.my_config
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from thuisdokterapp.my_config import OPENAI_API_KEY
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import json
from urllib.parse import quote
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai

def sanitize_uri_part(text):
    """Sanitize text for use in URIs by replacing spaces and special characters"""
    if not text:
        return None
    return quote(text.strip().lower().replace(" ", "_").replace("/", "_"))

class MichaelRag(RagSystem):
    def __init__(self):
        super().__init__("MichaelRag", "thuisdokterapp")
        self._init_session_state()

        nltk.download("punkt", quiet=True)

        self.client = self.get_openai_client()
        
        # Initialize sentence transformer for embeddings
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_cache = {}
        
        # Namespaces for RDF
        self.EX = Namespace("http://example.org/")
        self.EMB = Namespace("http://example.org/embedding/")

        # --- Load RDF ---
        TTL_FILE = "thuisdokterapp/thuisdokter_graph.ttl"
        if os.path.exists(TTL_FILE):
            self.rdf_graph = Graph()
            self.rdf_graph.parse(TTL_FILE, format="turtle")
        else:
            print(f"⚠️ Warning: RDF file not found at {TTL_FILE}. Run `process_document()` first.")
            self.rdf_graph = Graph()  # Initialize empty graph
        
        # Try to load embeddings cache
        self.load_embeddings()

        # Initialize nearest neighbors index
        self.nn_index = None
        self.index_nodes = []

        # Configure Gemini
        apikey = open("evaluation/apikey.txt", "r").read()
        genai.configure(api_key=apikey) # Replace with your actual API key
        self.gemini = genai.GenerativeModel('gemini-2.0-flash')

    # --- Session State ---
    def _init_session_state(self):
        if "comparison_log" not in st.session_state:
            st.session_state.comparison_log = []
            st.session_state.total_tokens = 0
            st.session_state.total_cost = 0.0

    def process_document(self, path=None):
        """
        Extracts knowledge triples from document
        """
        builder = ThuisdokterGraphBuilder(
            documents_path="./documents",  # or customize path
            output_jsonl="thuisdokterapp/processed_triples.jsonl",
            output_ttl="thuisdokterapp/thuisdokter_graph.ttl"
        )

        builder.extract_and_save_triple(path)

    def save(self):
        """Updates the RDF graph from the JSONL file."""
        builder = ThuisdokterGraphBuilder(
            documents_path="./documents",  # or customize path
            output_jsonl="thuisdokterapp/processed_triples.jsonl",
            output_ttl="thuisdokterapp/thuisdokter_graph.ttl"
        )
        print("🧠 Converting to RDF...")
        builder.jsonl_to_rdf()
        
        # After converting to RDF, create embedded graph
        self.create_embedded_graph()
        
        print("🎯 Graph updated with embeddings.")

        # Re-load the updated RDF graph into the app session
        self.rdf_graph = Graph()
        self.rdf_graph.parse("thuisdokterapp/thuisdokter_graph.ttl", format="turtle")

    def query(self, question) -> str:
        # Get query embedding once
        query_emb = self.get_embedding(question)
        
        # Pass embedding to functions that need it
        similar_nodes = self.query_similar(question, query_emb=query_emb, top_k=10)
        triple_context = self.graph_to_text_summary(query=question, query_emb=query_emb, limit=30)
        
        # Extract node text from URIs and prepare context
        context_nodes = []
        for node, similarity in similar_nodes:
            node_text = str(node).split('/')[-1].replace('_', ' ')
            context_nodes.append(f"{node_text} (similarity: {similarity:.3f})")
        
        # Create a context from both graph triples and similar nodes
        semantic_context = "Related concepts:\n" + "\n".join(context_nodes)
        combined_context = f"{triple_context}\n\n{semantic_context}"
        
        # GPT-only response
        #base_answer, base_usage = self.ask_gpt(question)

        # GraphRAG response
        graph_answer, graph_usage = self.ask_gpt(question, context=combined_context)

        # Cost calculations
        #gpt_cost = self.estimate_cost(base_usage)
        graph_cost = self.estimate_cost(graph_usage)
        #efficiency_gpt = (bleu + rouge + cosine) / (gpt_cost or 1e-6)
        #efficiency_graph = (bleu + rouge + cosine) / (graph_cost or 1e-6)

        # Log the result
        entry = {
            "question": question,
            #"gpt_answer": base_answer,
            "graphrag_answer": graph_answer,
            #"bleu": bleu,
            #"rouge_l": rouge,
            #"cosine_sim": cosine,
            "graphrag_tokens": graph_usage.total_token_count,
            #"gpt_cost": gpt_cost,
            "graphrag_cost": graph_cost,
            "total_tokens": graph_usage.total_token_count, # + base_usage.total_tokens,
            "total_cost": graph_cost # + gpt_cost,
            #"efficiency_gpt": efficiency_gpt,
            #"efficiency_graphrag": efficiency_graph,
        }

        st.session_state.comparison_log.append(entry)
        st.session_state.total_tokens += entry["total_tokens"]
        st.session_state.total_cost += entry["total_cost"]

        return graph_answer

    def graph_stats(self):
        """
        Print basic statistics about the RDF graph:
        - Total triples
        - Unique subjects, predicates, objects
        """
        g = self.rdf_graph
        subjects = set()
        predicates = set()
        objects = set()

        for s, p, o in g:
            subjects.add(s)
            predicates.add(p)
            objects.add(o)

        print("📊 Knowledge Graph Statistics:")
        print(f"• Total triples: {len(g)}")
        print(f"• Unique subjects: {len(subjects)}")
        print(f"• Unique predicates: {len(predicates)}")
        print(f"• Unique objects: {len(objects)}")

    # --- Embedding Functions ---
    def save_embeddings(self, cache_file="thuisdokterapp/embeddings_cache.npz"):
        """Save embeddings cache to file"""
        if self.embedding_cache:
            np.savez_compressed(
                cache_file,
                embeddings=np.array(list(self.embedding_cache.values())),
                keys=np.array(list(self.embedding_cache.keys()))
            )
            print(f"✅ Embeddings cache saved to: {cache_file}")
            
    def load_embeddings(self, cache_file="thuisdokterapp/embeddings_cache.npz"):
        """Load embeddings from cache file"""
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            self.embedding_cache = dict(zip(data['keys'], data['embeddings']))
            print(f"✅ Loaded {len(self.embedding_cache)} embeddings from cache")
            
    def get_embedding(self, text):
        """Get embedding from cache or compute new one"""
        if text not in self.embedding_cache:
            embedding = self.sbert.encode(text)
            self.embedding_cache[text] = embedding / np.linalg.norm(embedding)  # Normalize
        return self.embedding_cache[text]
    
    def create_embedded_graph(self, jsonl_path="thuisdokterapp/processed_triples.jsonl", 
                             ttl_output="thuisdokterapp/thuisdokter_graph.ttl"):
        """Create an RDF graph with embedded vectors from JSONL triples"""
        g = Graph()
        skipped = 0
        processed = 0

        def validate_triple(triple):
            """Validate a triple before processing."""
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                return False
            return all(isinstance(x, str) and x.strip() for x in triple)

        valid_triples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    entry = json.loads(line)
                    for triple in entry.get("triples", []):
                        if not validate_triple(triple):
                            skipped += 1
                            continue
                            
                        subj, pred, obj = [x.strip() for x in triple]
                        valid_triples.append((subj, pred, obj))
                        
                except Exception as e:
                    print(f"⚠️ Line {line_num} skipped: {str(e)}")
                    skipped += 1

        # Pre-compute embeddings in batches
        texts_to_encode = []
        for triple in valid_triples:
            subj, pred, obj = triple
            if subj not in self.embedding_cache:
                texts_to_encode.append(subj)
            if obj not in self.embedding_cache:
                texts_to_encode.append(obj)

        # Batch encode
        if texts_to_encode:
            batch_embeddings = self.sbert.encode(texts_to_encode)
            for i, text in enumerate(texts_to_encode):
                self.embedding_cache[text] = batch_embeddings[i] / np.linalg.norm(batch_embeddings[i])

        for triple in valid_triples:
            subj, pred, obj = triple
            
            # Create URIRefs
            s = URIRef(self.EX[sanitize_uri_part(subj)])
            p = URIRef(self.EX[sanitize_uri_part(pred)])
            o = URIRef(self.EX[sanitize_uri_part(obj)])
            
            # Generate embeddings using cache
            subj_emb = self.embedding_cache[subj]
            obj_emb = self.embedding_cache[obj]
            
            # Add triple to graph
            g.add((s, p, o))
            
            # Add embeddings as literals
            g.add((s, self.EMB['vector'], Literal(json.dumps(subj_emb.tolist()))))
            g.add((o, self.EMB['vector'], Literal(json.dumps(obj_emb.tolist()))))
            processed += 1

        g.serialize(destination=ttl_output, format="turtle")
        print(f"✅ Processed {processed} valid triples")
        print(f"⚠️ Skipped {skipped} invalid triples")
        print(f"✅ RDF graph exported to: {ttl_output}")
        
        # Update the instance's graph
        self.rdf_graph = g
        
        # Save embeddings cache
        self.save_embeddings()

        # Build nearest neighbors index
        self.build_similarity_index()
        
        return g

    def build_similarity_index(self):
        """Build nearest neighbors index for faster similarity search"""
        embeddings = []
        self.index_nodes = []
        
        for s, p, o in self.rdf_graph:
            if p == self.EMB['vector']:
                try:
                    node_emb = np.array(json.loads(str(o)))
                    embeddings.append(node_emb)
                    self.index_nodes.append(s)
                except:
                    continue
        
        if embeddings:
            self.nn_index = NearestNeighbors(n_neighbors=20, metric='cosine')
            self.nn_index.fit(np.array(embeddings))

    def query_similar(self, query_text, query_emb=None, top_k=5):
        if query_emb is None:
            query_emb = self.get_embedding(query_text)
        
        if self.nn_index is None:
            self.build_similarity_index()
        
        # Use the index for efficient search
        distances, indices = self.nn_index.kneighbors([query_emb], n_neighbors=min(top_k, len(self.index_nodes)))
        
        # Convert back to original format
        similarities = [(self.index_nodes[idx], 1.0 - dist) for idx, dist in zip(indices[0], distances[0])]
        return similarities

    # --- Utility Functions ---
    def get_openai_client(self):
        return OpenAI(api_key=OPENAI_API_KEY)

    def ask_gpt(self, question, context=""):
        context_part = f"Gebruik deze kennisdriehoeken:\n{context}\n\n" if context else ""
        prompt = f"""{context_part}Vraag: {question}\nAntwoord:"""
        response = self.gemini.generate_content(prompt)
        message = response.text.strip()
        return message, response.usage_metadata
        
    def graph_to_text_summary(self, query=None, query_emb=None, limit=50):
        # Add optimization for pre-computed embedding
        if query and query_emb is None:
            query_emb = self.get_embedding(query)
            
        """
        Create a text summary from graph triples, optionally filtered by relevance to a query
        
        Args:
            query (str, optional): Query to filter triples by relevance
            query_emb (np.ndarray, optional): Precomputed query embedding
            limit (int): Maximum number of triples to include
            
        Returns:
            str: Text summary of graph triples
        """
        if not query:
            # Just take the first 'limit' triples if no query
            summary = []
            count = 0
            for s, p, o in self.rdf_graph:
                if p != self.EMB['vector']:  # Skip embedding vectors
                    summary.append(f"{str(s).split('/')[-1].replace('_', ' ')} — {str(p).split('/')[-1].replace('_', ' ')} — {str(o).split('/')[-1].replace('_', ' ')}")
                    count += 1
                    if count >= limit:
                        break
        else:
            # Filter triples by relevance to query
            triple_scores = []
            
            # Get regular triples (not embedding vectors)
            for s, p, o in self.rdf_graph:
                if p != self.EMB['vector']:
                    # Create a text representation of the triple
                    triple_text = f"{str(s).split('/')[-1].replace('_', ' ')} {str(p).split('/')[-1].replace('_', ' ')} {str(o).split('/')[-1].replace('_', ' ')}"
                    
                    # Get embedding and compute similarity
                    triple_emb = self.get_embedding(triple_text)
                    sim = np.dot(query_emb, triple_emb)
                    
                    triple_scores.append((triple_text, sim))
            
            # Sort by similarity and take top 'limit'
            triple_scores.sort(key=lambda x: x[1], reverse=True)
            summary = [f"{triple} (relevance: {score:.3f})" for triple, score in triple_scores[:limit]]
        
        return "\n".join(summary)
        
    def estimate_cost(self, usage):
        return (usage.prompt_token_count * 0.01 + usage.candidates_token_count * 0.04) / 1000000

    def bleu_score(self, reference, candidate):
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)

    def rouge_l(self, reference, candidate):
        def lcs(X, Y):
            m, n = len(X), len(Y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if X[i] == Y[j]:
                        dp[i + 1][j + 1] = dp[i][j] + 1
                    else:
                        dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
            return dp[m][n]
        ref_tokens, cand_tokens = reference.split(), candidate.split()
        lcs_len = lcs(ref_tokens, cand_tokens)
        return lcs_len / len(ref_tokens) if ref_tokens else 0

    def embedding_similarity(self, text1, text2):
        emb1 = self.get_embedding(text1).reshape(1, -1)
        emb2 = self.get_embedding(text2).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]