import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import pickle
from typing import List, Dict, Any
import textwrap
import uuid

# Try to import required libraries, install if missing
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import gradio as gr
except ImportError:
    import subprocess
    import sys
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "sentence-transformers", "faiss-cpu", "beautifulsoup4", 
                          "requests", "gradio"])
    from sentence_transformers import SentenceTransformer
    import faiss
    import gradio as gr

# Configuration
CACHE_DIR = "cache"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for embeddings
DEFAULT_IEP_URL = "https://sites.ed.gov/idea/statuteregulations/"  # IDEA regulations

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

class SPEDLawVectorDB:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []  # List to store text chunks and their sources
        
    def scrape_iep_laws(self, url: str) -> List[Dict[str, str]]:
        """Scrape IEP laws from the provided URL"""
        print(f"Scraping IEP laws from {url}...")
        
        # Make request to the website
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch content from {url}")
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        laws = []
        
        # Find all sections that might contain law information
        # This needs to be customized based on the specific website structure
        content_sections = soup.find_all(['div', 'section', 'article'])
        
        for section in content_sections:
            # Find paragraphs with substantial text
            paragraphs = section.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 100:  # Only consider substantial chunks of text
                    # Try to find a heading for this section
                    heading = self._extract_title(p, soup)
                    laws.append({
                        "content": text,
                        "source": url,
                        "title": heading
                    })
        
        # If no structured content was found, fall back to all paragraphs
        if not laws:
            all_paragraphs = soup.find_all('p')
            for p in all_paragraphs:
                text = p.get_text().strip()
                if len(text) > 100:
                    laws.append({
                        "content": text,
                        "source": url,
                        "title": "IDEA Regulations"
                    })
        
        print(f"Scraped {len(laws)} law chunks")
        return laws
    
    def _extract_title(self, element, soup):
        """Try to extract a relevant title for the law chunk"""
        # Try to find the nearest heading
        for heading in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            prev_heading = element.find_previous(heading)
            if prev_heading:
                return prev_heading.get_text().strip()
        
        # If no heading found, use the page title
        title_element = soup.find('title')
        if title_element:
            return title_element.get_text().strip()
        
        return "IDEA Regulations"
    
    def chunk_text(self, laws: List[Dict[str, str]], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
        """Split long texts into smaller chunks with some overlap"""
        chunked_laws = []
        
        for law in laws:
            content = law["content"]
            if len(content) <= chunk_size:
                chunked_laws.append(law)
            else:
                # Split into chunks
                for i in range(0, len(content), chunk_size - overlap):
                    chunk = content[i:i + chunk_size]
                    if len(chunk) >= 100:  # Only keep substantial chunks
                        chunked_law = law.copy()
                        chunked_law["content"] = chunk
                        chunked_laws.append(chunked_law)
        
        return chunked_laws
    
    def create_vector_database(self, laws: List[Dict[str, str]]):
        """Create a vector database from the laws"""
        print("Creating vector database...")
        
        # Extract the text content from each law
        texts = [law["content"] for law in laws]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = laws
        
        print(f"Vector database created with {len(laws)} chunks")
    
    def save_vector_database(self, url_hash):
        """Save the vector database to disk"""
        # Use hashed URL to create unique filenames
        vector_db_path = f"{CACHE_DIR}/iep_laws_vector_db_{url_hash}.faiss"
        metadata_path = f"{CACHE_DIR}/iep_laws_metadata_{url_hash}.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, vector_db_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Vector database saved to {vector_db_path}")
        return vector_db_path, metadata_path
    
    def load_vector_database(self, url_hash):
        """Load the vector database from disk"""
        vector_db_path = f"{CACHE_DIR}/iep_laws_vector_db_{url_hash}.faiss"
        metadata_path = f"{CACHE_DIR}/iep_laws_metadata_{url_hash}.pkl"
        
        if os.path.exists(vector_db_path) and os.path.exists(metadata_path):
            # Load FAISS index
            self.index = faiss.read_index(vector_db_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"Loaded vector database with {len(self.metadata)} chunks")
            return True
        else:
            print("No existing vector database found")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector database for chunks relevant to the query"""
        if not self.index:
            raise Exception("Vector database not initialized")
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means no result
                results.append({
                    "content": self.metadata[idx]["content"],
                    "source": self.metadata[idx]["source"],
                    "title": self.metadata[idx]["title"],
                    "score": float(scores[0][i])
                })
        
        return results

class SPEDLawChatbot:
    def __init__(self):
        self.vector_db = SPEDLawVectorDB()
        self.url_databases = {}  # Map URLs to their database paths
        
    def get_url_hash(self, url):
        """Create a unique hash for the URL to use in filenames"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, url))[:8]
        
    def initialize_for_url(self, url):
        """Initialize the chatbot for a specific URL"""
        url_hash = self.get_url_hash(url)
        
        # Try to load existing vector database
        if not self.vector_db.load_vector_database(url_hash):
            # If not found, scrape the website and create a new one
            try:
                laws = self.vector_db.scrape_iep_laws(url)
                chunked_laws = self.vector_db.chunk_text(laws)
                self.vector_db.create_vector_database(chunked_laws)
                vector_db_path, metadata_path = self.vector_db.save_vector_database(url_hash)
                self.url_databases[url] = (vector_db_path, metadata_path)
                return True, f"Successfully processed {url}"
            except Exception as e:
                return False, f"Error processing {url}: {str(e)}"
        else:
            return True, f"Loaded existing database for {url}"
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the vector database"""
        # Search for relevant content
        results = self.vector_db.search(question, top_k=3)
        
        if not results:
            return "I couldn't find any information related to your question about IEP laws."
        
        # Format the response
        response = "Here's what I found about your question:\n\n"
        
        for i, result in enumerate(results):
            response += f"Source: {result['title']}\n"
            response += f"Relevance: {result['score']:.2f}\n"
            response += f"{result['content']}\n\n"
        
        response += "This information is sourced from IDEA regulations. For official legal advice, please consult with a special education attorney."
        
        return response

# Gradio interface functions
def process_url(url):
    """Process a URL and add it to the chatbot's knowledge base"""
    if not url:
        url = DEFAULT_IEP_URL
        
    chatbot = SPEDLawChatbot()
    success, message = chatbot.initialize_for_url(url)
    return message

def answer_question(url, question):
    """Answer a question using the chatbot"""
    if not url:
        url = DEFAULT_IEP_URL
        
    if not question:
        return "Please enter a question."
        
    chatbot = SPEDLawChatbot()
    success, message = chatbot.initialize_for_url(url)
    
    if not success:
        return f"Error: {message}"
        
    answer = chatbot.answer_question(question)
    return answer

def run_interactive_session():
    """Run an interactive chat session in the terminal"""
    print("=" * 80)
    print("SPED Law Chatbot")
    print("Ask questions about IEP laws and regulations")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 80)
    
    chatbot = SPEDLawChatbot()
    chatbot.initialize_for_url(DEFAULT_IEP_URL)
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using SPED Law Chatbot. Goodbye!")
            break
        
        answer = chatbot.answer_question(question)
        
        # Format and print the answer with line wrapping
        print("\nAnswer:")
        wrapped_answer = textwrap.fill(answer, width=80)
        print(wrapped_answer)
        print("\n" + "-" * 80)

def create_gradio_interface():
    """Create and launch the Gradio interface"""
    with gr.Blocks(title="SPED Law Chatbot") as interface:
        gr.Markdown("# SPED Law Chatbot")
        gr.Markdown("Ask questions about special education laws and regulations. You can use the default IDEA regulations URL or enter a different source URL.")
        
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(
                    label="URL of IEP Laws (optional)", 
                    placeholder=DEFAULT_IEP_URL,
                    info="Enter a URL containing IEP laws or leave blank to use default IDEA regulations"
                )
                process_button = gr.Button("Process URL")
                url_status = gr.Textbox(label="Status", interactive=False)
                
                process_button.click(
                    fn=process_url,
                    inputs=[url_input],
                    outputs=[url_status]
                )
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question", 
                    placeholder="What is the timeline for IEP evaluation?",
                    lines=2
                )
                submit_button = gr.Button("Ask")
                
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer", 
                    interactive=False,
                    lines=15
                )
                
        submit_button.click(
            fn=answer_question,
            inputs=[url_input, question_input],
            outputs=[answer_output]
        )
        
        gr.Markdown("### Disclaimer")
        gr.Markdown("This chatbot provides information sourced from public regulations. For official legal advice, please consult with a special education attorney.")
        
    return interface

def main():
    # Check if command-line or Gradio mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Command-line mode
        run_interactive_session()
    else:
        # Gradio interface mode
        interface = create_gradio_interface()
        interface.launch(share=True)

if __name__ == "__main__":
    main()