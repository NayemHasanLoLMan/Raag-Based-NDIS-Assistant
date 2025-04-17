import os
import fitz  # PyMuPDF for PDF processing
import numpy as np
import json
import openai
from typing import Dict, Tuple
import logging
import time

import dotenv
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchPDFVectorizerOpenAI:
    def __init__(self, folder_path: str):
        """Initialize the BatchPDFVectorizer with a folder path and OpenAI API key from environment."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        
        self.folder_path = folder_path
        self.embedding_model = "text-embedding-3-small"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
        
        # Determined once the first embedding is created
        self.embedding_dimension = None
        logger.info(f"Initialized OpenAI client with embedding model: {self.embedding_model}")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from each page of a PDF file."""
        page_dict = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    page_dict[page_num + 1] = text.strip()
            doc.close()
            logger.info(f"Extracted {len(page_dict)} pages from {pdf_path}.")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return page_dict

    def get_embedding_with_retry(self, text: str, max_retries=3, backoff_factor=2):
        """Get embedding with retry logic for API failures."""
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=text
                )
                embedding = np.array(response['data'][0]['embedding'])
                
                # Set embedding dimension on first successful call
                if self.embedding_dimension is None:
                    self.embedding_dimension = embedding.shape[0]
                    logger.info(f"Detected embedding dimension: {self.embedding_dimension}")
                
                return embedding
            except Exception as e:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Embedding API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                logger.warning(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        # If all retries fail, return zeros
        logger.error(f"Failed to get embedding after {max_retries} attempts")
        if self.embedding_dimension is None:
            # If we don't have a dimension yet, we can't proceed
            raise RuntimeError("Failed to determine embedding dimension")
        return np.zeros(self.embedding_dimension)

    def vectorize_pages(self, pdf_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """Generate vector embeddings for each page using OpenAI Embedding API."""
        page_dict = self.extract_text_from_pdf(pdf_path)
        page_embeddings = {}

        for page_num, text in page_dict.items():
            if len(text) < 10:  # Skip extremely short content
                logger.warning(f"Skipping Page {page_num} in {pdf_path}: content too short ({len(text)} chars)")
                continue
                
            try:
                # OpenAI has token limits (not exactly character limits)
                # 8000 chars is a safe approximation for most texts
                truncated_text = text[:8000]
                
                embedding = self.get_embedding_with_retry(truncated_text)
                page_embeddings[page_num] = embedding
                logger.info(f"Generated embedding for Page {page_num} in {pdf_path}, shape: {embedding.shape}")
            except Exception as e:
                logger.error(f"Error processing Page {page_num} in {pdf_path}: {str(e)}")
        
        return page_embeddings, page_dict

    def process_all_files(self, output_path: str):
        """Process all PDF files in the folder and save embeddings."""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path contains a directory part
            os.makedirs(output_dir, exist_ok=True)
        
        all_embeddings = {}
        all_metadata = {}
        
        # First pass: determine embedding dimension from first successful embedding
        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(self.folder_path, file_name)
                page_embeddings, page_texts = self.vectorize_pages(file_path)
                
                if page_embeddings:  # If we got any embeddings
                    for page_num, embedding in page_embeddings.items():
                        page_id = f"{file_name}_{page_num}"
                        all_embeddings[page_id] = embedding
                        all_metadata[page_id] = {
                            "file_name": file_name,
                            "page_number": page_num,
                            "text": page_texts.get(page_num, "")[:1000],  # First 1000 chars as preview
                            "char_count": len(page_texts.get(page_num, ""))
                        }
        
        if not all_embeddings:
            raise RuntimeError("No embeddings were successfully generated")
            
        self.save_combined_embeddings(all_embeddings, all_metadata, output_path)
        logger.info(f"Processing complete. Combined embeddings saved to {output_path}")

    def save_combined_embeddings(self, embeddings: Dict[str, np.ndarray], 
                                metadata: Dict[str, Dict], output_path: str):
        """Save all embeddings and metadata to a single file."""
        if not embeddings:
            raise ValueError("No embeddings to save")
            
        page_ids = list(embeddings.keys())
        
        # Verify all embeddings have the same dimension
        first_dim = embeddings[page_ids[0]].shape[0]
        for page_id in page_ids:
            if embeddings[page_id].shape[0] != first_dim:
                logger.error(f"Inconsistent embedding dimensions: {page_id} has {embeddings[page_id].shape[0]}, expected {first_dim}")
                # Fix by padding or truncating
                embeddings[page_id] = np.resize(embeddings[page_id], (first_dim,))
        
        # Stack all embeddings into a single array
        embedding_array = np.stack([embeddings[page_id] for page_id in page_ids], axis=0)
        logger.info(f"Embedding array shape before saving: {embedding_array.shape}")
        
        if embedding_array.ndim != 2:
            raise ValueError(f"Invalid embedding array shape: {embedding_array.shape}")
        
        try:
            np.savez(
                output_path,
                page_ids=page_ids,
                embeddings=embedding_array,
                metadata=json.dumps(metadata)
            )
            # Verify the file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                logger.info(f"Successfully saved {len(page_ids)} page embeddings to {output_path} ({file_size:.2f} MB)")
            else:
                logger.error(f"File was not created at {output_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings to {output_path}: {str(e)}")
            raise

def main():
    FOLDER_PATH = 'D:\\Sam_Project\\knowledge_base'
    OUTPUT_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_openai.npz'
    
    try:
        vectorizer = BatchPDFVectorizerOpenAI(FOLDER_PATH)
        vectorizer.process_all_files(OUTPUT_PATH)
    except Exception as e:
        logger.critical(f"Critical error during vectorization: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()