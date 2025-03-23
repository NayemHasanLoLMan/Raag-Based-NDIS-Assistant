# # import re
# # import openai
# # import numpy as np
# # import os
# # import fitz  # PyMuPDF for PDF processing
# # from typing import Dict, List, Tuple
# # import json

# # class BatchPDFVectorizerOpenAI:
# #     def __init__(self, folder_path: str, api_key: str, model_name: str = "text-embedding-3-small"):
# #         """
# #         Initialize the BatchPDFVectorizer with a folder path and OpenAI API key.
        
# #         Args:
# #             folder_path (str): Path to the folder containing PDF files.
# #             api_key (str): OpenAI API key.
# #             model_name (str): Name of the OpenAI embedding model (default: text-embedding-3-small).
# #         """
# #         if not os.path.exists(folder_path):
# #             raise FileNotFoundError(f"Folder not found at: {folder_path}")
        
# #         self.folder_path = folder_path
# #         self.model_name = model_name
# #         self.client = openai.OpenAI(api_key=api_key)
# #         print(f"Initialized OpenAI client with model: {model_name}")

# #     def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
# #         """
# #         Extract text from each page of a PDF file.
        
# #         Args:
# #             pdf_path (str): Path to the PDF file.
            
# #         Returns:
# #             Dict[int, str]: Dictionary mapping page numbers to their text content.
# #         """
# #         page_dict = {}
# #         try:
# #             doc = fitz.open(pdf_path)
# #             for page_num in range(len(doc)):
# #                 page = doc[page_num]
# #                 text = page.get_text()
# #                 if text.strip():  # Only include non-empty pages
# #                     page_dict[page_num + 1] = text.strip()  # Page numbers start at 1
# #             doc.close()
# #             print(f"Extracted {len(page_dict)} pages from {pdf_path}.")
# #         except Exception as e:
# #             print(f"Error extracting text from {pdf_path}: {str(e)}")
        
# #         return page_dict

# #     def vectorize_pages(self, pdf_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
# #         """
# #         Generate vector embeddings for each page using OpenAI's embedding API.
        
# #         Args:
# #             pdf_path (str): Path to the PDF file.
            
# #         Returns:
# #             Tuple[Dict[int, np.ndarray], Dict[int, str]]: Dictionary mapping page numbers to their 
# #             vector embeddings and dictionary mapping page numbers to their text content.
# #         """
# #         page_dict = self.extract_text_from_pdf(pdf_path)
# #         page_embeddings = {}
# #         embedding_dimensions = None

# #         for page_num, text in page_dict.items():
# #             try:
# #                 response = self.client.embeddings.create(
# #                     model=self.model_name,
# #                     input=text
# #                 )
# #                 embedding = np.array(response.data[0].embedding)
                
# #                 if embedding_dimensions is None:
# #                     embedding_dimensions = embedding.shape[0]
                
# #                 page_embeddings[page_num] = embedding
# #                 print(f"Generated embedding for Page {page_num} in {pdf_path}")
# #             except Exception as e:
# #                 print(f"Error generating embedding for Page {page_num} in {pdf_path}: {str(e)}")
# #                 if embedding_dimensions is not None:
# #                     page_embeddings[page_num] = np.zeros(embedding_dimensions)
        
# #         return page_embeddings, page_dict

# #     def process_all_files(self, output_path: str):
# #         """
# #         Process all PDF files in the specified folder, generate embeddings, 
# #         and save them to a single file.
        
# #         Args:
# #             output_path (str): Path to save the combined embeddings file.
# #         """
# #         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
# #         all_embeddings = {}
# #         all_metadata = {}
        
# #         for file_name in os.listdir(self.folder_path):
# #             if file_name.lower().endswith(".pdf"):
# #                 file_path = os.path.join(self.folder_path, file_name)
                
# #                 page_embeddings, page_texts = self.vectorize_pages(file_path)
                
# #                 for page_num, embedding in page_embeddings.items():
# #                     # Create a unique ID for each page: filename_pagenum
# #                     page_id = f"{file_name}_{page_num}"
# #                     all_embeddings[page_id] = embedding
                    
# #                     # Store metadata about this page
# #                     all_metadata[page_id] = {
# #                         "file_name": file_name,
# #                         "page_number": page_num,
# #                         "text": page_texts[page_num],
# #                         "char_count": len(page_texts[page_num])
# #                     }
        
# #         self.save_combined_embeddings(all_embeddings, all_metadata, output_path)
# #         print(f"Processing complete. Combined embeddings saved to {output_path}")

# #     def save_combined_embeddings(self, embeddings: Dict[str, np.ndarray], 
# #                                 metadata: Dict[str, Dict], output_path: str):
# #         """
# #         Save all embeddings and metadata to a single file.
        
# #         Args:
# #             embeddings (Dict[str, np.ndarray]): Dictionary of page IDs to embeddings.
# #             metadata (Dict[str, Dict]): Dictionary of page IDs to metadata.
# #             output_path (str): Path to save the combined embeddings file.
# #         """
# #         # Convert embeddings dict to arrays for efficient storage
# #         page_ids = list(embeddings.keys())
# #         embedding_array = np.array([embeddings[page_id] for page_id in page_ids])
        
# #         # Save embeddings and metadata
# #         np.savez(
# #             output_path,
# #             page_ids=page_ids,
# #             embeddings=embedding_array,
# #             metadata=json.dumps(metadata)  # Convert metadata to JSON string
# #         )
        
# #         print(f"Saved {len(page_ids)} page embeddings to {output_path}")

# # def main():
# #     FOLDER_PATH = 'D:\\Sam_Project\\knowledge_base_pdfs'  # Update with actual folder path
# #     API_KEY = 'YOUR_OPENAI_API_KEY'  # Replace with OpenAI API key
# #     OUTPUT_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings.npz'  # Single output file
    
# #     try:
# #         vectorizer = BatchPDFVectorizerOpenAI(FOLDER_PATH, API_KEY)
# #         vectorizer.process_all_files(OUTPUT_PATH)
# #     except Exception as e:
# #         print(f"Error during vectorization: {str(e)}")

# # if __name__ == "__main__":
# #     main()




# import os
# import fitz  # PyMuPDF for PDF processing
# import numpy as np
# import json
# from typing import Dict, List, Tuple
# import google.generativeai as genai

# class BatchPDFVectorizerGemini:
#     def __init__(self, folder_path: str, api_key: str):
#         """
#         Initialize the BatchPDFVectorizer with a folder path and Google Gemini API key.
        
#         Args:
#             folder_path (str): Path to the folder containing PDF files.
#             api_key (str): Google Gemini API key.
#         """
#         if not os.path.exists(folder_path):
#             raise FileNotFoundError(f"Folder not found at: {folder_path}")
        
#         self.folder_path = folder_path
        
#         # Configure Google Gemini API
#         genai.configure(api_key=api_key)
        
#         # Correct model name format for embeddings
#         self.embedding_model = "models/embedding-001"
        
#         print(f"Initialized Google Gemini client with embedding model: {self.embedding_model}")

#     def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
#         """
#         Extract text from each page of a PDF file.
        
#         Args:
#             pdf_path (str): Path to the PDF file.
            
#         Returns:
#             Dict[int, str]: Dictionary mapping page numbers to their text content.
#         """
#         page_dict = {}
#         try:
#             doc = fitz.open(pdf_path)
#             for page_num in range(len(doc)):
#                 page = doc[page_num]
#                 text = page.get_text()
#                 if text.strip():  # Only include non-empty pages
#                     page_dict[page_num + 1] = text.strip()  # Page numbers start at 1
#             doc.close()
#             print(f"Extracted {len(page_dict)} pages from {pdf_path}.")
#         except Exception as e:
#             print(f"Error extracting text from {pdf_path}: {str(e)}")
        
#         return page_dict

#     def vectorize_pages(self, pdf_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
#         """
#         Generate vector embeddings for each page using Google Gemini's embedding API.
        
#         Args:
#             pdf_path (str): Path to the PDF file.
            
#         Returns:
#             Tuple[Dict[int, np.ndarray], Dict[int, str]]: Dictionary mapping page numbers to their 
#             vector embeddings and dictionary mapping page numbers to their text content.
#         """
#         page_dict = self.extract_text_from_pdf(pdf_path)
#         page_embeddings = {}
#         embedding_dimensions = None

#         for page_num, text in page_dict.items():
#             try:
#                 # Handle text length limits (Gemini has a token limit)
#                 truncated_text = text[:8000]  # Approximate limit, adjust if needed
                
#                 # Create embeddings using Gemini API with correct model name format
#                 embedding_result = genai.embed_content(
#                     model=self.embedding_model,
#                     content=truncated_text,
#                     task_type="retrieval_document",  # For document retrieval
#                 )
                
#                 # Convert to numpy array
#                 embedding = np.array(embedding_result["embedding"])
                
#                 if embedding_dimensions is None:
#                     embedding_dimensions = embedding.shape[0]
                
#                 page_embeddings[page_num] = embedding
#                 print(f"Generated embedding for Page {page_num} in {pdf_path}")
#             except Exception as e:
#                 print(f"Error generating embedding for Page {page_num} in {pdf_path}: {str(e)}")
#                 if embedding_dimensions is not None:
#                     page_embeddings[page_num] = np.zeros(embedding_dimensions)
#                 else:
#                     # If no successful embeddings yet, create a placeholder
#                     # Default dimension for Gemini embeddings is 768
#                     page_embeddings[page_num] = np.zeros(768)
        
#         return page_embeddings, page_dict

#     def process_all_files(self, output_path: str):
#         """
#         Process all PDF files in the specified folder, generate embeddings, 
#         and save them to a single file.
        
#         Args:
#             output_path (str): Path to save the combined embeddings file.
#         """
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         all_embeddings = {}
#         all_metadata = {}
        
#         for file_name in os.listdir(self.folder_path):
#             if file_name.lower().endswith(".pdf"):
#                 file_path = os.path.join(self.folder_path, file_name)
                
#                 page_embeddings, page_texts = self.vectorize_pages(file_path)
                
#                 for page_num, embedding in page_embeddings.items():
#                     # Create a unique ID for each page: filename_pagenum
#                     page_id = f"{file_name}_{page_num}"
#                     all_embeddings[page_id] = embedding
                    
#                     # Store metadata about this page
#                     all_metadata[page_id] = {
#                         "file_name": file_name,
#                         "page_number": page_num,
#                         "text": page_texts[page_num][:1000],  # Store first 1000 chars only to save space
#                         "char_count": len(page_texts[page_num])
#                     }
        
#         self.save_combined_embeddings(all_embeddings, all_metadata, output_path)
#         print(f"Processing complete. Combined embeddings saved to {output_path}")

#     def save_combined_embeddings(self, embeddings: Dict[str, np.ndarray], 
#                                 metadata: Dict[str, Dict], output_path: str):
#         """
#         Save all embeddings and metadata to a single file.
        
#         Args:
#             embeddings (Dict[str, np.ndarray]): Dictionary of page IDs to embeddings.
#             metadata (Dict[str, Dict]): Dictionary of page IDs to metadata.
#             output_path (str): Path to save the combined embeddings file.
#         """
#         # Convert embeddings dict to arrays for efficient storage
#         page_ids = list(embeddings.keys())
#         embedding_array = np.array([embeddings[page_id] for page_id in page_ids])
        
#         # Save embeddings and metadata
#         np.savez(
#             output_path,
#             page_ids=page_ids,
#             embeddings=embedding_array,
#             metadata=json.dumps(metadata)  # Convert metadata to JSON string
#         )
        
#         print(f"Saved {len(page_ids)} page embeddings to {output_path}")


# def main():
#     FOLDER_PATH = 'D:\\Sam_Project\\knowledge_base'  # Update with actual folder path
#     API_KEY = 'Remove'  # Replace with Google AI API key
#     OUTPUT_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_gemini.npz'  # Single output file


#     try:
#         vectorizer = BatchPDFVectorizerGemini(FOLDER_PATH, API_KEY)
#         vectorizer.process_all_files(OUTPUT_PATH)
#     except Exception as e:
#         print(f"Error during vectorization: {str(e)}")

# if __name__ == "__main__":
#     main()





import os
import fitz  # PyMuPDF for PDF processing
import numpy as np
import json
from typing import Dict, Tuple
import google.generativeai as genai

class BatchPDFVectorizerGemini:
    def __init__(self, folder_path: str, api_key: str):
        """
        Initialize the BatchPDFVectorizer with a folder path and Google Gemini API key.
        
        Args:
            folder_path (str): Path to the folder containing PDF files.
            api_key (str): Google Gemini API key.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")
        
        self.folder_path = folder_path
        genai.configure(api_key=api_key)
        self.embedding_model = "models/embedding-001"
        print(f"Initialized Google Gemini client with embedding model: {self.embedding_model}")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text from each page of a PDF file.
        """
        page_dict = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    page_dict[page_num + 1] = text.strip()
            doc.close()
            print(f"Extracted {len(page_dict)} pages from {pdf_path}.")
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
        return page_dict

    def vectorize_pages(self, pdf_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """
        Generate vector embeddings for each page using Google Gemini's embedding API.
        """
        page_dict = self.extract_text_from_pdf(pdf_path)
        page_embeddings = {}
        embedding_dimensions = 768  # Gemini embedding-001 default dimension

        for page_num, text in page_dict.items():
            try:
                truncated_text = text[:8000]  # Approximate limit
                embedding_result = genai.embed_content(
                    model=self.embedding_model,
                    content=truncated_text,
                    task_type="retrieval_document",
                )
                embedding = np.array(embedding_result["embedding"])
                if embedding.shape[0] != embedding_dimensions:
                    raise ValueError(f"Unexpected embedding dimension: {embedding.shape[0]}")
                page_embeddings[page_num] = embedding
                print(f"Generated embedding for Page {page_num} in {pdf_path}, shape: {embedding.shape}")
            except Exception as e:
                print(f"Error generating embedding for Page {page_num} in {pdf_path}: {str(e)}")
                page_embeddings[page_num] = np.zeros(embedding_dimensions)
        
        return page_embeddings, page_dict

    def process_all_files(self, output_path: str):
        """
        Process all PDF files in the folder and save embeddings.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_embeddings = {}
        all_metadata = {}
        
        for file_name in os.listdir(self.folder_path):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(self.folder_path, file_name)
                page_embeddings, page_texts = self.vectorize_pages(file_path)
                
                for page_num, embedding in page_embeddings.items():
                    page_id = f"{file_name}_{page_num}"
                    all_embeddings[page_id] = embedding
                    all_metadata[page_id] = {
                        "file_name": file_name,
                        "page_number": page_num,
                        "text": page_texts[page_num][:1000],
                        "char_count": len(page_texts[page_num])
                    }
        
        self.save_combined_embeddings(all_embeddings, all_metadata, output_path)
        print(f"Processing complete. Combined embeddings saved to {output_path}")

    def save_combined_embeddings(self, embeddings: Dict[str, np.ndarray], 
                                metadata: Dict[str, Dict], output_path: str):
        """
        Save all embeddings and metadata to a single file.
        """
        page_ids = list(embeddings.keys())
        embedding_array = np.stack([embeddings[page_id] for page_id in page_ids], axis=0)
        print(f"Embedding array shape before saving: {embedding_array.shape}")
        
        if embedding_array.ndim != 2 or embedding_array.shape[1] != 768:
            raise ValueError(f"Invalid embedding array shape: {embedding_array.shape}")
        
        np.savez(
            output_path,
            page_ids=page_ids,
            embeddings=embedding_array,
            metadata=json.dumps(metadata)
        )
        print(f"Saved {len(page_ids)} page embeddings to {output_path}")

def main():
    FOLDER_PATH = 'D:\\Sam_Project\\knowledge_base'
    API_KEY = 'Remove'  # Replace with your Google AI API key
    OUTPUT_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_gemini.npz'
    
    try:
        vectorizer = BatchPDFVectorizerGemini(FOLDER_PATH, API_KEY)
        vectorizer.process_all_files(OUTPUT_PATH)
    except Exception as e:
        print(f"Error during vectorization: {str(e)}")

if __name__ == "__main__":
    main()