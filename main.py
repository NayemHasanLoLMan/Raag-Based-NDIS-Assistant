import numpy as np
import json
import os
import google.generativeai as genai
from scipy.spatial.distance import cosine

class EnhancedKnowledgeBaseChatbot:
    def __init__(self, embeddings_path, api_key, top_k=3):
        """
        Initialize the Enhanced Knowledge Base Chatbot with both general knowledge
        and specialized knowledge base capabilities.
        """
        self.load_embeddings(embeddings_path)
        genai.configure(api_key=api_key)
        self.embedding_model = "models/embedding-001"
        self.generation_model = "models/gemini-2.0-flash"  # Using the more capable model
        self.top_k = top_k
        self.conversation_history = []
        print(f"Enhanced Knowledge Base Chatbot initialized with {len(self.page_ids)} pages")

    def load_embeddings(self, embeddings_path):
        """
        Load embeddings and metadata from npz file.
        """
        try:
            data = np.load(embeddings_path, allow_pickle=True)
            self.page_ids = data['page_ids'].tolist()
            self.embeddings = data['embeddings']
            if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 768:
                raise ValueError(f"Embeddings must be 2D with 768 dimensions, got shape {self.embeddings.shape}")
            
            # Handle metadata loading
            metadata_raw = data['metadata']
            if isinstance(metadata_raw, np.ndarray) and metadata_raw.size > 0:
                metadata_str = metadata_raw.item() if metadata_raw.size == 1 else metadata_raw[0]
            else:
                metadata_str = metadata_raw
            
            if not isinstance(metadata_str, str):
                raise ValueError(f"Metadata must be a string, got {type(metadata_str)}")
            self.metadata = json.loads(metadata_str)
            
            print(f"Loaded {len(self.page_ids)} embeddings from {embeddings_path}, shape: {self.embeddings.shape}")
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            raise

    def get_query_embedding(self, query):
        """
        Get embedding for a query using Gemini API.
        """
        try:
            embedding_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query",
            )
            embedding = np.array(embedding_result["embedding"])
            if embedding.shape != (768,):
                raise ValueError(f"Query embedding must be 768-dimensional, got {embedding.shape}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding for query: {str(e)}")
            raise

    def get_most_relevant_pages(self, query_embedding):
        """
        Find the most relevant pages for a given query embedding.
        """
        similarities = []
        for i, page_id in enumerate(self.page_ids):
            similarity = 1 - cosine(query_embedding, self.embeddings[i])
            similarities.append((page_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = []
        for page_id, score in similarities[:self.top_k]:
            page_info = self.metadata[page_id].copy()
            page_info['similarity_score'] = score
            page_info['page_id'] = page_id
            top_results.append(page_info)
        return top_results

    def format_sources_for_human(self, relevant_pages):
        """
        Format source information for display to the user.
        """
        sources = []
        for i, page in enumerate(relevant_pages):
            file_name = page['file_name']
            page_num = page['page_number']
            similarity = page['similarity_score'] * 100
            text_excerpt = page['text'][:200] + "..." if len(page['text']) > 200 else page['text']
            source = f"Source {i+1}: {file_name}, Page {page_num} (Relevance: {similarity:.1f}%)\nExcerpt: \"{text_excerpt}\""
            sources.append(source)
        return "\n".join(sources)
    
    def format_sources_for_model(self, relevant_pages):
        """
        Format source information for the model to use.
        """
        context = ""
        for i, page in enumerate(relevant_pages):
            context += f"Source {i+1}: {page['file_name']}, Page {page['page_number']}:\n{page['text']}\n\n"
        return context

    def is_ndis_related(self, query):
        """
        Determine if a query is related to NDIS or disability services.
        """
        ndis_keywords = [
            "ndis", "national disability", "disability insurance", "disability service", 
            "provider", "participant", "support", "plan", "funding", "registered provider"
        ]
        
        query_lower = query.lower()
        for keyword in ndis_keywords:
            if keyword in query_lower:
                return True
        return False

    def generate_answer(self, query, relevant_pages):
        """
        Generate a natural answer based on the query type.
        """
        # Check if the query is NDIS-related
        is_ndis_query = self.is_ndis_related(query)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        if is_ndis_query and relevant_pages and relevant_pages[0]['similarity_score'] > 0.6:
            # For NDIS-related queries with relevant documents
            context = self.format_sources_for_model(relevant_pages)
            
            prompt = f"""
            You are a helpful assistant for the National Disability Insurance Scheme (NDIS).
            
            The user has asked: "{query}"
            
            Use the following information to provide a natural, conversational response:
            
            {context}
            
            Your response should:
            1. Be conversational and friendly, as if you're having a natural conversation
            2. Directly address the question in a helpful way
            3. Include relevant information from the sources provided
            4. Mention where information comes from (e.g., "According to [document name], page [page number]...")
            5. Be accurate and provide complete information where available
            6. If the information doesn't fully answer the question, acknowledge that and provide what is available
            """
            
            sources_available = True
        else:
            # For general queries or when no relevant NDIS documents found
            prompt = f"""
            You are a helpful, knowledgeable assistant engaging in a natural conversation.
            
            The user has asked: "{query}"
            
            Please provide a natural, conversational response based on your general knowledge.
            If this relates to NDIS (National Disability Insurance Scheme) but you don't have specific
            information, mention that NDIS-specific details would require consulting official documentation.
            """
            
            sources_available = False
        
        try:
            model = genai.GenerativeModel(model_name=self.generation_model)
            response = model.generate_content(prompt)
            self.conversation_history.append({"role": "assistant", "content": response.text})
            return response.text, sources_available
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            error_msg = "I'm sorry, I encountered an error while generating a response. Could you please try asking again?"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg, False

    def answer_query(self, query):
        """
        Process a user query and return a natural language answer with sources if relevant.
        """
        try:
            query_embedding = self.get_query_embedding(query)
            relevant_pages = self.get_most_relevant_pages(query_embedding)
            
            answer, sources_available = self.generate_answer(query, relevant_pages)
            
            if sources_available:
                sources = self.format_sources_for_human(relevant_pages)
                return {"answer": answer, "sources": sources, "has_sources": True}
            else:
                return {"answer": answer, "sources": None, "has_sources": False}
        except Exception as e:
            print(f"Error answering query: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered a technical issue while processing your question. Could you please try again or rephrase your question?",
                "sources": None,
                "has_sources": False
            }

def main():
    EMBEDDINGS_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_gemini.npz'
    API_KEY = 'Remove'  # Replace with your Gemini API key
    
    chatbot = EnhancedKnowledgeBaseChatbot(EMBEDDINGS_PATH, API_KEY)
    
    print("Enhanced Knowledge Base Chatbot")
    print("Ask me anything - for NDIS questions, I'll consult official documentation")
    print("-" * 60)
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye! Have a great day!")
            break
            
        result = chatbot.answer_query(user_query)
        
        print(f"\nAssistant: {result['answer']}")
        
        if result['has_sources']:
            print("\n--- Sources Referenced ---")
            print(result["sources"])
        
        print("-" * 60)

if __name__ == "__main__":
    main()