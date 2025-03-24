


# import numpy as np
# import json
# import os
# import google.generativeai as genai
# from scipy.spatial.distance import cosine

# class NDISAssistantBot:
#     def __init__(self, embeddings_path, api_key, budget_file_path, top_k=5, similarity_threshold=0.6):
#         """
#         Initialize the NDIS Assistant Bot with knowledge base and budget information.
#         """
#         # Load necessary data
#         self.load_embeddings(embeddings_path)
#         self.load_budget(budget_file_path)
        
#         # Configure API
#         genai.configure(api_key=api_key)
#         self.embedding_model = "models/embedding-001"
#         self.generation_model = "models/gemini-2.0-flash"
        
#         # Search parameters
#         self.top_k = top_k
#         self.similarity_threshold = similarity_threshold
#         self.conversation_history = []
        
#         print(f"NDIS Assistant initialized successfully")

#     def load_embeddings(self, embeddings_path):
#         """
#         Load knowledge base embeddings and metadata.
#         """
#         try:
#             data = np.load(embeddings_path, allow_pickle=True)
#             self.page_ids = data['page_ids'].tolist()
#             self.embeddings = data['embeddings']
            
#             # Handle metadata
#             metadata_raw = data['metadata']
#             if isinstance(metadata_raw, np.ndarray) and metadata_raw.size > 0:
#                 metadata_str = metadata_raw.item() if metadata_raw.size == 1 else metadata_raw[0]
#             else:
#                 metadata_str = metadata_raw
            
#             self.metadata = json.loads(metadata_str)
#             print(f"Loaded NDIS knowledge base with {len(self.page_ids)} documents")
#         except Exception as e:
#             print(f"Error loading knowledge base: {str(e)}")
#             raise

#     def load_budget(self, budget_file_path):
#         """
#         Load user's NDIS budget information.
#         """
#         try:
#             if os.path.exists(budget_file_path):
#                 with open(budget_file_path, 'r') as file:
#                     self.budget_info = file.read()
#                 print("Budget information loaded successfully")
#             else:
#                 self.budget_info = "No budget information available."
#                 print(f"Warning: Budget file not found at {budget_file_path}")
#         except Exception as e:
#             print(f"Error loading budget: {str(e)}")
#             self.budget_info = "Error loading budget information."

#     def get_embedding(self, query):
#         """
#         Get embedding for text using Gemini API.
#         """
#         try:
#             embedding_result = genai.embed_content(
#                 model=self.embedding_model,
#                 content=query,
#                 task_type="retrieval_query",
#             )
#             return np.array(embedding_result["embedding"])
#         except Exception as e:
#             print(f"Error generating embedding: {str(e)}")
#             raise

#     def find_relevant_content(self, query):
#         """
#         Find relevant content from knowledge base for a query.
#         """
#         try:
#             # Get query embedding
#             query_embedding = self.get_embedding(query)
            
#             # Calculate similarities
#             similarities = []
#             for i, page_id in enumerate(self.page_ids):
#                 similarity = 1 - cosine(query_embedding, self.embeddings[i])
#                 similarities.append((page_id, similarity))
            
#             # Sort by similarity
#             similarities.sort(key=lambda x: x[1], reverse=True)
            
#             # Get top results above threshold
#             relevant_pages = []
#             for page_id, score in similarities[:self.top_k]:
#                 if score >= self.similarity_threshold:
#                     page_info = self.metadata[page_id].copy()
#                     page_info['score'] = score
#                     relevant_pages.append(page_info)
            
#             # Format content for model
#             if relevant_pages:
#                 context = "NDIS INFORMATION:\n\n"
#                 for i, page in enumerate(relevant_pages):
#                     context += f"{page['text']}\n\n"
#                 return context
#             else:
#                 return ""
#         except Exception as e:
#             print(f"Error finding relevant content: {str(e)}")
#             return ""

#     def answer_question(self, query):
#         """
#         Answer a question about NDIS naturally using knowledge base and budget information.
#         """
#         try:
#             # Save query in conversation history
#             self.conversation_history.append({"role": "user", "content": query})
            
#             # Get relevant information
#             relevant_content = self.find_relevant_content(query)
            
#             # Create prompt for model
#             prompt = f"""
#             You are a helpful, natural-sounding NDIS assistant. You answer questions about the National Disability Insurance Scheme in Australia in a friendly, conversational way.
            
#             Guidelines:
#             - Sound natural and human-like in your responses
#             - Be helpful, supportive and empathetic
#             - Don't mention where your information comes from
#             - Don't say "based on the information provided" or similar phrases
#             - Don't format responses with numbering unless it's a step-by-step process
#             - Ask follow-up questions when appropriate
            
#             USER QUESTION: {query}
            
#             {relevant_content}
            
#             BUDGET INFORMATION:
#             {self.budget_info}
            
#             Please answer the question in a natural, helpful way. If the question relates to budget matters, use the budget information as appropriate without mentioning the source. If the information isn't available, provide general NDIS guidance or suggest contacting NDIS directly.
#             """
            
#             # Generate response
#             model = genai.GenerativeModel(model_name=self.generation_model)
#             response = model.generate_content(prompt)
#             self.conversation_history.append({"role": "assistant", "content": response.text})
            
#             return response.text
#         except Exception as e:
#             print(f"Error generating answer: {str(e)}")
#             return "I'm sorry, I'm having trouble answering that right now. Could you try asking your question again, maybe in a different way?"

# def main():
#     EMBEDDINGS_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_gemini.npz'
#     USER_BUDGET_INFORMATION = 'D:\\Sam_Project\\budget.txt'
#     API_KEY = 'Remove'  # Replace with your Gemini API key
    
#     chatbot = NDISAssistantBot(
#         embeddings_path=EMBEDDINGS_PATH,
#         api_key=API_KEY,
#         budget_file_path=USER_BUDGET_INFORMATION
#     )
    
#     print("NDIS Assistant")
#     print("Hi there! I'm here to help with any NDIS questions you might have.")
#     print("-" * 60)
    
#     while True:
#         user_query = input("\nYou: ")
#         if user_query.lower() in ['exit', 'quit', 'bye']:
#             print("Thanks for chatting! Take care and have a great day.")
#             break
            
#         answer = chatbot.answer_question(user_query)
#         print(f"\nNDIS Assistant: {answer}")
#         print("-" * 60)

# if __name__ == "__main__":
#     main()

######################## Version 2 ##############################
  
# import numpy as np
# import json
# import os
# import google.generativeai as genai
# from scipy.spatial.distance import cosine

# class NDISAssistantBot:
#     def __init__(self, embeddings_path, api_key, budget_file_path, top_k=5, similarity_threshold=0.6):
#         """
#         Initialize the NDIS Assistant Bot with knowledge base and budget information.
#         """
#         # Load necessary data
#         self.load_embeddings(embeddings_path)
#         self.load_budget(budget_file_path)
        
#         # Configure API
#         genai.configure(api_key=api_key)
#         self.embedding_model = "models/embedding-001"
#         self.generation_model = "models/gemini-2.0-flash"
        
#         # Search parameters
#         self.top_k = top_k
#         self.similarity_threshold = similarity_threshold
#         self.conversation_history = []
#         self.last_sources = []  # Store the sources used for the last response
        
#         print(f"NDIS Assistant initialized successfully")

#     def load_embeddings(self, embeddings_path):
#         """
#         Load knowledge base embeddings and metadata.
#         """
#         try:
#             data = np.load(embeddings_path, allow_pickle=True)
#             self.page_ids = data['page_ids'].tolist()
#             self.embeddings = data['embeddings']
            
#             # Handle metadata
#             metadata_raw = data['metadata']
#             if isinstance(metadata_raw, np.ndarray) and metadata_raw.size > 0:
#                 metadata_str = metadata_raw.item() if metadata_raw.size == 1 else metadata_raw[0]
#             else:
#                 metadata_str = metadata_raw
            
#             self.metadata = json.loads(metadata_str)
#             print(f"Loaded NDIS knowledge base with {len(self.page_ids)} documents")
#         except Exception as e:
#             print(f"Error loading knowledge base: {str(e)}")
#             raise

#     def load_budget(self, budget_file_path):
#         """
#         Load user's NDIS budget information.
#         """
#         try:
#             if os.path.exists(budget_file_path):
#                 with open(budget_file_path, 'r') as file:
#                     self.budget_info = file.read()
#                 print("Budget information loaded successfully")
#             else:
#                 self.budget_info = "No budget information available."
#                 print(f"Warning: Budget file not found at {budget_file_path}")
#         except Exception as e:
#             print(f"Error loading budget: {str(e)}")
#             self.budget_info = "Error loading budget information."

#     def get_embedding(self, query):
#         """
#         Get embedding for text using Gemini API.
#         """
#         try:
#             embedding_result = genai.embed_content(
#                 model=self.embedding_model,
#                 content=query,
#                 task_type="retrieval_query",
#             )
#             return np.array(embedding_result["embedding"])
#         except Exception as e:
#             print(f"Error generating embedding: {str(e)}")
#             raise

#     def find_relevant_content(self, query):
#         """
#         Find relevant content from knowledge base for a query.
#         Returns the relevant content and source information.
#         """
#         try:
#             # Get query embedding
#             query_embedding = self.get_embedding(query)
            
#             # Calculate similarities
#             similarities = []
#             for i, page_id in enumerate(self.page_ids):
#                 similarity = 1 - cosine(query_embedding, self.embeddings[i])
#                 similarities.append((page_id, similarity))
            
#             # Sort by similarity
#             similarities.sort(key=lambda x: x[1], reverse=True)
            
#             # Get top results above threshold
#             relevant_pages = []
#             self.last_sources = []  # Reset sources for this query
            
#             for page_id, score in similarities[:self.top_k]:
#                 if score >= self.similarity_threshold:
#                     page_info = self.metadata[page_id].copy()
#                     page_info['score'] = score
#                     relevant_pages.append(page_info)
                    
#                     # Save source information using the correct keys from your metadata
#                     source_info = {
#                         'document': page_info.get('file_name', 'Unknown document'),
#                         'page': page_info.get('page_number', 'Unknown page'),
#                         'text': page_info.get('text', ''),
#                         'score': round(score, 3)
#                     }
#                     self.last_sources.append(source_info)
            
#             # Format content for model
#             if relevant_pages:
#                 context = "NDIS INFORMATION:\n\n"
#                 for i, page in enumerate(relevant_pages):
#                     context += f"{page['text']}\n\n"
#                 return context
#             else:
#                 return ""
#         except Exception as e:
#             print(f"Error finding relevant content: {str(e)}")
#             self.last_sources = []
#             return ""

#     def answer_question(self, query):
#         """
#         Answer a question about NDIS naturally using knowledge base and budget information.
#         Returns the answer text.
#         """
#         try:
#             # Save query in conversation history
#             self.conversation_history.append({"role": "user", "content": query})
            
#             # Get relevant information
#             relevant_content = self.find_relevant_content(query)
            
#             # Create prompt for model
#             prompt = f"""
#             You are a helpful, natural-sounding NDIS assistant. You answer questions about the National Disability Insurance Scheme in Australia in a friendly, conversational way.
            
#             Guidelines:
#             - Sound natural and human-like in your responses
#             - Be helpful, supportive and empathetic
#             - Don't mention where your information comes from
#             - Don't say "based on the information provided" or similar phrases
#             - Don't format responses with numbering unless it's a step-by-step process
#             - Ask follow-up questions when appropriate
            
#             USER QUESTION: {query}
            
#             {relevant_content}
            
#             BUDGET INFORMATION:
#             {self.budget_info}
            
#             Please answer the question in a natural, helpful way. If the question relates to budget matters, use the budget information as appropriate without mentioning the source. If the information isn't available, provide general NDIS guidance or suggest contacting NDIS directly.
#             """
            
#             # Generate response
#             model = genai.GenerativeModel(model_name=self.generation_model)
#             response = model.generate_content(prompt)
#             self.conversation_history.append({"role": "assistant", "content": response.text})
            
#             return response.text
#         except Exception as e:
#             print(f"Error generating answer: {str(e)}")
#             return "I'm sorry, I'm having trouble answering that right now. Could you try asking your question again, maybe in a different way?"

#     def print_sources(self):
#         """
#         Print the sources of information used for the last response.
#         """
#         if not self.last_sources:
#             print("\nSource Information: No specific sources were used for this response.")
#             return
        
#         print("\nSource Information:")
#         print("-" * 60)
#         for i, source in enumerate(self.last_sources, 1):
#             print(f"Source {i}:")
#             print(f"  Document: {source['document']}")
#             print(f"  Page: {source['page']}")
#             print(f"  Relevance Score: {source['score']}")
#             print(f"  Context: {source['text'][:150]}...")  # Print first 150 chars of the text
#             print("-" * 60)

# def main():
#     EMBEDDINGS_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_gemini.npz'
#     USER_BUDGET_INFORMATION = 'D:\\Sam_Project\\budget.txt'
#     API_KEY = 'Remove'  # Replace with your Gemini API key
    
#     chatbot = NDISAssistantBot(
#         embeddings_path=EMBEDDINGS_PATH,
#         api_key=API_KEY,
#         budget_file_path=USER_BUDGET_INFORMATION
#     )
    
#     print("NDIS Assistant")
#     print("Hi there! I'm here to help with any NDIS questions you might have.")
#     print("-" * 60)
    
#     while True:
#         user_query = input("\nYou: ")
#         if user_query.lower() in ['exit', 'quit', 'bye']:
#             print("Thanks for chatting! Take care and have a great day.")
#             break
            
#         answer = chatbot.answer_question(user_query)
#         print(f"\nNDIS Assistant: {answer}")
        
#         # Print source information separately
#         chatbot.print_sources()
        
#         print("-" * 60)

# if __name__ == "__main__":
#     main()



######################### version 3 ##############################
import numpy as np
import json
import os
import google.generativeai as genai
from scipy.spatial.distance import cosine

class NDISAssistantBot:
    def __init__(self, embeddings_path, api_key, budget_file_path, top_k=5, similarity_threshold=0.6)  :
        """Initialize the NDIS Assistant Bot."""
        self.load_embeddings(embeddings_path)
        self.load_budget(budget_file_path)
        genai.configure(api_key=api_key)
        self.embedding_model = "models/embedding-001"
        self.generation_model = "models/gemini-2.0-flash"
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.conversation_history = []
        self.last_sources = []
        print("NDIS Assistant initialized successfully")

    def load_embeddings(self, embeddings_path):
        """Load knowledge base embeddings and metadata."""
        try:
            data = np.load(embeddings_path, allow_pickle=True)
            self.page_ids = data['page_ids'].tolist()
            self.embeddings = data['embeddings']
            metadata_raw = data['metadata']
            self.metadata = json.loads(metadata_raw.item() if metadata_raw.size == 1 else metadata_raw[0])
            print(f"Loaded NDIS knowledge base with {len(self.page_ids)} documents")
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            raise

    def load_budget(self, budget_file_path):
        """Load user's NDIS budget information."""
        try:
            if os.path.exists(budget_file_path):
                with open(budget_file_path, 'r') as file:
                    self.budget_info = file.read()
                print("Budget information loaded successfully")
            else:
                self.budget_info = "No budget information available."
                print(f"Warning: Budget file not found at {budget_file_path}")
        except Exception as e:
            print(f"Error loading budget: {str(e)}")
            self.budget_info = "Error loading budget information."

    def get_embedding(self, query):
        """Get embedding for text using Gemini API."""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query",
            )
            return np.array(result["embedding"])
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def find_relevant_content(self, query):
        """Find relevant content from knowledge base."""
        try:
            query_embedding = self.get_embedding(query)
            similarities = [
                (page_id, 1 - cosine(query_embedding, self.embeddings[i]))
                for i, page_id in enumerate(self.page_ids)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            relevant_pages = []
            self.last_sources = []
            
            for page_id, score in similarities[:self.top_k]:
                if score >= self.similarity_threshold:
                    page_info = self.metadata[page_id].copy()
                    page_info['score'] = score
                    relevant_pages.append(page_info)
                    self.last_sources.append({
                        'document': page_info.get('file_name', 'Unknown'),
                        'page': page_info.get('page_number', 'Unknown'),
                        'text': page_info.get('text', ''),
                        'score': round(score, 3)
                    })
            
            if relevant_pages:
                context = "RELEVANT KNOWLEDGE BASE CONTEXT:\n"
                for i, page in enumerate(relevant_pages):
                    context += f"Chunk {i+1} (Score: {page['score']:.3f}):\n{page['text']}\n\n"
                return context
            return "No relevant context found."
        except Exception as e:
            print(f"Error finding content: {str(e)}")
            self.last_sources = []
            return "Error retrieving context."

    def answer_question(self, query):
        """Generate a concise, context-aware answer."""
        try:
            self.conversation_history.append({"role": "user", "content": query})
            relevant_content = self.find_relevant_content(query)
            
            # Build conversation context
            history_context = "CONVERSATION HISTORY:\n"
            for entry in self.conversation_history[:-1]:  # Exclude current query
                history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"
            
            prompt = f"""
            ### Role Summary
            You are an AI chatbot designed to assist NDIS participants in Australia. Your primary function is to provide excellent, friendly, and efficient replies, listening attentively to understand user needs and offering helpful solutions or resource directions. If a question is unclear, ask clarifying questions. End responses positively.

            ### Constraints
            1. Never mention access to training data.
            2. Stay focused on NDIS topics; politely redirect unrelated queries.
            3. Rely on provided budget info, knowledge base, and general NDIS knowledge; if data is missing, use fallback or link to ndis.gov.au.
            4. Do not answer non-NDIS-related questions or perform unrelated tasks.

            ### Instructions
            - Chat naturally, like a friendly back-and-forth conversation (e.g., Grok, GPT, Gemini).
            - Provide precise, to-the-point answers; only give detailed explanations if explicitly needed.
            - Use budget info from BUDGET INFORMATION only when directly relevant to the query—don’t mention it otherwise.
            - Include URLs only if tied to source info from the knowledge base and explicitly available (e.g., ndis.gov.au links).
            - Format responses cleanly with dot points or numbered steps where it fits naturally.
            - Keep answers concise, accurate, and friendly, focusing on NDIS policies, funding, and plan management.
            - Ask follow-up questions if needed to clarify user intent.
            - If info is unavailable, provide brief NDIS guidance and a relevant link (e.g., ndis.gov.au/providers).
            - For condition-specific queries, respond gently and accurately.

            {history_context}

            CURRENT QUESTION: {query}

            {relevant_content}

            BUDGET INFORMATION:
            {self.budget_info}

            Answer directly with practical, natural advice. Stay in character and keep it light where possible!
            """
            
            model = genai.GenerativeModel(model_name=self.generation_model)
            response = model.generate_content(prompt)
            answer = response.text.strip()
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Sorry, I can’t quite get that—mind rephrasing it?"

    def print_sources(self):
        """Print sources used for the last response."""
        if not self.last_sources:
            print("\nSources: None used.")
            return
        print("\nSources:")
        for i, source in enumerate(self.last_sources[:3], 1):
            print(f"{i}. {source['document']} (Page {source['page']}) ")

def main():
    EMBEDDINGS_PATH = 'D:\\Sam_Project\\knowledge_base_embeddings_gemini.npz'
    USER_BUDGET_INFORMATION = 'D:\\Sam_Project\\budget.txt'
    API_KEY = 'Remove'  # Replace with your Gemini API key
    
    chatbot = NDISAssistantBot(
        embeddings_path=EMBEDDINGS_PATH,
        api_key=API_KEY,
        budget_file_path=USER_BUDGET_INFORMATION
    )
    
    print("NDIS Assistant: Hi! Ready to help with your NDIS queries.")
    print("-" * 50)
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("NDIS Assistant: Bye! Chat anytime you need.")
            break
            
        answer = chatbot.answer_question(user_query)
        print(f"\nNDIS Assistant: {answer}")
        chatbot.print_sources()  # Remove slicing as it's not implemented in the method
        print("-" * 50)

if __name__ == "__main__":
    main()