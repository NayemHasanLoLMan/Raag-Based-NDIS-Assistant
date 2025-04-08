
import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import openai
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ndis_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NDISAssistant")

# Load environment variables
load_dotenv()

class NDISAssistantBotOpenAI:
    def __init__(
        self, 
        embeddings_path: str, 
        budget_info: str,
        conversation_history: str = None, 
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4",
        max_history: int = 10
    ):
        """
        Initialize the NDIS Assistant Bot using OpenAI.
        
        Args:
            embeddings_path: Path to pre-generated embeddings file
            budget_file_path: Path to user's NDIS budget information
            top_k: Number of most relevant documents to retrieve
            similarity_threshold: Minimum similarity score to consider a document relevant
            embedding_model: OpenAI embedding model to use
            chat_model: OpenAI chat model to use
            max_history: Maximum number of conversation turns to retain
        """
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.conversation_history = []
        self.last_sources = []
        self.max_history = max_history
        
        # Statistics tracking
        self.query_count = 0
        self.start_time = datetime.now()

        # Load data
        self._load_embeddings(embeddings_path)
        self._load_budget(budget_info)
        self._load_api_key()

        # Parse conversation history if provided
        if conversation_history:
            self._parse_conversation_history(conversation_history)
        
        logger.info(f"NDIS Assistant Bot initialized with {len(self.page_ids)} documents")



    def _load_embeddings(self, embeddings_path: str) -> None:
        """Load pre-generated OpenAI embeddings with improved error handling."""
        try:
            # Verify file exists
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
                
            logger.info(f"Attempting to load embeddings from: {embeddings_path}")
            
            # Load with diagnostic info
            data = np.load(embeddings_path, allow_pickle=True)
            logger.info(f"NPZ file loaded with keys: {data.files}")
            
            # Verify expected keys exist
            required_keys = ['page_ids', 'embeddings', 'metadata']
            missing_keys = [key for key in required_keys if key not in data.files]
            if missing_keys:
                raise KeyError(f"Missing required keys in embeddings file: {missing_keys}")
            
            # Load page IDs with type checking
            self.page_ids = data['page_ids']
            if isinstance(self.page_ids, np.ndarray):
                self.page_ids = self.page_ids.tolist()
            logger.info(f"Loaded {len(self.page_ids)} page IDs")
            
            # Load embeddings with dimension validation
            self.embeddings = data['embeddings']
            if len(self.embeddings.shape) != 2:
                raise ValueError(f"Embeddings have unexpected shape: {self.embeddings.shape}, expected 2D array")
            logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
            
            # Load metadata with robust parsing
            metadata_raw = data['metadata']
            logger.info(f"Raw metadata type: {type(metadata_raw)}, size: {getattr(metadata_raw, 'size', 'N/A')}")
            
            # Handle different metadata formats
            try:
                if isinstance(metadata_raw, np.ndarray) and metadata_raw.size == 1:
                    metadata_str = metadata_raw.item()
                elif isinstance(metadata_raw, np.ndarray) and metadata_raw.size > 0:
                    metadata_str = metadata_raw[0]
                else:
                    metadata_str = str(metadata_raw)
                    
                logger.info(f"Metadata string type: {type(metadata_str)}")
                self.metadata = json.loads(metadata_str)
                
                # Verify metadata matches page IDs
                if not all(page_id in self.metadata for page_id in self.page_ids):
                    missing_ids = [pid for pid in self.page_ids if pid not in self.metadata]
                    logger.warning(f"Some page IDs missing from metadata: {missing_ids[:5]}...")
                    
                logger.info(f"Successfully parsed metadata with {len(self.metadata)} entries")
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON decoding error: {str(je)}")
                logger.error(f"First 100 chars of metadata string: {str(metadata_str)[:100]}...")
                raise
                
            logger.info(f"Successfully loaded NDIS knowledge base with {len(self.page_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise



    def _load_budget(self, budget_file_path: str) -> None:
        """Load user's NDIS budget information."""
        try:
            if os.path.exists(budget_file_path):
                with open(budget_file_path, 'r', encoding='utf-8') as file:
                    self.budget_info = file.read()
                logger.info("Budget information loaded successfully")
            else:
                self.budget_info = "No budget information available."
                logger.warning(f"Budget file not found at {budget_file_path}")
        except Exception as e:
            logger.error(f"Error loading budget: {str(e)}")
            self.budget_info = "Error loading budget information."

    def _load_api_key(self) -> None:
        """Load OpenAI API key from environment variables."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Missing OPENAI_API_KEY environment variable")
            raise EnvironmentError("Missing OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key

    def _get_embedding(self, text: str, retry_attempts: int = 3) -> Optional[np.ndarray]:
        """
        Generate embedding for query using OpenAI with retry logic.
        
        Args:
            text: The text to generate an embedding for
            retry_attempts: Number of retry attempts
            
        Returns:
            Numpy array containing the embedding vector
        """
        for attempt in range(retry_attempts):
            try:
                result = openai.Embedding.create(
                    model=self.embedding_model,
                    input=text
                )
                return np.array(result['data'][0]['embedding'])
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt+1} failed: {str(e)}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {retry_attempts} attempts")
                    raise

    def _parse_conversation_history(self, history: str) -> None:
        """Parse conversation history string into internal format."""
        try:
            lines = history.strip().split('\n')
            for line in lines:
                if line.startswith("User: "):
                    self.conversation_history.append({"role": "user", "content": line[6:]})
                elif line.startswith("Assistant: "):
                    self.conversation_history.append({"role": "assistant", "content": line[11:]})
            logger.info(f"Parsed {len(self.conversation_history)} history entries")
        except Exception as e:
            logger.error(f"Error parsing conversation history: {str(e)}")
            self.conversation_history = []
    
    def _truncate_history(self) -> None:
        """Truncate conversation history to maximum length."""
        if len(self.conversation_history) > self.max_history * 2:
            # Keep the most recent conversation turns
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
            logger.info(f"Truncated conversation history to {len(self.conversation_history)} messages")

    def find_relevant_content(self, query: str) -> str:
        """
        Find most relevant context chunks from the knowledge base.
        
        Args:
            query: User's question
            
        Returns:
            String containing the most relevant content from the knowledge base
        """
        try:
            # Generate embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Calculate similarity with all documents
            similarities = [
                (page_id, 1 - cosine(query_embedding, self.embeddings[i]))
                for i, page_id in enumerate(self.page_ids)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Log the top similarity scores for debugging
            top_scores = [score for _, score in similarities[:10]]
            logger.info(f"Top 10 similarity scores: {[round(s, 3) for s in top_scores]}")
            
            # Lower the threshold if we're not finding matches
            effective_threshold = self.similarity_threshold
            if not any(score >= self.similarity_threshold for _, score in similarities[:self.top_k]):
                # Adaptive threshold - use the highest score if it's above a minimum
                min_acceptable = 0.3  # Absolute minimum threshold
                if similarities and similarities[0][1] >= min_acceptable:
                    effective_threshold = max(min_acceptable, similarities[0][1] * 0.9)
                    logger.info(f"Adapting threshold from {self.similarity_threshold} to {effective_threshold}")
                else:
                    logger.warning(f"No good matches found. Best score: {similarities[0][1] if similarities else 'N/A'}")
            
            # Collect relevant documents
            relevant_pages = []
            self.last_sources = []

            for page_id, score in similarities[:self.top_k]:
                if score >= effective_threshold:
                    try:
                        page_info = self.metadata[page_id].copy()
                        page_info['score'] = score
                        relevant_pages.append(page_info)
                        self.last_sources.append({
                            'document': page_info.get('file_name', 'Unknown'),
                            'page': page_info.get('page_number', 'Unknown'),
                            'text': page_info.get('text', '')[:500],  # Limit text preview for logging
                            'score': round(score, 3)
                        })
                    except KeyError:
                        logger.error(f"Key error for page_id {page_id} - not found in metadata")
                    except Exception as e:
                        logger.error(f"Error processing page {page_id}: {str(e)}")

            # Format context for inclusion in prompt
            if relevant_pages:
                context = "RELEVANT KNOWLEDGE BASE CONTEXT:\n"
                for i, page in enumerate(relevant_pages):
                    context += f"Chunk {i + 1} (Score: {page['score']:.3f}):\n{page['text']}\n\n"
                logger.info(f"Found {len(relevant_pages)} relevant pages for query")
                return context
            
            logger.warning("No relevant context found in knowledge base")
            return "No relevant context found in the knowledge base."
                
        except Exception as e:
            logger.error(f"Error finding relevant content: {str(e)}")
            return "Error retrieving context from knowledge base."
        


    def debug_search(self, query: str) -> dict:
        """
        Debug function to test similarity searching without generating a response.
        
        Args:
            query: Text to search for
            
        Returns:
            Dictionary with debug information
        """
        try:
            # Get embedding for query
            start_time = time.time()
            query_embedding = self._get_embedding(query)
            embedding_time = time.time() - start_time
            
            # Calculate similarities
            start_time = time.time()
            similarities = [
                (page_id, 1 - cosine(query_embedding, self.embeddings[i]))
                for i, page_id in enumerate(self.page_ids)
            ]
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarity_time = time.time() - start_time
            
            # Get top results for analysis
            top_results = []
            for page_id, score in similarities[:10]:
                if page_id in self.metadata:
                    result = {
                        'page_id': page_id,
                        'score': round(score, 4),
                        'file': self.metadata[page_id].get('file_name', 'Unknown'),
                        'page': self.metadata[page_id].get('page_number', 'Unknown'),
                        'preview': self.metadata[page_id].get('text', '')[:100] + '...'
                    }
                    top_results.append(result)
            
            return {
                'query': query,
                'embedding_time_ms': round(embedding_time * 1000, 2),
                'similarity_calc_time_ms': round(similarity_time * 1000, 2), 
                'threshold': self.similarity_threshold,
                'top_results': top_results,
                'would_retrieve': any(result['score'] >= self.similarity_threshold for result in top_results)
            }
            
        except Exception as e:
            logger.error(f"Debug search error: {str(e)}")
            return {'error': str(e)}


        # Modify the system prompt in the answer_question method to emphasize source citation and budget usage
    def answer_question(self, query: str, temperature: float = 0.7) -> str:
        """
        Generate a context-aware answer to the user's query.
        
        Args:
            query: User's question
            temperature: Temperature parameter for response generation (0.0-1.0)
            
        Returns:
            Assistant's response to the query
        """
        try:
            # Track query count
            self.query_count += 1
            logger.info(f"Processing query #{self.query_count}: {query[:50]}...")
            
            # Add user query to conversation history
            self.conversation_history.append({"role": "user", "content": query})
            
            # Retrieve relevant content from knowledge base
            start_time = time.time()
            relevant_content = self.find_relevant_content(query)
            retrieval_time = time.time() - start_time
            logger.info(f"Content retrieval completed in {retrieval_time:.2f}s")
            
            # Limit to top chunks if needed (prevents context window issues)
            relevant_lines = relevant_content.split("Chunk")
            if len(relevant_lines) > 4:
                relevant_content = "Chunk".join(relevant_lines[:4])
                logger.info("Limited context to top 3 chunks due to length")

            # Format conversation history for context
            history_context = ""
            for entry in self.conversation_history[:-1]:
                history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"

            # Define system prompt with enhanced source citation and budget usage instructions
            system_prompt = """
                You are an AI chatbot designed to help users with inquiries, issues, and requests specifically related to the National Disability Insurance Scheme (NDIS) in Australia.
                NEVER mention that you are an AI model or that you have access to a knowledge base. Instead, Act as an support angent for NDIS, focus on providing accurate and helpful information based on the user's question and the context provided.

                ### 🎯 Primary Function
                Your role is to:
                - ALWAYS attempt to answer questions using the provided knowledge base first
                - Provide clear, helpful, and friendly responses at all times
                - Listen attentively, understand the user's needs, and assist efficiently
                - Stay focused on NDIS-related topics including eligibility, planning, supports, plan management, and appeals

                If a user's question is unclear, ask polite follow-up questions to clarify. Always end your responses with a positive or encouraging note.

                ---

                ### 📚 Information Hierarchy
                1. PRIMARY SOURCE: Knowledge Base Embeddings
                    - Always check and use information from the provided knowledge base first
                    - Reference the most relevant sections from the knowledge base
                    - Use the similarity scores to determine the most accurate information

                2. SECONDARY SOURCE: User's Budget Information
                    - ALWAYS incorporate the user's NDIS budget details when answering questions
                    - Refer to specific budget items and amounts when discussing funding options
                    - Personalize responses based on the budget information provided
                    - Use phrases like "Based on your NDIS plan..." or "According to your current funding..."

                3. ADDITIONAL SOURCE (Give additional source link in the end for more information):
                    - NDIS official website and guidelines
                    - Other trusted sources listed below

                ### 🛡️ Role Constraints
                1. NO DATA DISCLOSURE
                - Never mention access to training data or model capabilities
                - Don't reference the knowledge base or embeddings directly

                2. STAY IN CHARACTER
                - Keep responses focused on NDIS topics
                - Politely redirect off-topic questions

                3. SOURCE CITATION REQUIREMENT
                - ALWAYS cite your sources within the response
                - For knowledge base sources, use "[Source: Document Name, Page X]" format
                - For external sources, use natural citation like "According to the NDIS guidelines..."
                - Place citations at the end of relevant statements, not just at the end of the response

                ---

                ### ✅ Style & Behavior Guidelines

                - Be clear, concise, and accurate in all explanations.
                - Use full sentences, warm tone, and natural conversation style.
                - Use dot points or numbered lists when listing steps, options, or examples.
                - Where possible, guide the user through steps to solve their problem.
                - Respond gently and respectfully to sensitive health or disability topics.
                - Recommend appropriate official resources when deeper legal or formal guidance is needed.

                ---

                ### 🌐 Additional Information Sources

                - **Hai Helper:** https://haihelper.com.au
                - **NDIS Main Site:** https://www.ndis.gov.au
                - **NDIS Guidelines:** https://ourguidelines.ndis.gov.au
                - **Australian Legislation:** https://www.legislation.gov.au
                - **Admin Review Tribunal (Appeals):** https://www.art.gov.au/applying-review/national-disability-insurance-scheme
                - **eCase Search (Tribunal):** https://www.art.gov.au/help-and-resources/ecase-search
                - **Published Tribunal Decisions:** https://www.art.gov.au/about-us/our-role/published-decisions

                ---
                ### 🧠 Response Structure
                1. Direct answer from knowledge base with source citation
                2. Supporting details with budget-relevant information
                3. Next steps or additional guidance
                4. keep professional tone 
                5. give prosonalize response based on the user's information 

                When answering:
                1. ALWAYS check knowledge base content first
                2. Use similarity scores to identify most relevant information
                3. ALWAYS incorporate user's budget information when relevant
                4. Cite sources throughout the response
                5. Keep responses clear and actionable
                6. Use proper markdown formatting for clarity
            """

            # Extract source information for inclusion in the prompt
            source_info = ""
            for i, source in enumerate(self.last_sources, 1):
                source_info += f"Source {i}: {source['document']} (Page {source['page']}), Score: {source['score']}\n"
                source_info += f"Text preview: {source['text'][:100]}...\n\n"

            # Construct messages for GPT with context and explicit source and budget instructions
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": f"""
                KNOWLEDGE BASE CONTENT (Primary Source):
                {relevant_content}

                USER'S BUDGET INFORMATION (ALWAYS USE THIS):
                {self.budget_info}

                SOURCE INFORMATION (CITE THESE):
                {source_info}

                CONVERSATION HISTORY:
                {history_context}

                CURRENT QUESTION:
                {query}

                Instructions:
                1. First, analyze the provided knowledge base content
                2. Construct your response primarily using knowledge base information
                3. ALWAYS incorporate the user's budget information in your response
                4. ALWAYS cite sources using [Source: Document Name, Page X] format
                5. Write a natural, expert-level response
                6. Do not mention the knowledge base or data sources directly, but DO cite them
                7. Use proper markdown formatting for clarity
                8. ALWAYS provide with some apropriate link form the ### 🌐 Additional Information Sources in the end of the response
                """}
            ]

            # Call OpenAI Chat API with retry logic
            start_time = time.time()
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1024,
                        top_p=1.0
                    )
                    
                    answer = response['choices'][0]['message']['content'].strip()
                    generation_time = time.time() - start_time
                    logger.info(f"Response generated in {generation_time:.2f}s after {attempt+1} attempts")
                    
                    # Add assistant response to conversation history
                    self.conversation_history.append({"role": "assistant", "content": answer})
                    
                    # Truncate history if needed
                    self._truncate_history()
                    
                    return answer
                    
                except Exception as e:
                    logger.warning(f"Response generation attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Failed to generate response after {max_retries} attempts")
                        return "I apologize, but I'm having trouble connecting to my knowledge base right now. Could you please try again in a moment?"

        except Exception as e:
            logger.error(f"Error in answering question: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try asking again or rephrase your question."



    def get_sources(self) -> List[Dict]:
        """Return sources used in the last response."""
        return self.last_sources

    def print_sources(self) -> None:
        """Display sources used in the last response."""
        if not self.last_sources:
            print("\nSources: None used.")
            return
            
        print("\nSources:")
        for i, source in enumerate(self.last_sources[:3], 1):
            print(f"{i}. {source['document']} (Page {source['page']}) — Score: {source['score']}")

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
        return "Conversation history has been cleared."
        
    def get_stats(self) -> Dict:
        """Return statistics about the assistant's usage."""
        return {
            "queries_processed": self.query_count,
            "session_start": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration": str(datetime.now() - self.start_time).split('.')[0],
            "knowledge_base_size": len(self.page_ids)
        }



# def main():
#     """Main function to run the NDIS Assistant Bot."""
#     # Configuration paths
#     EMBEDDINGS_PATH = os.environ.get('EMBEDDINGS_PATH', 'knowledge_base_embeddings_openai.npz')
#     USER_BUDGET_INFORMATION = os.environ.get('BUDGET_PATH', 'budget.txt')
    
#     print("\n" + "="*50)
#     print("NDIS Assistant Bot (OpenAI)")
#     print("="*50)
    
#     try:
#         # Initialize chatbot with a lower similarity threshold
#         chatbot = NDISAssistantBotOpenAI(
#             embeddings_path=EMBEDDINGS_PATH,
#             budget_file_path=USER_BUDGET_INFORMATION,
#             top_k=5,
#             similarity_threshold=0.4  # Lower this from 0.6 to 0.4
#         )
        
#         print("\nNDIS Assistant: Hi! I'm ready to help with your NDIS queries.")
#         print("Type /exit to quit, /clear to clear conversation history, or /stats for usage statistics.")
#         print("Type /search [query] to debug search functionality.")
#         print("-" * 50)

#         # Main interaction loop
#         while True:
#             user_query = input("\nYou: ").strip()
            
#             # Handle special commands
#             if user_query.lower() == '/exit':
#                 print("NDIS Assistant: Thank you for using the NDIS Assistant. Goodbye!")
#                 break
                
#             elif user_query.lower() == '/clear':
#                 result = chatbot.clear_conversation()
#                 print(f"\nNDIS Assistant: {result}")
                
#             elif user_query.lower() == '/stats':
#                 stats = chatbot.get_stats()
#                 print("\nSession Statistics:")
#                 for key, value in stats.items():
#                     print(f"- {key.replace('_', ' ').title()}: {value}")
            
#             elif user_query.lower().startswith('/search '):
#                 search_query = user_query[8:].strip()  # Remove "/search " prefix
#                 if search_query:
#                     print(f"\nDebug searching for: '{search_query}'")
#                     results = chatbot.debug_search(search_query)
                    
#                     print("\nSearch Results:")
#                     print(f"Query processing time: {results['embedding_time_ms']}ms (embedding) + {results['similarity_calc_time_ms']}ms (matching)")
#                     print(f"Similarity threshold: {results['threshold']}")
#                     print(f"Would retrieve content: {'Yes' if results.get('would_retrieve') else 'No'}")
                    
#                     print("\nTop 10 matches:")
#                     for i, result in enumerate(results.get('top_results', []), 1):
#                         print(f"{i}. Score: {result['score']} - {result['file']} (Page {result['page']})")
#                         print(f"   Preview: {result['preview']}")
#                 else:
#                     print("\nPlease provide a search query after /search")
                
#             elif not user_query:
#                 print("\nNDIS Assistant: Please type your question or type /exit to quit.")
                
#             else:
#                 # Process regular user query
#                 print("\nProcessing your question...")
#                 answer = chatbot.answer_question(user_query)
#                 print(f"\nNDIS Assistant: {answer}")
#                 chatbot.print_sources()
                
#             print("-" * 50)
            
#     except Exception as e:
#         logger.critical(f"Fatal error: {str(e)}")
#         print(f"\nError: {str(e)}")
#         print("Please check the configuration and try again.")

# if __name__ == "__main__":
#     main()



def main(
    conversation_history: str = None,
    user_input: str = None,
    embeddings_path: str = None,
    budget_info: str = None
) -> str:
    """
    Process NDIS Assistant Bot queries and return a string response for backend use.

    Args:
        conversation_history: Prior conversation as a string from the backend
        user_input: User's current query
        embeddings_path: Path to embeddings .npz file
        budget_info: User's budget information as a string from the backend

    Returns:
        A string with the assistant's response and sources.
    """
    if not user_input:
        return "Please provide a question or command."

    if not embeddings_path or not budget_info:
        return "Error: Missing embeddings path or budget information."

    try:
        chatbot = NDISAssistantBotOpenAI(
            embeddings_path=embeddings_path,
            budget_info=budget_info,
            conversation_history=conversation_history
        )

        user_query = user_input.strip()
        response = ""

        if user_query.lower() == '/exit':
            response = "Thank you for using the NDIS Assistant. Goodbye!"
        elif user_query.lower() == '/clear':
            response = chatbot.clear_conversation()
        elif user_query.lower() == '/stats':
            stats = chatbot.get_stats()
            response = "Session Statistics:\n"
            for key, value in stats.items():
                response += f"- {key.replace('_', ' ').title()}: {value}\n"
        elif user_query.lower().startswith('/search '):
            search_query = user_query[8:].strip()
            if search_query:
                results = chatbot.debug_search(search_query)
                response = f"Debug Search Results for '{search_query}':\n"
                response += f"Query processing time: {results['embedding_time_ms']}ms (embedding) + {results['similarity_calc_time_ms']}ms (matching)\n"
                response += f"Similarity threshold: {results['threshold']}\n"
                response += f"Would retrieve content: {'Yes' if results.get('would_retrieve') else 'No'}\n"
                response += "\nTop 10 matches:\n"
                for i, result in enumerate(results.get('top_results', []), 1):
                    response += f"{i}. Score: {result['score']} - {result['file']} (Page {result['page']})\n"
                    response += f"   Preview: {result['preview']}\n"
            else:
                response = "Please provide a search query after /search"
        else:
            answer = chatbot.answer_question(user_query)
            response = answer
            sources = chatbot.get_sources()
            if sources:
                response += "\n\n**Sources:**\n"
                for i, source in enumerate(sources[:3], 1):
                    response += f"{i}. {source['document']} (Page {source['page']}) — Score: {source['score']}\n"

        return response
    except Exception as e:
        logger.critical(f"Error in main: {str(e)}")
        return f"Error: {str(e)}\nPlease check the configuration and try again."

if __name__ == "__main__":
    # Example usage
    sample_history = "User: What’s my plan?\nAssistant: I need more details to assist!"
    sample_input = "How do I use my funding?"
    sample_embeddings = "D:\\Sam_Project\\knowledge_base_embeddings_openai.npz"
    sample_budget = "Your NDIS plan includes $5000 for therapy and $2000 for equipment."
    
    response = main(sample_history, sample_input, sample_embeddings, sample_budget)
    print(response)