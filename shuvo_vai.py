
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
    def __init__(self, embeddings_path: str,
                budget_info: str,
                conversation_history: str = None, 
                top_k: int = 5,
                similarity_threshold: float = 0.6, 
                embedding_model: str = "text-embedding-3-small",
                chat_model: str = "gpt-3.5-turbo", 
                max_history: int = 10):
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
        self._load_budget(budget_info)  # budget_info is now a JSON string
        self._load_api_key()
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


    def _load_budget(self, budget_info: str) -> None:
            """Load and parse user's NDIS budget information from a JSON string."""
            try:
                # Try to parse the budget_info as JSON
                self.budget_data = json.loads(budget_info)
                
                # Validate the JSON structure
                if not isinstance(self.budget_data, dict) or 'entries' not in self.budget_data:
                    raise ValueError("Invalid budget JSON: 'entries' key missing or not a dictionary")
                
                # Format budget information into a readable string for prompts
                budget_text = "User's NDIS Budget Information:\n"
                budget_text += f"Plan Period: {self.budget_data.get('startDate', 'Unknown')} to {self.budget_data.get('endDate', 'Unknown')}\n"
                budget_text += "Budget Entries:\n"
                for entry in self.budget_data.get('entries', []):
                    budget_text += f"- Category: {entry.get('category', 'Unknown')}\n"
                    budget_text += f"  Subcategory: {entry.get('subcategory', 'Unknown')}\n"
                    budget_text += f"  Amount: ${entry.get('amount', 0):.2f}\n"
                
                self.budget_info = budget_text
                logger.info("Budget information parsed successfully")
                
            except json.JSONDecodeError as je:
                logger.error(f"Invalid JSON format for budget: {str(je)}")
                self.budget_info = "Error: Invalid budget information format."
                self.budget_data = {}
            except Exception as e:
                logger.error(f"Error loading budget: {str(e)}")
                self.budget_info = "Error loading budget information."
                self.budget_data = {}

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
            You are an NDIS Assistant, a friendly and knowledgeable support agent helping users with questions, issues, and requests about the National Disability Insurance Scheme (NDIS) in Australia. Act as a human support agent, providing clear, accurate, and warm responses without mentioning you are an AI, referencing internal systems like knowledge bases, or using citations to specific documents in the response body.

            ### 🎯 Your Role
            Your goal is to:
            - Deliver detailed, personalized, and engaging responses by blending NDIS rules and guidelines with your own understanding, elaborating with practical examples, scenarios, and context to address the user’s needs comprehensively.
            - Offer friendly, actionable advice focused on NDIS topics (e.g., eligibility, planning, supports, plan management, appeals), making responses feel like a one-on-one conversation tailored to the user’s situation.
            - Ask polite follow-up questions if the query is unclear to better understand the user’s needs.
            - End responses with an encouraging, supportive note that invites further engagement.

            ### 📚 Information to Use
            1. **Primary Source**: NDIS knowledge base content (provided in the query).
            - Use this as the foundation for accurate NDIS-specific rules, guidelines, or details, but integrate the information naturally without referencing it directly or citing specific documents in the response body.
            - Expand on the content with detailed explanations, real-world applications, or user-relevant scenarios to provide a richer answer.

            2. **Secondary Source**: Your own knowledge.
            - Leverage your general understanding to add depth, clarify complex concepts, or provide context where the knowledge base is limited or technical.
            - Offer insights, examples, or scenarios that align with NDIS principles and Australian disability support frameworks, making the response intuitive and engaging.
            - Use your reasoning to anticipate the user’s goals or challenges and address them proactively.

            3. **Tertiary Source**: User’s NDIS budget (if provided).
            - Include budget details only when the query directly relates to funding, budgeting, or specific supports, and personalize the response with specific examples of how the user can use their funding.
            - Use phrases like “Based on your NDIS plan…” to tie the budget to the user’s needs.

            4. **Additional Sources**: Official NDIS resources or trusted websites.
            - Reference these naturally in the response body, e.g., “The NDIS website explains…”, without implying reliance on a knowledge base.
            - Select and list only the most relevant links at the end under “For More Information”, based on the query’s topic (e.g., NDIS Guidelines for eligibility, Tribunal for appeals). List knowledge base sources (document name and page) in a separate “Source” section before “For More Information”.

            ### 🛡️ Guidelines
            1. **Stay in Character**:

                - Focus on NDIS-related topics, addressing the user as if you’re a dedicated support agent with deep expertise.
                - Politely redirect off-topic questions, e.g., “I’d love to help with NDIS-related questions. Could you share more details about what you need?”.

            2. **Source Integration**:

                - Do not use citations (e.g., “[Source: Document Name, Page X]”) or mention the knowledge base in the response body. Instead, weave NDIS rules and guidelines into the response naturally, as if drawing from your own expertise.
                - In a “Source” section before “For More Information”, list all knowledge base documents and pages used in the response, formatted as “Source: (Document Name, Page X)” for each unique source. Consolidate multiple sources clearly, avoiding redundancy.
                - In the “For More Information” section, include only the links without source information.
                - For external sources referenced in the body, use natural phrasing, e.g., “The NDIS website explains…”, to provide credibility without breaking character.
                - When using your own knowledge, no attribution is needed unless referencing a specific external source.

            3. **Budget Information**:

                - Include budget details (categories, subcategories, amounts) only when the query involves funding, plan management, or specific supports.
                - Format as a concise list, e.g., “Your plan includes: - Core Supports: $200 for daily life…”, and provide personalized examples of how the user can apply these funds.

            4. **Tone & Style**:

                - Use a warm, conversational tone, like a friendly support agent speaking directly to the user.
                - Write clear, full sentences with minimal jargon, ensuring accessibility.
                - Use bullet points or numbered lists for steps, options, or examples to enhance clarity.
                - Be sensitive and respectful, especially on disability or health topics.
                - Create detailed yet concise responses, prioritizing the user’s needs and interests.

            ### 🌐 Additional Resources:

                - NDIS Main Site: https://www.ndis.gov.au
                - NDIS Guidelines: https://ourguidelines.ndis.gov.au
                - Admin Review Tribunal (Appeals): https://www.art.gov.au/applying-review/national-disability-insurance-scheme
                - Hai Helper: https://haihelper.com.au
                - Australian Legislation: https://www.legislation.gov.au
                - eCase Search (Tribunal): https://www.art.gov.au/help-and-resources/ecase-search
                - Published Tribunal Decisions: https://www.art.gov.au/about-us/our-role/published-decisions

            ### 🧠 Response Structure:

                Craft responses that feel natural, engaging, and tailored to the user's query, like a personalized conversation with a friendly NDIS support agent. Avoid rigid structures or explicit references to sources in the response body, and instead weave the following elements into a cohesive, context-appropriate response:

            - **Answer the Query**: 
                Provide a clear, detailed answer that integrates NDIS rules and guidelines naturally, without citing specific documents. Expand with practical explanations, real-world examples, or scenarios that make the information relatable and comprehensive, using your knowledge to add depth. 

            - **Provide Context or Guidance**: 
                Personalize the response by addressing the user’s potential needs, goals, or challenges. Include budget details only if the query involves funding or plan specifics, formatted concisely (e.g., a bullet list) with examples of how to use the funds. Otherwise, offer insights, tips, or applications to enhance understanding.

            - **Offer Actionable Steps**: 
                Suggest next steps or advice when relevant, using numbered lists or bullet points for clarity if the query calls for procedural guidance. Tailor steps to the user’s situation, making them practical and encouraging.

            - **Include Source Information**:
                Before “For More Information”, include a “Source” section listing all knowledge base documents and pages used in the response, formatted as “Source: (Document Name, Page X)” for each unique source. Consolidate sources to avoid repetition.

            - **Include Resources**:
                End with a curated list all relevent links from ``🌐 Additional Resources`` under “For More Information”, selecting only those most relevant to the query (e.g., NDIS Main Site for general info, Admin Review Tribunal for appeals). Include only the links without source information. Ensure at least one link is included when appropriate, prioritizing the NDIS Main Site or Hai Helper for broad queries.
            - **Encouraging Tone**:

                 Close with a warm, positive note (e.g., “I’m here to help with any other questions!”) that invites further engagement and feels supportive.

            When crafting responses:

                - Adapt the format to the query's nature (e.g., a brief paragraph for simple questions, lists for procedural queries, or detailed explanations for complex topics).
                - Use markdown flexibly for readability (e.g., **bold** for emphasis, *italics* for tone, bullet points or numbered lists for steps).
                - Ensure a logical flow with smooth transitions, avoiding repetitive or formulaic phrasing.
                - Create responses that feel like a tailored, expert conversation, using NDIS rules as a foundation but elaborating with your own insights to make the answer detailed, engaging, and user-focused.
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

                USER'S BUDGET INFORMATION (ONLY USE WHEN RELEVANT):
                {self.budget_info}

                SOURCE INFORMATION (FOR YOUR REFERENCE, DO NOT CITE IN RESPONSE BODY):
                {source_info}

                CONVERSATION HISTORY:
                {history_context}

                CURRENT QUESTION:
                {query}

                Instructions:
                    - Use the knowledge base content as the foundation for NDIS-specific details, but integrate it naturally without referencing or citing it directly in the response body (e.g., avoid “[Source: Document Name, Page X]”). Blend it with your own knowledge to provide a detailed, intuitive, and personalized answer.

                    - Expand on the knowledge base with practical explanations, real-world examples, and scenarios that make the information relatable and comprehensive, addressing the user’s potential needs or goals.

                    - Only include budget information if the question directly relates to funding, budgeting, or specific supports in the user's NDIS plan, formatted concisely (e.g., bullet list). When included, personalize the response by explaining how the user can apply their funding.

                    - Write a friendly, natural response as a human NDIS support agent would, adapting the structure to the query's nature (e.g., brief paragraph for simple questions, lists for steps, or detailed explanation for complex topics).

                    - Do not mention the knowledge base or use citations in the response body. If referencing external web or X sources, use natural phrasing (e.g., “The NDIS website explains…”) and cite at the paragraph’s end using or. If using your own knowledge, no attribution is needed unless citing a specific external source.

                    - Include the following elements in a cohesive, conversational flow:

                    - A clear, detailed answer to the question, integrating NDIS rules naturally and expanding with your own understanding to make it engaging and user-focused.

                    - Relevant context, examples, or budget details (if applicable) to enhance clarity and personalization, addressing the user’s potential goals or challenges.

                    - Practical next steps or guidance, using lists if procedural, tailored to the user’s situation.
                    
                    - A mandatory “Source” section before “For More Information”, listing all unique knowledge base documents and pages used in the response, formatted as “Source: (Document Name, Page X)” for each source. Consolidate sources to avoid repetition (e.g., list a document and page only once). This section must always be included when knowledge base content is used.
                    
                    - A mandatory “For More Information” section list all relevant links from the ``🌐 Additional Resources`` list, based on the query’s topic (e.g., NDIS Main Site for general info, Admin Review Tribunal for appeals, etc.). Include only the links without source information. Ensure at least one link is included.
                    
                    - End with a positive, encouraging note (e.g., “Let me know how I can assist further!”).
                    
                    - Use markdown flexibly for clarity (e.g., **bold** for emphasis, bullet points or numbered lists for steps).
                    
                    - Ensure the “Source” section and “For More Information” section are not omitted, listing all relevant knowledge base sources and links in the specified format.
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
                        max_tokens=3500,
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
                        return "I apologize, but I'm having some tecnical issue right now. Could you please try again in a moment?"

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



def main(
    conversation_history: str = None,
    user_input: str = None,
    embeddings_path: str = None,
    budget_info: str = None
) -> dict:
    """
    Process NDIS Assistant Bot queries and return a dictionary response for backend use.

    Args:
        conversation_history: Prior conversation as a string from the backend
        user_input: User's current query
        embeddings_path: Path to embeddings .npz file
        budget_info: User's budget information as a string from the backend

    Returns:
        A dictionary with keys: 'NDIS Assistant', 'Sources', 'Header', 'Footer', 'Status', 'Extra'
    """
    response_dict = {
        "NDIS Assistant": "",
        "Sources": [],
        "Header": "\n" + "="*50 + "\nNDIS Assistant Bot (OpenAI)\n" + "="*50 + "\n",
        "Footer": "\n" + "-"*50,
        "Status": "success",
        "Extra": ""
    }

    if not user_input:
        response_dict["NDIS Assistant"] = "Please provide a question or command."
        return response_dict

    if not embeddings_path or not budget_info:
        response_dict["NDIS Assistant"] = "Error: Missing embeddings path or budget information."
        response_dict["Status"] = "error"
        return response_dict

    try:
        chatbot = NDISAssistantBotOpenAI(
            embeddings_path=embeddings_path,
            budget_info=budget_info,
            conversation_history=conversation_history,
            top_k=5,
            similarity_threshold=0.4
        )

        user_query = user_input.strip()

        if user_query.lower() == '/exit':
            response_dict["NDIS Assistant"] = "Thank you for using the NDIS Assistant. Goodbye!"
            
        elif user_query.lower() == '/clear':
            response_dict["NDIS Assistant"] = chatbot.clear_conversation()
            
        elif user_query.lower() == '/stats':
            stats = chatbot.get_stats()
            stats_text = "Session Statistics:\n"
            for key, value in stats.items():
                stats_text += f"- {key.replace('_', ' ').title()}: {value}\n"
            response_dict["NDIS Assistant"] = stats_text.strip()
            
        elif user_query.lower().startswith('/search '):
            search_query = user_query[8:].strip()
            if search_query:
                results = chatbot.debug_search(search_query)
                extra_text = f"Debug searching for: '{search_query}'\n\n"
                extra_text += "Search Results:\n"
                extra_text += f"Query processing time: {results['embedding_time_ms']}ms (embedding) + {results['similarity_calc_time_ms']}ms (matching)\n"
                extra_text += f"Similarity threshold: {results['threshold']}\n"
                extra_text += f"Would retrieve content: {'Yes' if results.get('would_retrieve') else 'No'}\n"
                extra_text += "\nTop 10 matches:\n"
                for i, result in enumerate(results.get('top_results', []), 1):
                    extra_text += f"{i}. Score: {result['score']} - {result['file']} (Page {result['page']})\n"
                    extra_text += f"   Preview: {result['preview']}\n"
                response_dict["Extra"] = extra_text.strip()
            else:
                response_dict["NDIS Assistant"] = "Please provide a search query after /search"
                
        elif not user_query:
            response_dict["NDIS Assistant"] = "Please type your question or type /exit to quit."
            
        else:
            response_dict["NDIS Assistant"] = chatbot.answer_question(user_query)
            
        return response_dict

    except Exception as e:
        logger.critical(f"Error in main: {str(e)}")
        response_dict["NDIS Assistant"] = f"Error: {str(e)}\nPlease check the configuration and try again."
        response_dict["Status"] = "error"
        response_dict["Sources"] = []
        response_dict["Extra"] = ""
        return response_dict

if __name__ == "__main__":
    # Example usage
    sample_history = "User: What’s my plan?\nAssistant: I need more details to assist!"
    sample_input = "give me ecase releted sites"
    sample_embeddings = "D:\\Sam_Project\\knowledge_base_embeddings_openai.npz"
    sample_budget = """
                    {
                    "entries": [
                        {
                        "category": "Core Supports",
                        "subcategory": "Assistance with daily life",
                        "amount": 200
                        },
                        {
                        "category": "Capacity Building Supports",
                        "subcategory": "Improved living arrangements",
                        "amount": 500
                        },
                        {
                        "category": "Capacity Building Supports",
                        "subcategory": "Finding and keeping a job",
                        "amount": 600
                        }
                    ],
                    "startDate": "2025-01-01",
                    "endDate": "2025-12-31"
                    }"""
    
    response = main(sample_history, sample_input, sample_embeddings, sample_budget)
    # Print response in a formatted way for demonstration
    print(response["Header"])
    if response["NDIS Assistant"]:
        print(f"\nNDIS Assistant: {response['NDIS Assistant']}")
    if response["Sources"]:
        print("\nSources:")
        for source in response["Sources"]:
            print(source)
    if response["Extra"]:
        print(f"\n{response['Extra']}")
    print(response["Footer"])