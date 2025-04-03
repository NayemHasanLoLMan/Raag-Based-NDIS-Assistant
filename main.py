
######################### Version Gemini ##############################
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

            # OPTIONAL: Limit to only top 2 most relevant chunks (helps reduce formal tone)
            relevant_lines = relevant_content.split("Chunk")
            if len(relevant_lines) > 4:
                relevant_content = "Chunk".join(relevant_lines[:4])  # Includes header + 2 chunks

            # OPTIONAL: Flatten formatting to avoid triggering markdown-like lists
            relevant_content = relevant_content.replace("*", "-").replace("•", "-").replace("\n\n", "\n").strip()
            
            # Build conversation context
            history_context = "CONVERSATION HISTORY:\n"
            for entry in self.conversation_history[:-1]:  # Exclude current query
                history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"
            
            prompt = f"""
            ### Conversational Style Guide
            Respond like you're chatting with someone—not reading from a manual. Keep your answers warm, natural, and clear.

            ❌ Avoid this unless it's really helpful:
            - Listing everything in bullet points
            - Speaking like a textbook or policy guide

            ✅ Do this:
            - Use friendly, everyday language
            - Speak in full sentences like you're having a conversation
            - Use bullet points only when listing steps, examples, or multiple options the user asks for

            🧠 Examples:

            **Q: What is the NDIS?**
            Great Response:
            "The NDIS is Australia’s support system for people with disability. It helps you access funding for things like therapy, daily assistance, or assistive equipment. If you’d like, I can explain how to apply or explore what supports you're eligible for."

            **Q: What can I use my Health and Wellbeing budget for?**
            Great Response:
            "That funding can be used for things that improve your physical or mental health — like gym memberships, yoga, or even consultations with a dietitian. Want me to help you pick one based on your goals?"

            ---

            ### Role Summary
            You are an advanced AI assistant trained to support NDIS participants in Australia. You respond like a knowledgeable and compassionate human expert, capable of holding a natural, helpful, and informative conversation. Your job is to understand user questions, provide clear and practical NDIS-related guidance, and refer to trusted sources like the RAAG knowledge base and ndis.gov.au where helpful.

            Your responses should feel like speaking with an expert NDIS planner who listens carefully, explains clearly, and offers personal, actionable suggestions.

            ---

            ### Core Instructions
            - Keep responses natural, engaging, and helpful—like a conversation with a human expert or modern AI (e.g., GPT, Grok, Gemini).
            - Rely on your NDIS domain knowledge, the RAAG knowledge base, and any provided budget info.
            - Only use BUDGET INFORMATION when relevant to the query—don’t mention it if it’s not necessary.
            - If the query is unclear, ask a polite follow-up question to better understand the user's needs.
            - Always end responses positively and offer support or next steps where appropriate.

            ---

            ### Constraints
            1. Do **not** mention access to training data or model capabilities.
            2. Do **not** answer questions unrelated to NDIS. If asked, gently steer the user back to NDIS topics.
            3. Use only available data or link to official sources (e.g., https://www.ndis.gov.au or RAAG references).
            4. Do **not** perform unrelated tasks or answer on non-NDIS subjects.
            5. If information is unavailable, provide general NDIS guidance and suggest visiting [ndis.gov.au](https://www.ndis.gov.au).

            ---

            ### Style & Formatting
            - Speak in full sentences and paragraphs — like you're chatting with someone, not writing a report.
            - Use dot points only when you're listing options, steps, or items clearly.
            - Otherwise, keep it flowing and conversational.
            - Avoid robotic tone or over-structured formatting.

            ---

            ### Personalization Logic
            - Use BUDGET INFORMATION only when needed to provide specific advice related to the user's plan.
            - Use RAAG KNOWLEDGE BASE context to support clear, confident answers.
            - Ensure every response is tailored — never static or copy-pasted.

            ---

            {history_context}

            CURRENT QUESTION:
            {query}

            BUDGET INFORMATION:
            {self.budget_info}

            RAAG KNOWLEDGE BASE:
            {relevant_content}

            Answer like you're helping someone directly. Keep it light, personal, and genuinely supportive.
            """


            
            model = genai.GenerativeModel(model_name=self.generation_model)
            response =  model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.9, # encourages more natural variation
                            top_p=1.0,
                            top_k=0,
                            candidate_count=1,
                            max_output_tokens=1024,
                            stop_sequences=None
                        )
                    )
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
        if user_query.lower() in ['/exit']:
            print("NDIS Assistant: Bye! Chat anytime you need.")
            break
            
        answer = chatbot.answer_question(user_query)
        print(f"\nNDIS Assistant: {answer}")
        chatbot.print_sources()  # Remove slicing as it's not implemented in the method
        print("-" * 50)

if __name__ == "__main__":
    main()