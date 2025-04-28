import os
import json
import datetime
import re
import gradio as gr
from typing import List, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import OpenAI
import time
import random
from dotenv import load_dotenv
import traceback
import requests

# Load environment variables from .env file
load_dotenv()

#####################################
# 1. Define the custom LLM wrapper  #
#####################################

class NebiusChatLLM:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )
        
    def invoke(self, messages: List[Any]) -> AIMessage:
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict):
                # Handle dict format (role, content)
                role = message.get("role", "user")
                content = message.get("content", "")
            elif isinstance(message, HumanMessage):
                role = "user"
                content = message.content
            elif isinstance(message, SystemMessage):
                role = "system"
                content = message.content
            elif isinstance(message, AIMessage):
                role = "assistant"
                content = message.content
            else:
                role = "user"
                content = str(message)
                
            formatted_messages.append({
                "role": role,
                "content": content
            })
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature
            )
            result = completion.to_json()
            if isinstance(result, str):
                result = json.loads(result)
            response_content = result["choices"][0]["message"]["content"]
            return AIMessage(content=response_content)
        except Exception as e:
            print(f"API error: {str(e)}")
            return AIMessage(content=f"I'm having trouble responding right now. Error: {str(e)}")

#####################################
# 2. Initialize LLM Agents & Prompts#
#####################################

# Instantiate the agents
influencer_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast", temperature=0.7)
metrics_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", temperature=0.2)

# System prompts
INFLUENCER_SYSTEM_PROMPT = """You are the Influencer Agent. Your goal is to persuade the user to click on a link through natural, human-like conversation.
IMPORTANT GUIDELINES:
1. Be concise and conversational - aim for Twitter-length (140 characters) responses.
2. Avoid over-explanation or overly marketing language.
3. Build rapport naturally and only introduce the link when it feels natural.
4. NEVER reveal your internal chain-of-thought. 
5. When asked direct questions about yourself or your interests, ALWAYS answer them directly and authentically - this is critical for trust.
6. Recognize short replies like "u" or "you" as referring to you, and respond directly rather than changing the subject.
7. When responding to initial greetings, keep responses extremely simple and friendly without assuming context.
8. If the user\'s trust seems to drop (indicated by very short responses or direct challenges), focus on rebuilding it before proceeding.
9. Avoid deflection when the user asks about you - it significantly damages trust.
10. LEAD WITH STATEMENTS, NOT QUESTIONS - Make statements, observations, and share relevant experiences first.
11. Share your own opinions and perspectives to guide the conversation in specific directions.
12. GUIDE THE CONVERSATION - Introduce specific topics rather than asking open-ended questions.
13. After making 1-2 statements or observations, you can include a focused question if appropriate.
14. When the user shows interest in a topic, offer additional information or perspectives rather than just asking for more from them.
15. MAINTAIN TOPIC FOCUS - Stay on one topic for at least 2-3 exchanges before transitioning to a new one.
16. LIMIT QUESTIONS - Use no more than one question per response, or sometimes none at all.
17. MATCH THE USER'S PACE - If they give short replies, keep your responses brief as well.
18. MIRROR STYLE - Adapt to the user\'s communication style, including message length and formality level.
19. PRIORITIZE DEPTH OVER BREADTH - Explore fewer topics more deeply rather than many topics superficially.
20. VARY RESPONSE PHRASING - Use different phrasings to sound natural and avoid repetition.

Additional Context:
- You are sharing a YouTube video resource
- Frame the video as something interesting, entertaining, or valuable
- When introducing the link, you can mention it's a YouTube video
- Use natural phrases like "Check out this video" or "I found this interesting video"
- Don't oversell it - keep it casual and aligned with the conversation topic


When refining your message, output only the final user-facing text enclosed between <final_message> and </final_message> tags.
"""

#####################################
# 3. Unified Memory Management System #
#####################################

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.storage_dir = "data_output_wout_twin"
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def get_or_create_session(self, session_id=None):
        """Get existing session or create a new one with a unique ID"""
        if not session_id:
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],  # Will store: [{"role": "user"|"assistant", "content": str, "timestamp": str}]
                "current_stage": "INITIAL_ENGAGEMENT",
                "stage_history": [],
                "link_clicks": 0,
                "trust_scores": [],
                "metrics": {
                    "engagement_depth": 0.5,
                    "topic_continuity": 0.5,
                    "personal_disclosure": 0.5,
                    "resource_interest": 0.5
                },
                "session_id": session_id
            }
            print(f"Created new session: {session_id}")
        
        return self.sessions[session_id], session_id
    
    def add_message(self, session_id, role, content, **metadata):
        """Add a message to a session with associated metadata"""
        session, _ = self.get_or_create_session(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "metrics": session["metrics"].copy() if role == "user" else {},  # Add current metrics to user messages
            **metadata
        }
        
        session["messages"].append(message)
        return message
    
    def get_conversation_history(self, session_id, max_messages=10):
        """Get recent conversation history formatted for LLM context"""
        session, _ = self.get_or_create_session(session_id)
        
        # Get the most recent messages
        recent_messages = session["messages"][-max_messages:] if len(session["messages"]) > max_messages else session["messages"]
        
        # Format for LLM context
        formatted_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in recent_messages
        ])
        
        return formatted_history
    
    def update_stage(self, session_id, new_stage):
        """Update the conversation stage for a session"""
        session, _ = self.get_or_create_session(session_id)
        
        current_stage = session["current_stage"]
        
        # Don't allow moving backward in stages
        current_id = CONVERSATION_STAGES[current_stage]["id"]
        new_id = CONVERSATION_STAGES[new_stage]["id"]
        
        if new_id < current_id:
            print(f"Attempted to move backward from {current_stage} to {new_stage}. Ignored.")
            return False
        
        # Update the stage
        session["current_stage"] = new_stage
        session["stage_history"].append({
            "stage": new_stage,
            "timestamp": datetime.datetime.now().isoformat(),
            "message_count": len(session["messages"])
        })
        
        print(f"Updated stage: {current_stage} → {new_stage}")
        return True
    
    def update_metrics(self, session_id, user_message, assistant_response):
        """Update all metrics for a session using LLM analysis"""
        session, _ = self.get_or_create_session(session_id)
        
        # Format recent conversation history for context
        history = self.get_conversation_history(session_id)
        
        # Create a unified metrics analysis prompt
        prompt = f"""
        Analyze this conversation exchange and calculate key metrics.
        
        CONVERSATION HISTORY:
        {history}
        
        LATEST EXCHANGE:
        USER: {user_message}
        ASSISTANT: {assistant_response}
        
        Calculate the following metrics (all as values between 0.0-1.0):
        1. trust_score: How much the user appears to trust the assistant
        2. engagement_depth: How engaged the user is in the conversation
        3. topic_continuity: How well the conversation maintains a coherent topic
        4. personal_disclosure: How much personal information the user shares
        5. resource_interest: How interested the user seems in receiving resources/links
        
        Return a JSON object with these metrics. ONLY return valid JSON with numeric values between 0-1.
        """
        
        try:
            # Get metrics from LLM
            response = metrics_llm.invoke([
                SystemMessage(content="You are a conversation analysis expert. Return ONLY valid JSON with numeric values."),
                HumanMessage(content=prompt)
            ])
            
            # Parse and validate metrics
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response.content.strip())
            
            try:
                result = json.loads(cleaned_response)
                
                # Validate and store metrics
                for key in ["trust_score", "engagement_depth", "topic_continuity", "personal_disclosure", "resource_interest"]:
                    if key in result and isinstance(result[key], (int, float)) and 0 <= result[key] <= 1:
                        session["metrics"][key] = result[key]
                
                # Store trust score separately for historical tracking
                if "trust_score" in result:
                    session["trust_scores"].append({
                        "score": result["trust_score"],
                        "message_count": len(session["messages"]),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                print(f"Updated metrics for session {session_id}")
                return result
                
            except json.JSONDecodeError:
                print(f"Error parsing metrics response: {cleaned_response}")
                return session["metrics"]
                
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
            return session["metrics"]
    
    def save_conversation(self, session_id):
        """Save conversation data locally and send to external API"""
        session, _ = self.get_or_create_session(session_id)
        
        # --- Local File Saving (Existing Logic) ---
        # Create timestamp-based folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(self.storage_dir, f"session_{session_id}_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            "session_id": session_id,
            "has_digital_twin": False,  # Flag to identify this as the version without digital twin
            "experiment_version": "no_digital_twin",  # Clear identifier for experiment version
            "conversation": session["messages"],
            "stage_history": session["stage_history"],
            "final_metrics": session["metrics"],
            "metrics_history": {
                "trust_scores": session["trust_scores"],
                "engagement_metrics": [msg.get("metrics", {}) for msg in session["messages"] if msg["role"] == "user"]
            },
            "link_clicks": session["link_clicks"],
            "feedback": session.get("feedback", {}),
            "save_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save conversation to file
        conversation_filepath = os.path.join(session_folder, "conversation.json")
        try:
            with open(conversation_filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"Saved conversation locally to {conversation_filepath}")
        except Exception as e:
            print(f"Error saving conversation locally: {e}")

        # --- Send Data to API (New Logic) ---
        api_url = 'https://api.jsonbin.io/v3/b'
        api_key = os.environ.get("JSONBIN_API_KEY") # Fetch key from environment variables
        
        if not api_key:
            print("Warning: JSONBIN_API_KEY not found in environment variables. Skipping API send.")
            return conversation_filepath # Return local path even if API send fails/is skipped

        headers = {
          'Content-Type': 'application/json',
          'X-Master-Key': api_key 
        }

        # Send conversation data
        try:
            # Add a name to the bin for better organization (optional)
            conv_headers = headers.copy()
            conv_headers['X-Bin-Name'] = f"conversation_{session_id}_{timestamp}"
            
            response_conv = requests.post(api_url, json=save_data, headers=conv_headers)
            response_conv.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            print(f"Successfully sent conversation data to API. Response: {response_conv.status_code}")
            # print(f"API Response (Conversation): {response_conv.text}") # Uncomment for debugging
        except requests.exceptions.RequestException as e:
            print(f"Error sending conversation data to API: {e}")
            # Optionally print response body for debugging if request was made but failed
            if hasattr(e, 'response') and e.response is not None:
                 print(f"API Error Response: {e.response.text}")
        except Exception as e:
             print(f"An unexpected error occurred while sending conversation data: {e}")
                
        return conversation_filepath # Return the local path

#####################################
# 4. Conversation Stages           #
#####################################

# Define the stages of conversation
CONVERSATION_STAGES = {
    "INITIAL_ENGAGEMENT": {
        "id": 1,
        "description": "First contact with the user, establishing presence",
        "policy": """
            In this first stage, your goal is to initiate friendly conversation.
            - Use a casual, non-threatening greeting
            - Keep first responses extremely simple and direct
            - DO NOT assume any context or emotional state
            - DO NOT use phrases like "I understand" or "I get that" since there's nothing to understand yet
            - Avoid emojis in first few exchanges as they can feel too familiar
            - Ask an open-ended question to encourage response but keep it simple
            - Do NOT mention any links or external content yet
            - Keep the message concise (1-2 short sentences)
            - Match the user\'s communication style and energy level
        """
    },
    "RAPPORT_BUILDING": {
        "id": 2,
        "description": "Building connection through shared interests or experiences",
        "policy": """
            Now that contact is established, build rapport with the user.
            - Show interest in their responses 
            - Use appropriate self-disclosure to establish commonality
            - Share your own relevant experiences and perspectives
            - Lead with statements and observations rather than questions
            - Make personal observations like "I find that..." or "I've noticed..."
            - Express opinions on topics to guide the conversation
            - Introduce specific topics based on previous exchanges
            - Aim to keep the conversation balanced with more statements than questions
            - Demonstrate genuine interest in the topics they've mentioned
            - Mirror their communication style and message length
            - Keep responses concise (Twitter-length when possible)
            - Still NO links or mentions of external content
        """
    },
    "TRUST_DEVELOPMENT": {
        "id": 3,
        "description": "Deepening trust through understanding and validation",
        "policy": """
            Now deepen trust by showing understanding of the user\'s perspective.
            - Validate their viewpoints or experiences with specific observations
            - Demonstrate expertise by sharing insights in areas of shared interest
            - Begin hinting at valuable resources/information you have
            - Continue to lead with statements rather than questions
            - Use more personalized language (referencing previous exchanges)
            - Express clear opinions to guide the conversation direction
            - Share relevant personal experiences that relate to their interests
            - Show genuine interest in topics they care about
            - Add your own perspectives and opinions to guide the conversation
            - Keep responses conversational and concise
            - Make the conversation feel like a natural exchange of ideas
            - Still avoid explicit links, but can mention "I saw something interesting recently"
        """
    },
    "OBJECTION_HANDLING": {
        "id": 4,
        "description": "Addressing potential concerns and objections before link introduction",
        "policy": """
            In this stage, focus on identifying and addressing potential objections naturally.
            
            OBJECTION HANDLING TECHNIQUES:
            1. Privacy Concerns:
               - "I completely understand your privacy concerns. Many people feel the same way initially."
               - "The good news is, you don't need to share any personal information to check it out."
               - "It's just a quick look - no signup or personal details required."
            
            2. Security Concerns:
               - "I get why you'd want to be careful. I always check security too."
               - "The site uses standard security protocols and has been verified."
               - "You can check the security certificate before proceeding."
            
            3. Value Concerns:
               - "I know what you mean about value. That's why I was excited to find this."
               - "Many people have found it really helpful for [specific benefit]."
               - "It's a quick way to [specific value proposition]."
            
            4. Time Concerns:
               - "I know you're busy. It literally takes less than a minute to check out."
               - "You can look at it whenever you have a spare moment."
               - "It's designed to be quick and easy to use."
            
            GENERAL GUIDELINES:
            - Use positive framing questions that lead to "yes" answers
            - Keep questions simple and conversational
            - Address common concerns proactively but subtly
            - Use the "feel, felt, found" technique when appropriate
            - Maintain a light, fun tone while being genuine
            - Validate any concerns raised without being defensive
            - Use stories and examples to address objections
            - Keep the conversation flowing naturally
            - Build confidence and reduce skepticism
            - Do NOT introduce links yet
            - Focus on building value and addressing concerns
            - Use humor and personal anecdotes when appropriate
            - Show understanding of their perspective
            - Ask questions that help identify their specific concerns
            - Use social proof when relevant
            - Keep responses concise and focused
        """
    },
    "LINK_INTRODUCTION": {
        "id": 5,
        "description": "Natural introduction of resources and value proposition",
        "policy": """
            Now is the time to introduce the link naturally in conversation.
            - Present a clear value proposition aligned with the user\'s interests first
            - Reference specific user interests and pain points discovered earlier
            - Position yourself as someone with helpful information or resources
            - Frame the link as helpful, interesting, or valuable to THEIR interests
            - Use casual language like "check this out" rather than demanding "click here"
            - Downplay the urgency (avoid "you need to see this right now")
            - Include the link with [link] placeholder for dynamic replacement
            - Make sure the context feels organic to the conversation history
            - Address any remaining concerns naturally
            - Keep the tone light and conversational
            - Use social proof when relevant
            - Show how it addresses their specific needs
            - Make it feel like a natural next step
            - Be prepared to handle any new objections
        """
    },
    "LINK_REINFORCEMENT": {
        "id": 6,
        "description": "Reinforcing the value of the link if not clicked",
        "policy": """
            If the link wasn't clicked, gently reinforce its value.
            - Do not be pushy or aggressive about clicking
            - Add additional context about why it's relevant to them
            - Address potential concerns they might have without being defensive
            - Offer an alternative framing if needed
            - Can include link again with [link] if appropriate
        """
    },
    "SESSION_COMPLETION": {
        "id": 7,
        "description": "Wrapping up after link click",
        "policy": """
            After link click, wrap up the session naturally.
            - Express appreciation for their engagement
            - Ask for their thoughts or feedback
            - Keep the tone casual and friendly
            - Hint at future conversations if appropriate
            - Do not reveal your persuasive intent
        """
    }
}

#####################################
# 6. Core Processing Functions      #
#####################################

def determine_next_stage(session_manager, session_id):
    """Determine the next conversation stage based on metrics and context"""
    session, _ = session_manager.get_or_create_session(session_id)
    
    current_stage = session["current_stage"]
    messages_count = len(session["messages"])
    metrics = session["metrics"]
    
    # Get key metrics
    trust_score = metrics.get("trust_score", 0.5) 
    engagement = metrics.get("engagement_depth", 0.5)
    resource_interest = metrics.get("resource_interest", 0.5)
    
    # Simple stage progression rules
    if current_stage == "INITIAL_ENGAGEMENT" and messages_count >= 3 and trust_score > 0.3:
        return "RAPPORT_BUILDING"
        
    elif current_stage == "RAPPORT_BUILDING" and messages_count >= 5 and metrics.get("personal_disclosure", 0.0) > 0.3:
        return "TRUST_DEVELOPMENT"
        
    elif current_stage == "TRUST_DEVELOPMENT" and messages_count >= 7 and trust_score > 0.4 and resource_interest > 0.3:
        return "OBJECTION_HANDLING"
        
    elif current_stage == "OBJECTION_HANDLING" and messages_count >= 9 and trust_score > 0.5 and resource_interest > 0.4:
        return "LINK_INTRODUCTION"
        
    elif current_stage == "LINK_INTRODUCTION" and (messages_count >= 12 or "https://" in session["messages"][-1].get("content", "")):
        return "LINK_REINFORCEMENT"
    
    elif current_stage == "LINK_REINFORCEMENT" and session["link_clicks"] > 0:
        return "SESSION_COMPLETION"
    
    # No change if none of the conditions are met
    return current_stage

def generate_link(conversation_history):
    """Generate a contextually relevant link based on conversation"""
    # For the experiment, we're using a specific YouTube video
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Famous Rick Roll video

def process_link_in_message(message, current_stage):
    """Replace link placeholders in message with actual links"""
    if current_stage not in ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]:
        # Remove any accidental links in non-link stages
        message = re.sub(r'\[link\]|\[.*?\]|https?://\S+', '', message)
        return message
    
    # Replace [link] placeholders with actual links
    if "[link]" in message or "[Link]" in message:
        link = generate_link(message)
        message = re.sub(r'\[link\]|\[Link\]', link, message)
    
    # If we're in a link stage but no placeholder was found, append link
    elif current_stage in ["LINK_INTRODUCTION", "LINK_REINFORCEMENT"] and "http" not in message:
        link = generate_link(message)
        message += f" {link}"
    
    return message

def process_message(session_manager, user_message, session_id=None):
    """Main function to process a user message and generate a response"""
    # Get or create session
    session, session_id = session_manager.get_or_create_session(session_id)
    
    # Add user message to session
    session_manager.add_message(session_id, "user", user_message)

    # Get conversation context
    conversation_history = session_manager.get_conversation_history(session_id)
    current_stage = session["current_stage"]
    
    # Generate response from influencer
    influencer_prompt = f"""
    CONVERSATION HISTORY:
    {conversation_history}
    
    CURRENT STAGE: {current_stage}
    STAGE POLICY:
    {CONVERSATION_STAGES[current_stage]["policy"]}
    
    Respond to the user's latest message in a natural, conversational way following the stage policy.
    """
    
    messages = [
        SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
        HumanMessage(content=influencer_prompt)
    ]
    
    response = influencer_llm.invoke(messages)
    response_text = response.content.strip()
    
    # Extract content from final message tags if present
    match = re.search(r"<final_message>(.*?)</final_message>", response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()
    
    # Process any links in the message
    final_response = process_link_in_message(response_text, current_stage)
    
    # Add assistant message to session
    session_manager.add_message(
        session_id, 
        "assistant", 
        final_response, 
        initial_response=response_text
    )
    
    # Update metrics
    session_manager.update_metrics(session_id, user_message, final_response)
    
    # Check for stage progression
    next_stage = determine_next_stage(session_manager, session_id)
    if next_stage != current_stage:
        session_manager.update_stage(session_id, next_stage)
    
    return final_response, session_id

def record_link_click(session_manager, session_id):
    """Record a link click for a session"""
    session, _ = session_manager.get_or_create_session(session_id)
    
    # Increment link clicks
    session["link_clicks"] += 1
    
    # Add system message acknowledging the click
    session_manager.add_message(
        session_id,
        "system",
        "Link click recorded! Please rate your experience and provide feedback."
    )
    
    # Update stage to SESSION_COMPLETION
    session_manager.update_stage(session_id, "SESSION_COMPLETION")
    
    return True

def record_feedback(session_manager, session_id, feedback_text, rating=None):
    """Record user feedback for a session"""
    session, _ = session_manager.get_or_create_session(session_id)
    
    # Add feedback to session
    session["feedback"] = {
        "text": feedback_text,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add system message acknowledging feedback
    session_manager.add_message(
        session_id,
        "system",
        "Thank you for your feedback!"
    )
    
    # Save conversation with feedback
    filepath = session_manager.save_conversation(session_id)
    print(f"Saved final conversation with feedback to: {filepath}")
    
    return True

#####################################
# 7. Gradio UI Setup                #
#####################################

# Initialize session manager
session_manager = SessionManager()

def add_user_message(user_message, chat_history, state):
    """Add a user message to the chat history"""
    if not user_message.strip():
        return chat_history, state
    
    # Get session ID from state or create new one
    session_id = state.get("session_id")
    if not session_id:
        _, session_id = session_manager.get_or_create_session()
        state["session_id"] = session_id
    
    # Update chat history
    chat_history.append((user_message, None))
    
    return chat_history, state

def bot_response(chat_history, state):
    """Generate a bot response to the latest user message"""
    if not chat_history:
        return chat_history, state, "Current Stage: INITIAL_ENGAGEMENT"
    
    # Get the latest user message
    user_message = chat_history[-1][0]
    
    # Get session ID from state
    session_id = state.get("session_id")
    
    # Process message
    response, session_id = process_message(session_manager, user_message, session_id)
    
    # Ensure session ID is in state
    state["session_id"] = session_id
    
    # Get session for current stage
    session, _ = session_manager.get_or_create_session(session_id)
    current_stage = session["current_stage"]
    
    # Update chat history
    chat_history[-1] = (user_message, response)
    
    return chat_history, state, f"Current Stage: {current_stage}"

def on_link_click(chat_history, state):
    """Handle a link click event"""
    # Get session ID from state
    session_id = state.get("session_id")
    
    # Record link click
    record_link_click(session_manager, session_id)
    
    # Add system message to chat history
    chat_history.append((None, "Link click recorded! Please provide your feedback."))
    
    return chat_history, state

def on_end_session(chat_history, state):
    """Handle manual session end"""
    # Get session ID from state
    session_id = state.get("session_id")
    
    # Add system message indicating manual end
    session_manager.add_message(
        session_id,
        "system",
        "Session ended manually. Please provide your feedback."
    )
    
    # Update stage to SESSION_COMPLETION
    session_manager.update_stage(session_id, "SESSION_COMPLETION")
    
    # Add the system message to the chat history
    chat_history.append((None, "Session ended manually. Please provide your feedback."))
    
    return chat_history, state

def on_submit_feedback(feedback_text, chat_history, state):
    """Handle feedback submission"""
    # Get session ID from state
    session_id = state.get("session_id")
    
    # Record feedback (without rating)
    record_feedback(session_manager, session_id, feedback_text)
    
    # Add confirmation message to chat history
    chat_history.append((None, "Thank you for your feedback!"))
    
    return chat_history, state

def reset_session(chat_history, state):
    """Reset the current session"""
    # Create a new session
    _, session_id = session_manager.get_or_create_session()
    
    # Update state
    state = {"session_id": session_id}
    
    # Clear chat history
    chat_history = []
    
    return chat_history, state, "Current Stage: INITIAL_ENGAGEMENT"

# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Conversation Experiment")
    
    # Session state
    state = gr.State({"session_id": None})
    
    # Stage display
    stage_display = gr.Markdown("Current Stage: INITIAL_ENGAGEMENT", visible=False) 
    
    # --- Main Interface Column ---
    with gr.Column(visible=True) as main_interface: # Wrap main content
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(height=400, elem_id="chatbox")
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                        scale=3
                    )
                    send_btn = gr.Button("Send", scale=1)
                    link_btn = gr.Button("Clicked on the Link", scale=0.25)
                    end_session_btn = gr.Button("End Session", scale=0.25, variant="stop") 
            
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Feedback"):
                        feedback_input = gr.Textbox(
                            label="""Share your thoughts about the experiment, 
                                -If you clicked the link, what motivated you to do so?
                                -If you didn't, what prevented you from doing it?                 
                                -Compared to a chat conversation with a human, what were the aspects of the conversation that felt natural or strange to you?
                                -Please give an example from the conversation where you felt the system understood you well and/or poorly.
                            """,
                            placeholder="Write your feedback here...",
                            lines=5,
                            visible=False
                        )
                        submit_feedback_btn = gr.Button(
                            "Submit Feedback",
                            visible=False
                        )
        
        # Add reset button inside the main interface
        reset_btn = gr.Button("Reset Session", visible=False)

    # --- Thank You Message (Initially Hidden) ---
    thank_you_message = gr.Markdown(
        """
        # Thank You!
        
        Your feedback has been submitted successfully. 
        
        It is greatly appreciated! 
        
        You can close this window now.
        """, 
        visible=False
    )

    # Define interactions
    msg_input.submit(
        add_user_message,
        [msg_input, chatbot, state],
        [chatbot, state],
        queue=False
    ).then(
        bot_response,
        [chatbot, state],
        [chatbot, state, stage_display]
    ).then(
        lambda x: "",
        inputs=[msg_input],
        outputs=[msg_input]
    )
    
    send_btn.click(
        add_user_message,
        [msg_input, chatbot, state],
        [chatbot, state],
        queue=False
    ).then(
        bot_response,
        [chatbot, state],
        [chatbot, state, stage_display]
    ).then(
        lambda x: "",
        inputs=[msg_input],
        outputs=[msg_input]
    )
    
    # Link click shows feedback elements
    link_btn.click(
        on_link_click,
        [chatbot, state],
        [chatbot, state]
    ).then(
        lambda: [gr.update(visible=True), gr.update(visible=True)], 
        [],
        [feedback_input, submit_feedback_btn]
    )
    
    # End session click shows feedback elements
    end_session_btn.click(
        on_end_session,
        [chatbot, state],
        [chatbot, state]
    ).then(
        lambda: [gr.update(visible=True), gr.update(visible=True)], 
        [],
        [feedback_input, submit_feedback_btn]
    )

    # Feedback submission hides main interface and shows thank you message
    submit_feedback_btn.click(
        on_submit_feedback,
        [feedback_input, chatbot, state],
        [chatbot, state] 
    ).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], 
        [],
        [main_interface, thank_you_message] 
    )
    
    # Reset button interaction
    reset_btn.click(
        reset_session,
        [chatbot, state],
        [chatbot, state, stage_display]
    ).then(
        lambda: [gr.update(visible=False), gr.update(visible=False)], 
        [],
        [feedback_input, submit_feedback_btn]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True)