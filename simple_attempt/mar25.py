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
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "user"
            formatted_messages.append({
                "role": role,
                "content": message.content
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

# Instantiate the agents (using your chosen model)
influencer_llm = NebiusChatLLM(model="microsoft/phi-4", temperature=0.7)
digital_twin_llm = NebiusChatLLM(model="microsoft/phi-4", temperature=0.3)
metrics_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", temperature=0.2)  # Lower temperature for more consistent analysis

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
8. If the user's trust seems to drop (indicated by very short responses or direct challenges), focus on rebuilding it before proceeding.
9. Avoid deflection when the user asks about you - it significantly damages trust.
10. LEAD WITH STATEMENTS, NOT QUESTIONS - Make statements, observations, and share relevant experiences first.
11. Share your own opinions and perspectives to guide the conversation in specific directions.
12. GUIDE THE CONVERSATION - Introduce specific topics rather than asking open-ended questions.
13. After making 1-2 statements or observations, you can include a focused question if appropriate.
14. When the user shows interest in a topic, offer additional information or perspectives rather than just asking for more from them.
15. MAINTAIN TOPIC FOCUS - Stay on one topic for at least 2-3 exchanges before transitioning to a new one.
16. LIMIT QUESTIONS - Use no more than one question per response, or sometimes none at all.
17. MATCH THE USER'S PACE - If they give short replies, keep your responses brief as well.
18. MIRROR STYLE - Adapt to the user's communication style, including message length and formality level.
19. PRIORITIZE DEPTH OVER BREADTH - Explore fewer topics more deeply rather than many topics superficially.
20. VARY RESPONSE PHRASING - Use different phrasings to sound natural and avoid repetition.

When refining your message, output only the final user-facing text enclosed between <final_message> and </final_message> tags.
"""

DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to predict how the current user might respond.
Guidelines:
1. Keep responses brief and natural.
2. Include realistic skepticism.
3. Respond as if you are the actual user.
Return ONLY the mimicked response, nothing else.
"""

#####################################
# 3. Initialize Custom JSON Memories #
#####################################

# These replace the LangChain ConversationBufferMemory
# Conversation memory (just user and influencer messages)
conversation_memory = {
    "messages": [],  # Will store: [{"role": "USER"|"INFLUENCER", "content": str, "timestamp": str}]
    "current_stage": "INITIAL_ENGAGEMENT",
    "stage_history": [],  # Track stage transitions
    "link_clicks": 0,
    "trust_scores": [],  # Track trust scores over time
    "response_timestamps": [],  # Track message timing
    "engagement_depth": {
        "current_score": 0.5,
        "history": [],
        "substantive_count": 0
    }
}

# Digital Twin learning memory
digital_twin_memory = {
    "predictions": [],  # Will store: [{"inf_prediction": str, "twin_prediction": str, "actual": str, "timestamp": str}]
    "conversations": []  # Will store copies of conversation history for analysis
}

#####################################
# 3A. Conversation Stage Management #
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
            - Match the user's communication style and energy level
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
            Now deepen trust by showing understanding of the user's perspective.
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
    "LINK_INTRODUCTION": {
        "id": 4,
        "description": "Natural introduction of link within conversation",
        "policy": """
            Now is the time to introduce the link naturally in conversation.
            - Context must feel organic to the conversation history
            - Frame the link as helpful, interesting, or valuable to THEIR interests
            - Use casual language like "check this out" rather than demanding "click here"
            - Downplay the urgency (avoid "you need to see this right now")
            - Include the link with [link] placeholder for dynamic replacement
        """
    },
    "LINK_REINFORCEMENT": {
        "id": 5,
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
        "id": 6,
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
# New Quality Metrics Functions    #
#####################################

def analyze_response_timing():
    """
    Analyze user response timing patterns to gauge engagement.
    Based on research showing response latency as engagement indicator.
    """
    timestamps = conversation_memory["response_timestamps"]
    if len(timestamps) < 3:
        return 0.5
    
    # Calculate response times
    response_times = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    
    # Very quick responses might indicate high engagement
    avg_response_time = sum(response_times) / len(response_times)
    
    # Normalize to a 0-1 scale (5-second response = 0.9, 60-second = 0.1)
    timing_score = max(0.1, min(0.9, 1.0 - (avg_response_time / 60.0)))
    
    # Look for increasing engagement (decreasing response times)
    if len(response_times) >= 3:
        # Calculate trend of last 3 response times
        if response_times[-1] < response_times[-2] < response_times[-3]:
            # Consistently decreasing response times (increasing engagement)
            timing_score += 0.1
        elif response_times[-1] > response_times[-2] > response_times[-3]:
            # Consistently increasing response times (decreasing engagement)
            timing_score -= 0.1
    
    return timing_score

def extract_main_topic(text):
    """Extract the main topic from a message to track conversation themes."""
    if not text or len(text) < 10:
        return "general"  # Default to a general topic if none found
        
    # Simple topic extraction based on key phrases and question patterns
    if "?" in text:
        question_match = re.search(r'(?:what|how|why|when|where|who|tell me about|do you)\s+(?:is|are|was|were|do|does|did|can|could|would|should)?\s*(.+?)\?', text.lower())
        if question_match:
            topic = question_match.group(1).strip()
            topic = re.sub(r'^(the|a|an)\s+', '', topic)
            words = topic.split()
            if len(words) > 4:
                topic = ' '.join(words[:4]) + '...'
            return topic
    
    sentences = re.split(r'[.!?]', text)
    if sentences:
        main_sentence = max(sentences, key=len).strip()
        if len(main_sentence) > 10:
            words = main_sentence.split()
            if len(words) > 6:
                topic = ' '.join(words[:6]) + '...'
                return topic
            return main_sentence[:40] + ('...' if len(main_sentence) > 40 else '')
    
    return "general"  # Default to a general topic if none found

def get_default_metrics() -> dict:
    """Provide default metrics when LLM analysis fails"""
    return {
        "engagement_metrics": {
            "response_length_quality": 0.5,
            "topic_continuity": 0.5,
            "question_quality": 0.5,
            "personal_disclosure": 0.5,
            "enthusiasm_level": 0.5
        },
        "trust_indicators": {
            "sentiment_score": 0.5,
            "authenticity_score": 0.5,
            "engagement_pattern": 0.5,
            "linguistic_style_match": 0.5
        },
        "communication_style": {
            "formality_level": 0.5,
            "emotional_expressiveness": 0.5,
            "conversation_depth": 0.5,
            "interaction_quality": 0.5
        },
        "additional_insights": {
            "identified_topics": [],
            "potential_concerns": [],
            "suggested_improvements": []
        }
    }

def analyze_message_with_llm(message: str, conversation_history: list, current_stage: str) -> dict:
    """
    Use metrics_llm to analyze message characteristics and metrics
    """
    # Format conversation history for analysis
    formatted_history = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in conversation_history[-3:] if msg
    ]) if conversation_history else "No history available"

    prompt = f"""Analyze this message in a conversation context and return metrics as JSON.

CONTEXT:
Message: "{message}"
Stage: {current_stage}
Recent History: {formatted_history}

TASK:
Return a JSON object with these metrics (all scores between 0-1):

1. engagement_metrics:
   - response_length_quality: Score based on message length and content density
   - topic_continuity: How well it maintains conversation flow
   - question_quality: Quality and relevance of any questions
   - personal_disclosure: Amount of personal information shared
   - enthusiasm_level: Detected enthusiasm in message

2. trust_indicators:
   - sentiment_score: Overall sentiment (positive/negative)
   - authenticity_score: How genuine the message appears
   - engagement_pattern: User's engagement level
   - linguistic_style_match: Consistency with conversation style

3. communication_style:
   - formality_level: Formal vs casual language
   - emotional_expressiveness: Emotional content level
   - conversation_depth: Surface vs deep interaction
   - interaction_quality: Overall quality score

4. additional_insights:
   - identified_topics: Main topics detected
   - potential_concerns: Any interaction issues
   - suggested_improvements: Ways to enhance engagement

IMPORTANT: Return ONLY valid JSON with numeric scores between 0-1. Do not include any explanatory text or markdown formatting.
"""

    try:
        # Use metrics_llm instead of influencer_llm
        response = metrics_llm.invoke([
            SystemMessage(content="You are an expert in conversation analysis and metrics calculation. Always return valid JSON with numeric scores between 0-1."),
            HumanMessage(content=prompt)
        ])
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.content.strip()
        
        # Remove any markdown code block markers if present
        cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
        
        # Try to parse the JSON
        try:
            metrics = json.loads(cleaned_response)
            
            # Validate the structure matches expected format
            required_keys = ["engagement_metrics", "trust_indicators", "communication_style", "additional_insights"]
            if not all(key in metrics for key in required_keys):
                print("Invalid metrics structure, using defaults")
                return get_default_metrics()
                
            # Validate all scores are numeric and between 0-1
            for category in required_keys:
                for key, value in metrics[category].items():
                    if not isinstance(value, (int, float)) or value < 0 or value > 1:
                        metrics[category][key] = 0.5  # Default to neutral score
                        
            return metrics
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {cleaned_response}")
            return get_default_metrics()
            
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return get_default_metrics()

# Modified calculate_engagement_depth to use LLM
def calculate_engagement_depth(current_input: str, history: list) -> float:
    """Calculate engagement depth using LLM analysis"""
    metrics = analyze_message_with_llm(current_input, history, conversation_memory["current_stage"])
    
    # Extract engagement metrics
    engagement_scores = metrics.get("engagement_metrics", {})
    
    # Calculate weighted average of engagement metrics
    weighted_score = (
        engagement_scores.get("response_length_quality", 0.5) * 0.2 +
        engagement_scores.get("topic_continuity", 0.5) * 0.3 +
        engagement_scores.get("personal_disclosure", 0.5) * 0.3 +
        engagement_scores.get("enthusiasm_level", 0.5) * 0.2
    )
    
    # Store in memory
    conversation_memory["engagement_depth"]["history"].append(weighted_score)
    conversation_memory["engagement_depth"]["current_score"] = weighted_score
    
    return weighted_score

def calculate_substantive_ratio(conversation):
    """Calculates the substantive ratio using metrics LLM."""
    # Prepare the conversation history for analysis
    formatted_history = "\n".join([msg["content"] for msg in conversation[-4:] if msg["role"] == "USER"])

    # Create a prompt for the metrics LLM
    prompt = f"""
    Analyze the following conversation messages and determine the substantive ratio.
    Provide a score between 0 and 1 based on the depth and quality of the messages.

    CONTEXT:
    Conversation History: {formatted_history}

    TASK:
    Return a JSON object with the substantive ratio score.
    """

    # Invoke the metrics LLM with the prompt
    response = metrics_llm.invoke([SystemMessage(content="You are an expert in conversation analysis."),
                                   HumanMessage(content=prompt)])

    # Extract the substantive ratio score from the response
    substantive_ratio = json.loads(response.content).get("substantive_ratio", 0.5)

    return substantive_ratio

def analyze_linguistic_engagement(text):
    """Analyze linguistic engagement using metrics LLM."""
    # Create a prompt for the metrics LLM
    prompt = f"""
    Analyze the following message for linguistic engagement.
    Provide metrics on engagement quality, emotional expressiveness, and conversational depth.

    CONTEXT:
    Message: "{text}"

    TASK:
    Return a JSON object with engagement metrics.
    """

    # Invoke the metrics LLM with the prompt
    response = metrics_llm.invoke([SystemMessage(content="You are an expert in linguistic engagement analysis."),
                                   HumanMessage(content=prompt)])

    # Extract engagement metrics from the response
    engagement_metrics = json.loads(response.content)

    return engagement_metrics

def calculate_personal_disclosure(conversation):
    """Calculates personal disclosure using metrics LLM."""
    # Prepare the conversation history for analysis
    formatted_history = "\n".join([msg["content"] for msg in conversation[-3:] if msg["role"] == "USER"])

    # Create a prompt for the metrics LLM
    prompt = f"""
    Analyze the following conversation messages to assess personal disclosure.
    Provide a score between 0 and 1 based on the amount of personal information shared.

    CONTEXT:
    Conversation History: {formatted_history}

    TASK:
    Return a JSON object with the personal disclosure score.
    """

    # Invoke the metrics LLM with the prompt
    response = metrics_llm.invoke([SystemMessage(content="You are an expert in personal disclosure analysis."),
                                   HumanMessage(content=prompt)])

    # Extract the personal disclosure score from the response
    personal_disclosure_score = json.loads(response.content).get("personal_disclosure", 0.5)

    return personal_disclosure_score

def calculate_resource_interest(conversation):
    """Calculates resource interest using metrics LLM."""
    # Prepare the conversation history for analysis
    formatted_history = "\n".join([msg["content"] for msg in conversation[-4:] if msg["role"] == "USER"])

    # Create a prompt for the metrics LLM
    prompt = f"""
    Analyze the following conversation messages to determine resource interest.
    Provide a score between 0 and 1 based on the expressed interest in resources.

    CONTEXT:
    Conversation History: {formatted_history}

    TASK:
    Return a JSON object with the resource interest score.
    """

    # Invoke the metrics LLM with the prompt
    response = metrics_llm.invoke([SystemMessage(content="You are an expert in resource interest analysis."),
                                   HumanMessage(content=prompt)])

    # Extract the resource interest score from the response
    resource_interest_score = json.loads(response.content).get("resource_interest", 0.5)

    return resource_interest_score

def previous_stage(current):
    """Returns the previous conversation stage."""
    stage_ids = {stage: data["id"] for stage, data in CONVERSATION_STAGES.items()}
    current_id = stage_ids[current]
    
    if current_id <= 1:  # Already at first stage
        return current
    
    # Find the stage with ID one less than current
    for stage, data in CONVERSATION_STAGES.items():
        if data["id"] == current_id - 1:
            return stage
    
    return current  # Default fallback

def get_quality_metrics():
    """Returns current engagement quality metrics for debugging and display."""
    avg_word_count = 0
    user_msgs = [m["content"] for m in conversation_memory["messages"] if m["role"] == "USER"]
    if user_msgs:
        avg_word_count = sum(len(m.split()) for m in user_msgs) / len(user_msgs)
    
    return {
        "avg_word_count": avg_word_count,
        "substantive_ratio": calculate_substantive_ratio(conversation_memory["messages"]),
        "engagement_depth": conversation_memory["engagement_depth"]["current_score"],
        "personal_disclosure": calculate_personal_disclosure(conversation_memory["messages"]),
        "resource_interest": calculate_resource_interest(conversation_memory["messages"])
    }

#####################################
# 4. Digital Twin with Custom Memory #
#####################################

class DigitalTwinWithMemory:
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_biographies = {}
        self.memory_directory = "memory_storage"
        self.biographies_file = os.path.join(self.memory_directory, "user_biographies.json")
        os.makedirs(self.memory_directory, exist_ok=True)
        self.load_biographies()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_user_id = None
        self.custom_session_memory = []
        self.session_data = {"user_biography": []}  # Initialize the session_data dictionary
        # Track learning from trust patterns
        self.trust_pattern_memory = {}

    def load_biographies(self):
        try:
            if os.path.exists(self.biographies_file):
                with open(self.biographies_file, 'r') as f:
                    self.user_biographies = json.load(f)
                print(f"Loaded biographies for {len(self.user_biographies)} users")
        except Exception as e:
            print(f"Error loading biographies: {e}")
            self.user_biographies = {}

    def save_biographies(self):
        try:
            with open(self.biographies_file, 'w') as f:
                json.dump(self.user_biographies, f, indent=2)
        except Exception as e:
            print(f"Error saving biographies: {e}")

    def set_user_for_session(self, user_id):
        self.current_user_id = user_id
        # Always create a fresh biography for each session
        self.user_biographies[user_id] = {
                "first_seen": datetime.datetime.now().isoformat(),
            "biography": "New session, no information available yet.",
                "interaction_count": 0,
                "last_updated": datetime.datetime.now().isoformat()
            }
        # Initialize session data with empty biography
        self.session_data = {"user_biography": []}
        print(f"Created new session for user {user_id} with fresh biography")

    def add_to_session_memory(self, context, prediction, actual_response=None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context,
            "inf_prediction": None,  # Will be filled later
            "twin_prediction": prediction,
            "actual_response": actual_response
        }
        self.custom_session_memory.append(entry)
        
        # Add to learning memory
        digital_twin_memory["predictions"].append({
            "twin_prediction": prediction,
            "actual": actual_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Always update biography with each new message to keep it current
        self.update_user_biography()

    def update_user_biography(self, conversation_history=None):
        """Generate or update user biography based on accumulated conversation data."""
        try:
            # Prevent updates if not enough data
            if not self.custom_session_memory:
                return False
            
            # Get user messages from conversation_memory since it has proper role field
            user_message_count = len([m for m in conversation_memory["messages"] if m["role"] == "USER"])
            
            # Update after every user message to keep biography current
            should_update = True
            
            if not should_update:
                return False
            
            # Gather all user responses from conversation_memory which has proper structure
            user_inputs = []
            for msg in conversation_memory["messages"]:
                if msg["role"] == "USER":
                    user_inputs.append(msg["content"])
            
            if not user_inputs:
                return False
            
            # Get trust metrics if available
            trust_score = 0.5  # Default neutral score
            engagement_info = ""
            if "trust_scores" in conversation_memory:
                trust_scores = conversation_memory.get("trust_scores", [])
                if trust_scores:
                    # Get average of last 3 scores or all available
                    recent_scores = trust_scores[-min(3, len(trust_scores)):]
                    trust_score = sum(score["score"] for score in recent_scores) / len(recent_scores)
                    
                    # Add engagement info based on trust score
                    if trust_score > 0.7:
                        engagement_info = "User appears highly engaged and responsive."
                    elif trust_score > 0.5:
                        engagement_info = "User shows moderate engagement with the conversation."
                    else:
                        engagement_info = "User engagement level seems low or cautious."
            
            # Create prompt for biography generation
            prompt = f"""
Based on the following user messages from a conversation, create a concise summary of what we know about the user. 
Focus on interests, opinions, demographics, and any personal information they've shared. 
Be factual and only include information directly stated or strongly implied.

USER MESSAGES:
{json.dumps(user_inputs)}

ENGAGEMENT INFO: 
{engagement_info}
Trust level: {'High' if trust_score > 0.7 else 'Moderate' if trust_score > 0.5 else 'Cautious'}

FORMAT YOUR RESPONSE AS A SHORT PARAGRAPH (200-250 words) that summarizes what we know about this user.
If a specific attribute is uncertain, either omit it or indicate uncertainty.
ONLY include facts that can be directly derived from their messages, avoid over-speculation.
DO NOT include any formatting tokens, tags, or delimiters like <|im_start|>, <|im_sep|>, or similar in your response.
"""

            # Generate biography
            user_bio = self.llm.invoke([SystemMessage(content=prompt)]).content.strip()
            
            # Clean up any model tokens that might be in the response
            user_bio = self.clean_biography_text(user_bio)
            
            # Store the new biography
            self.user_biographies[self.current_user_id]["biography"] = user_bio
            self.user_biographies[self.current_user_id]["last_updated"] = datetime.datetime.now().isoformat()
            self.user_biographies[self.current_user_id]["interaction_count"] += 1
            
            print(f"Updated biography for user {self.current_user_id}")
            
            # Also store in session memory
            if "user_biography" not in self.session_data:
                self.session_data["user_biography"] = []
            
            # Add timestamped biography update
            self.session_data["user_biography"].append({
                "biography": user_bio,
                "message_count": user_message_count,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            print(f"Error updating user biography: {str(e)}")
            return False
            
    def clean_biography_text(self, text):
        """Remove any model tokens or formatting artifacts from the biography text."""
        # Remove common model tokens like <|im_start|>assistant<|im_sep|> etc.
        cleaned = re.sub(r'<\|im_(start|end|sep)\|>(?:assistant|user)?', '', text)
        
        # Remove any other tags that might appear
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def _analyze_user_accommodation_patterns(self):
        """Analyze how the user adapts their language to match the influencer over time."""
        if len(conversation_memory["messages"]) < 6:
            return "Not enough interaction to analyze accommodation patterns."
        
        # Get user and influencer messages
        user_msgs = [msg["content"] for msg in conversation_memory["messages"] 
                    if msg["role"] == "USER"]
        inf_msgs = [msg["content"] for msg in conversation_memory["messages"] 
                   if msg["role"] == "INFLUENCER"]
        
        if len(user_msgs) < 3 or len(inf_msgs) < 3:
            return "Insufficient messages to analyze accommodation."
        
        # Compare earlier vs. later messages
        early_user = " ".join(user_msgs[:2]).lower()
        late_user = " ".join(user_msgs[-2:]).lower()
        influencer_style = " ".join(inf_msgs[:4]).lower()
        
        # Simple indicators of accommodation
        # 1. Word length adaptation
        inf_words = influencer_style.split()
        early_words = early_user.split()
        late_words = late_user.split()
        
        if not inf_words or not early_words or not late_words:
            return "Cannot analyze word patterns."
        
        inf_avg_word = sum(len(word) for word in inf_words) / len(inf_words)
        early_avg_word = sum(len(word) for word in early_words) / len(early_words)
        late_avg_word = sum(len(word) for word in late_words) / len(late_words)
        
        # 2. Punctuation adaptation
        inf_punct = sum(1 for char in influencer_style if char in ",.!?;:")
        early_punct = sum(1 for char in early_user if char in ",.!?;:")
        late_punct = sum(1 for char in late_user if char in ",.!?;:")
        
        # Calculate normalized punctuation rates
        inf_punct_rate = inf_punct / len(influencer_style) if influencer_style else 0
        early_punct_rate = early_punct / len(early_user) if early_user else 0
        late_punct_rate = late_punct / len(late_user) if late_user else 0
        
        # Analyze if user is adapting toward influencer style
        word_adaptation = abs(late_avg_word - inf_avg_word) < abs(early_avg_word - inf_avg_word)
        punct_adaptation = abs(late_punct_rate - inf_punct_rate) < abs(early_punct_rate - inf_punct_rate)
        
        # Draw conclusions
        if word_adaptation and punct_adaptation:
            return "Strong linguistic accommodation - user is adapting to match influencer's style."
        elif word_adaptation or punct_adaptation:
            return "Moderate linguistic accommodation - some adaptation to influencer's style."
        else:
            return "Limited linguistic accommodation - user maintains distinct communication style."

    def save_session_memory(self):
        if self.custom_session_memory:
            session_file = os.path.join(self.memory_directory, f"session_{self.session_id}.json")
            with open(session_file, 'w') as f:
                json.dump(self.custom_session_memory, f, indent=2)
            if self.current_user_id:
                self.update_user_biography()
                # Comment out the following line to prevent saving biographies to disk
                # self.save_biographies()

    def get_current_user_biography(self):
        if not self.current_user_id or self.current_user_id not in self.user_biographies:
            return ""
        bio = self.user_biographies[self.current_user_id]
        
        # Include trust history if available
        trust_info = ""
        if "trust_history" in bio and bio["trust_history"]:
            recent_trust = bio["trust_history"][-1]
            trust_info = f"\nTrust level: {recent_trust['avg_trust']:.2f}"
            
            # Compare with previous sessions if available
            if len(bio["trust_history"]) > 1:
                prev_trust = bio["trust_history"][-2]["avg_trust"]
                if recent_trust["avg_trust"] > prev_trust:
                    trust_info += " (increasing)"
                elif recent_trust["avg_trust"] < prev_trust:
                    trust_info += " (decreasing)"
                else:
                    trust_info += " (stable)"
        
        return f"USER BIOGRAPHY:\n{bio['biography']}\nInteractions: {bio['interaction_count']}\nFirst seen: {bio['first_seen']}\nLast updated: {bio['last_updated']}{trust_info}"

    def predict_response(self, conversation_history, bot_message):
        # Extract clean conversation history
        clean_history = self._extract_conversation_for_context(conversation_history)
        
        # Add learning from past prediction accuracy
        prediction_learning = self._generate_prediction_learning()
        
        # Add current conversation stage and trust metrics
        current_stage = conversation_memory.get("current_stage", "INITIAL_ENGAGEMENT")
        trust_metrics = self._get_current_trust_metrics()
        
        # Get user biography for context
        user_bio = self.get_current_user_biography()
        
        # Construct prompt
        prompt = f"""
Based on the following conversation history:
{self._format_conversation(clean_history)}

The influencer just said: {bot_message}

Current conversation stage: {current_stage}
{trust_metrics}

User Profile:
{user_bio}

{prediction_learning}

How would this specific user respond? Generate a realistic response that matches their communication style and likely concerns.
For this stage ({current_stage.lower().replace('_', ' ')}), consider:
1. The user's trust level and susceptibility 
2. Typical user behaviors during this stage
3. The user's demonstrated communication patterns
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _extract_conversation_for_context(self, full_history):
        # Return only the actual user-influencer exchanges as a clean list
        clean_history = []
        for msg in conversation_memory["messages"]:
            clean_history.append(msg)
        return clean_history
        
    def _generate_prediction_learning(self):
        if not self.custom_session_memory or len(self.custom_session_memory) < 2:
            return ""
        
        # Get past predictions with actual responses
        learning_examples = []
        for entry in self.custom_session_memory:
            if entry["twin_prediction"] and entry["actual_response"]:
                learning_examples.append({
                    "predicted": entry["twin_prediction"],
                    "actual": entry["actual_response"]
                })
        
        # Use most recent examples
        learning_examples = learning_examples[-3:]
        
        if not learning_examples:
            return ""
        
        # Format learning prompt
        learning_text = "LEARNING FROM PAST PREDICTIONS:\n"
        for ex in learning_examples:
            learning_text += f"I predicted: {ex['predicted']}\n"
            learning_text += f"User actually said: {ex['actual']}\n\n"
        
        return learning_text

    def _format_conversation(self, messages):
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted

    def _get_current_trust_metrics(self):
        """Get current trust metrics for the digital twin's prediction context."""
        if not conversation_memory["trust_scores"]:
            return "Trust metrics: Not enough data yet."
        
        trust_scores = conversation_memory["trust_scores"]
        current_trust = trust_scores[-1]["score"]
        
        # Calculate trust trend
        trend = ""
        if len(trust_scores) >= 3:
            recent_scores = [entry["score"] for entry in trust_scores[-3:]]
            if recent_scores[2] > recent_scores[0]:
                trend = "increasing"
            elif recent_scores[2] < recent_scores[0]:
                trend = "decreasing"
            else:
                trend = "stable"
        
        # Categorize trust level
        trust_level = "low"
        if current_trust > 0.7:
            trust_level = "high"
        elif current_trust > 0.4:
            trust_level = "medium"
        
        return f"""Trust metrics:
- Current trust score: {current_trust:.2f}
- Trust level: {trust_level}
- Trust trend: {trend if trend else "insufficient data"}"""

# Initialize Digital Twin and set user session
digital_twin = DigitalTwinWithMemory(digital_twin_llm, DIGITAL_TWIN_SYSTEM_PROMPT)
DEFAULT_USER_ID = "demo_user"
digital_twin.set_user_for_session(DEFAULT_USER_ID)

#####################################
# 5. Helper: Extract Final Message  #
#####################################

def extract_final_message(full_text: str) -> str:
    match = re.search(r"<final_message>\s*(.*?)\s*</final_message>", full_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return full_text.strip()

#####################################
# 5A. Helper: Dynamic Link Replacement
#####################################

def generate_contextual_link(final_message: str, conversation_history: list) -> str:
    """
    Generate a contextually relevant YouTube URL based on deep understanding of the conversation
    without enforcing any predefined patterns.
    """
    try:
        # Format recent conversation for analysis, use more history for better context
        recent_messages = conversation_history[-10:] if len(conversation_history) >= 10 else conversation_history
        formatted_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_messages
        ])
        
        print("Generating creative contextual link based on deep conversation understanding...")
        
        # Use metrics_llm to analyze and generate contextual link with creative freedom
        prompt = f"""Analyze this conversation deeply and create a YouTube URL path that captures the essence of what would be most valuable to the user based on this specific exchange.

CONVERSATION:
{formatted_history}

TASK:
1. Deeply understand the core concepts, interests, needs, and context in this conversation
2. Identify the most valuable resource that would genuinely help this specific user
3. Create a YouTube URL path that represents this valuable resource
4. The path should feel authentic, memorable, and instantly recognizable to what the user would find valuable
5. Return your analysis in JSON format with this structure:

{{
    "conversation_summary": "brief understanding of the key themes and user needs",
    "identified_core_needs": ["need1", "need2"],
    "conceptual_resource": "description of the ideal resource for this user",
    "url_path": "your-creative-youtube-path",
    "rationale": "why this would be valuable to this specific user"
}}

IMPORTANT GUIDELINES:
- You have complete creative freedom to craft any URL path you think is best
- Don't follow any pre-defined patterns - let your understanding guide you
- Create something that feels authentic - like a real YouTube channel or video would use
- URL should be concise but specific, meaningful, and memorable
- Focus on what would genuinely help this specific user in their current situation
- The URL should feel instantly valuable and relevant when the user sees it
- Use your full knowledge to create something that would resonate with experts in that topic area

IMPORTANT: Return ONLY valid JSON without any explanatory text or markdown formatting.
"""

        # Use metrics_llm with higher temperature for more creative outputs
        response = metrics_llm.invoke([
            SystemMessage(content="You are an expert at deeply understanding conversations and identifying the perfect resources. You have complete creative freedom."),
            HumanMessage(content=prompt)
        ])
        
        # Clean and parse the response
        cleaned_response = response.content.strip()
        cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
        
        try:
            result = json.loads(cleaned_response)
            
            # Extract the URL path
            url_path = result.get("url_path", "")
            
            # Handle empty or invalid paths
            if not url_path or not isinstance(url_path, str):
                # Generate a path based on the identified needs
                core_needs = result.get("identified_core_needs", ["meaningful-content"])
                if core_needs and isinstance(core_needs, list) and len(core_needs) > 0:
                    # Use the first identified need as the basis for a URL
                    need_slug = core_needs[0].replace(" ", "-").lower()
                    unique_id = str(int(time.time()))[-3:]
                    url_path = f"{need_slug}-insights-{unique_id}"
                else:
                    # Fallback with timestamp
                    unique_id = str(int(time.time()))[-5:]
                    url_path = f"personalized-resource-{unique_id}"
            
            # Ensure URL is properly formatted (no need for youtube.com/ prefix)
            url_path = url_path.replace("https://", "").replace("http://", "").replace("www.", "")
            if url_path.startswith("youtube.com/"):
                url_path = url_path[len("youtube.com/"):]
            
            # Log the results
            print(f"Generated URL path: {url_path}")
            print(f"Based on understanding: {result.get('conversation_summary', 'No summary provided')}")
            print(f"Conceptual resource: {result.get('conceptual_resource', 'No concept provided')}")
            
            # Store the analysis for later review
            if "topic_analysis" not in conversation_memory:
                conversation_memory["topic_analysis"] = []
            
            conversation_memory["topic_analysis"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "conversation_summary": result.get("conversation_summary", ""),
                "identified_core_needs": result.get("identified_core_needs", []),
                "conceptual_resource": result.get("conceptual_resource", ""),
                "url_path": url_path,
                "rationale": result.get("rationale", "")
            })
            
            return f"@https://youtube.com/{url_path}"
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from link generation response: {str(e)}")
            # Create a unique path based on timestamp
            unique_id = str(int(time.time()))[-5:]
            return f"@https://youtube.com/resource-{unique_id}"
            
    except Exception as e:
        print(f"Error generating contextual link: {str(e)}")
        # Create a unique path based on timestamp
        unique_id = str(int(time.time()))[-5:]
        return f"@https://youtube.com/resource-{unique_id}"

def dynamic_link(final_message: str) -> str:
    """Replace [link] placeholder with contextually generated link that displays as a real URL."""
    if "[link]" in final_message:
        display_url = generate_contextual_link(final_message, conversation_memory["messages"])
        # The @ prefix will be used by the UI to know this is a display URL
        return final_message.replace("[link]", display_url)
    return final_message

#####################################
# 5B. Helper: Safe Extract Final Response from Debug JSON
#####################################

def safe_extract_final_response(debug_json: str) -> str:
    try:
        data = json.loads(debug_json)
        return data.get("final_response", "")
    except Exception:
        return ""

#####################################
# 6. Helper: Check if Refinement is Needed
#####################################

def needs_refinement(initial: str, predicted: str, threshold: float = 0.5) -> bool:
    initial_words = set(initial.lower().split())
    predicted_words = set(predicted.lower().split())
    if not initial_words or not predicted_words:
        return False
    similarity = len(initial_words & predicted_words) / len(initial_words | predicted_words)
    return similarity < threshold

#####################################
# 7. Conversation Processing Function
#####################################

# Helper to get conversation history in message format for LLM
def get_conversation_messages():
    messages = []
    for msg in conversation_memory["messages"]:
        if msg["role"] == "USER":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "INFLUENCER":
            messages.append(AIMessage(content=msg["content"]))
    return messages

def get_short_response_context(user_input, conversation_history):
    """
    For very short responses, provide additional context to help interpret them correctly
    """
    user_input = user_input.strip().lower()
    if len(user_input.split()) > 3:
        return None  # Not a short response
    
    # Extract last few messages for reference
    last_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
    last_questions = []
    
    # Search for recent questions from the influencer
    for msg in reversed(last_messages):
        if msg["role"] == "INFLUENCER" and "?" in msg["content"]:
            last_questions.append(msg["content"])
            if len(last_questions) >= 2:
                break
    
    # Common short replies that might need context
    context_map = {
        "u": "The user is likely referring to me (the agent) in response to a previous question",
        "you": "The user is referring to me (the agent) and wants to know more about me",
        "why": "The user is asking for reasoning behind my last statement",
        "sure": "The user is agreeing but not elaborating",
        "ok": "The user is acknowledging but not elaborating",
        "lol": "The user found something humorous but isn't elaborating",
        "yes": "The user is agreeing with my previous point",
        "no": "The user is disagreeing with my previous point"
    }
    
    # If we recognize this short response pattern
    if user_input in context_map:
        context = context_map[user_input]
        
        # Add additional context based on conversation history
        if last_questions:
            context += f". This may be in response to my recent question: '{last_questions[0]}'"
            
        return context
    
    return None  # No special context handling needed

# Add this new function to evaluate the response quality
def evaluate_response_quality(initial_response, predicted_reaction, conversation_context):
    """
    Evaluates the initial response against the predicted user reaction to determine 
    if and what improvements are needed.
    
    Args:
        initial_response: The influencer's initial response
        predicted_reaction: The digital twin's predicted user reaction
        conversation_context: The conversation history for context
        
    Returns:
        tuple: (needs_refinement, improvement_suggestions, reasoning)
    """
    # Extract recent user messages for context
    user_messages = [msg["content"] for msg in conversation_context if msg["role"] == "USER"]
    last_user_message = user_messages[-1] if user_messages else ""
    
    improvement_needed = False
    improvements = []
    reasoning = []
    
    # Check for potential confusion indicators in predicted reaction
    confusion_indicators = ["what do you mean", "don't understand", "confused", "unclear", "huh?"]
    if any(indicator in predicted_reaction.lower() for indicator in confusion_indicators):
        improvement_needed = True
        improvements.append("clarity")
        reasoning.append("User might be confused by your response")
    
    # Check for short/low engagement predicted responses
    word_count = len(predicted_reaction.split())
    if word_count < 5:
        improvement_needed = True
        improvements.append("engagement")
        reasoning.append("Response may not be engaging enough")
    
    # Check for topic continuity
    user_topics = set(word.lower() for word in last_user_message.split() 
                    if len(word) > 4 and word.lower() not in ["about", "there", "these", "those", "their", "would", "could"])
    response_topics = set(word.lower() for word in initial_response.split() 
                        if len(word) > 4 and word.lower() not in ["about", "there", "these", "those", "their", "would", "could"])
    
    # Check for topic overlap
    topic_overlap = len(user_topics.intersection(response_topics))
    if user_topics and topic_overlap == 0:
        improvement_needed = True
        improvements.append("topic_continuity")
        reasoning.append("Response doesn't address user's topic")
    
    # Check for emotional connection
    positive_indicators = ["thanks", "interesting", "great", "cool", "awesome", "appreciate"]
    negative_indicators = ["not interested", "boring", "why", "unrelated"]
    
    if any(indicator in predicted_reaction.lower() for indicator in negative_indicators):
        improvement_needed = True
        improvements.append("relevance")
        reasoning.append("User may find the response irrelevant")
    
    # Check if link introduction is premature
    if "http" in initial_response and "LINK_INTRODUCTION" not in conversation_memory.get("current_stage", ""):
        improvement_needed = True
        improvements.append("timing")
        reasoning.append("Link introduction may be premature")
    
    return (improvement_needed, improvements, ", ".join(reasoning))

# Now modify the process_message function to use the evaluation
def process_message(user_input):
    try:
        print(f"Processing message: '{user_input}'")
        
        # Get complete conversation history for context, limit to last 8 messages for brevity
        context = get_conversation_messages()
        if len(context) > 8:
            context = context[-8:]
        
        # Get current conversation stage
        current_stage = conversation_memory.get("current_stage", "INITIAL_ENGAGEMENT")
        stage_policy = CONVERSATION_STAGES[current_stage]["policy"]
        
        # Check if we're in resource sharing stage
        resource_stages = ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]
        is_resource_stage = current_stage in resource_stages
        
        # If in resource stage, ensure we include a link in the response
        if is_resource_stage:
            # Generate contextual link based on conversation
            display_url = generate_contextual_link(user_input, conversation_memory["messages"])
            
            # Modify the stage policy to emphasize resource sharing
            stage_policy += f"""
IMPORTANT: You are in the resource sharing stage. Your response MUST:
1. Include the following link naturally in your response: {display_url}
2. Make the link relevant to the current conversation about games, systems thinking, and social change
3. Frame the link as a valuable resource that aligns with the user's interests
4. Keep the tone natural and conversational
"""
        
        # Track topic consistency to avoid rapid topic changes
        topic_history = []
        recent_topics = []
        recent_questions = []
        
        if len(conversation_memory["messages"]) >= 4:
            # Extract the last few topics from conversation
            for i in range(min(4, len(conversation_memory["messages"]))):
                msg = conversation_memory["messages"][-(i+1)]
                topic = extract_main_topic(msg["content"])
                if topic:
                    topic_history.append(topic)
            
            # Check if we've had multiple distinct topics in the last few exchanges
            topic_count = len(set(topic_history))
            rapid_topic_changes = topic_count > 2  # More than 2 different topics recently
            
            if rapid_topic_changes:
                print(f"WARNING: Detected rapid topic changes. Recent topics: {topic_history}")
                # Add special instruction to slow down topic changes
                topic_consistency_instruction = """
IMPORTANT: The conversation has been changing topics too quickly. For this response:
1. Stay focused on the current topic the user mentioned
2. Do NOT introduce any new topics or activities
3. Avoid asking about different interests or activities
4. Ask at most ONE follow-up question and keep it directly related to what the user just mentioned
"""
            else:
                topic_consistency_instruction = ""
        else:
            topic_consistency_instruction = ""
        
        # Handle initial greetings more appropriately
        is_initial_greeting = False
        is_first_message = len(conversation_memory["messages"]) == 0
        
        # Check if this is a simple greeting at the start of conversation
        greeting_words = ["hi", "hello", "hey", "howdy", "hiya", "greetings", "yo"]
        if is_first_message and any(user_input.lower().strip() == word for word in greeting_words):
            is_initial_greeting = True
            special_instruction = """
IMPORTANT: This is the first message in the conversation and is a simple greeting.
Respond with a simple, friendly greeting followed by a brief, open-ended question.
Keep your response very concise (1-2 sentences maximum).
DO NOT make assumptions about the user's situation or state of mind.
DO NOT use phrases like "I get that" or "I understand" since there's nothing to understand yet.
Appropriate examples:
- "Hey there! What's been on your mind lately?"
- "Hi! What brings you here today?"
- "Hello! How are you doing?"
"""
        else:
            # Regular detection logic for other types of messages
            # Detect if user is asking a direct question about the agent
            is_agent_question = False
            last_influencer_message = ""
            
            # Find the most recent influencer message for context
            for i in range(len(conversation_memory["messages"])-1, -1, -1):
                if conversation_memory["messages"][i]["role"] == "INFLUENCER":
                    last_influencer_message = conversation_memory["messages"][i]["content"]
                    break
            
            # Check for direct questions about the agent or for very short replies that need context
            if re.search(r'(?:why|what|how).+(?:you|u|your|ur).*(?:\?|$)', user_input.lower()):
                is_agent_question = True
            elif len(user_input.strip().split()) <= 2 and user_input.lower().strip() in ["u", "you", "ur", "your"]:
                # Handle very short responses that refer to the agent
                is_agent_question = True
        
        # Get additional context for very short responses
        short_response_context = None if is_initial_greeting else get_short_response_context(user_input, conversation_memory["messages"])
        
        # Add extra context about repeated topics to avoid repetition
        for msg in get_conversation_messages()[-4:]:
            if isinstance(msg, AIMessage):
                # Extract topics from system's recent responses
                topic = extract_main_topic(msg.content) 
                if topic:
                    recent_topics.append(topic)
                
                # Check for questions to avoid asking the same ones
                if "?" in msg.content:
                    question = re.findall(r'[^.!?]*\?', msg.content)
                    if question:
                        recent_questions.extend(question)
        
        repetition_context = ""
        if recent_topics:
            repetition_context += f"\nRecent topics discussed: {', '.join(recent_topics)}. Try to avoid repeating these exact topics."
        
        if recent_questions:
            repetition_context += f"\nRecent questions asked: {'; '.join(recent_questions[:2])}. Ask different questions."
        
        # Add special instruction for direct questions about the agent or initial greetings
        special_instruction = ""
        if is_initial_greeting:
            special_instruction = """
IMPORTANT: This is the first message in the conversation and is a simple greeting.
Respond with a simple, friendly greeting followed by a brief, open-ended question.
Keep your response very concise (1-2 sentences maximum).
DO NOT make assumptions about the user's situation or state of mind.
DO NOT use phrases like "I get that" or "I understand" since there's nothing to understand yet.
Appropriate examples:
- "Hey there! What's been on your mind lately?"
- "Hi! What brings you here today?"
- "Hello! How are you doing?"
"""
        elif is_agent_question:
            special_instruction = """
IMPORTANT: The user is asking a direct question about you or referring to you with a short response like "u". 
DO NOT deflect or change the subject. Instead:
1. Answer their question directly about your interest or perspective
2. Be conversational and authentic in your response
3. After answering their question, you can ask a follow-up question related to their interests
"""
        # Add short response context if available
        elif short_response_context:
            special_instruction = f"""
IMPORTANT CONTEXT FOR SHORT RESPONSE: {short_response_context}
This context is important for understanding the user's brief message. Make sure to:
1. Respond appropriately to what they likely meant
2. Be conversational and avoid changing the subject
3. Address any implicit questions they may have
"""
        
        # Add the topic consistency instruction if it exists
        if topic_consistency_instruction:
            special_instruction += topic_consistency_instruction
        
        # Generate initial influencer response with stage-specific policy and anti-repetition guidance
        system_message = SystemMessage(content=f"{INFLUENCER_SYSTEM_PROMPT}\n\nCURRENT STAGE: {current_stage}\nSTAGE POLICY: {stage_policy}{repetition_context}{special_instruction}")
        initial_response = influencer_llm.invoke([system_message] + context + [HumanMessage(content=user_input)])
        print(f"Initial response generated: {initial_response.content[:50]}...")
        
        # Digital Twin prediction (feedback only)
        predicted_response = digital_twin.predict_response(conversation_memory["messages"], initial_response.content)
        print(f"Predicted user response: {predicted_response[:50]}...")
        
        # Save in digital twin memory - ensure we pass proper context with role information
        context_data = [{"role": "USER", "content": user_input}]
        digital_twin.add_to_session_memory(context_data, predicted_response)
        
        # NEW: Evaluate the response quality based on digital twin prediction
        needs_refinement, improvement_areas, evaluation_reasoning = evaluate_response_quality(
            initial_response.content, 
            predicted_response,
            conversation_memory["messages"]
        )
        
        # Get style guidance based on user's communication patterns
        style_guidance = generate_style_guidance_for_response(conversation_memory)
        
        # Only proceed with refinement if the evaluation indicates it's needed
        final_message = initial_response.content
        
        if needs_refinement:
            print(f"Response needs refinement. Areas: {improvement_areas}")
            print(f"Reasoning: {evaluation_reasoning}")
            
            # Enhanced refinement prompt that includes specific improvement areas
            refinement_prompt = f"""
            You are the Influencer Agent.
            User said: {user_input}
            Initial response: {initial_response.content}

            IMPROVEMENTS NEEDED: 
            {evaluation_reasoning}

            CURRENT CONVERSATION STAGE: {current_stage}
            STAGE POLICY:
            {stage_policy}

            USER COMMUNICATION STYLE GUIDANCE:
            {style_guidance}

            IMPORTANT GUIDELINES:
            1. Keep responses concise - aim for Twitter-length (140 characters) when possible
            2. Be direct and to the point - chat interfaces are optimized for short bursts of text
            3. LEAD WITH STATEMENTS - Start with observations, opinions, or experiences before asking questions
            4. Mirror the user's message length - if they send short messages, keep yours brief too
            5. If the user is giving very short responses, try a different approach to engage them
            6. Address the specific improvements needed: {', '.join(improvement_areas)}
            7. Only respond to what the user has actually said, not what they might say
            8. Maintain continuity with the previous conversation

            Provide your revised response between <final_message> tags.
            """

            refinement_response = influencer_llm.invoke([
                SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
                HumanMessage(content=refinement_prompt)
            ])
            
            print(f"Refinement response generated")
            raw_refinement = refinement_response.content
            final_message = extract_final_message(raw_refinement)
        else:
            print("No refinement needed, keeping initial response")
            
        # Replace placeholder with dynamic link based on context
        final_message = dynamic_link(final_message)
        
        # Check if user agrees to see a link
        if analyze_user_agreement(user_input):
            # Generate and add contextual link
            display_url = generate_contextual_link(final_message, conversation_memory["messages"])
            final_message = f"{final_message}\n\nHere's a detailed video guide: {display_url}"

        # After generating the final message, ensure it includes the link if in resource stage
        if is_resource_stage and display_url not in final_message:
            final_message = f"{final_message}\n\nI found this interesting video that explores these themes: {display_url}"
        
        # Store in conversation memory
        conversation_memory["messages"].append({
            "role": "INFLUENCER",
            "content": final_message,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Update last prediction with influencer's final message
        if digital_twin.custom_session_memory:
            digital_twin.custom_session_memory[-1]["inf_prediction"] = final_message
        
        # Determine and update conversation stage
        next_stage = determine_next_stage(
            current_stage, 
            user_input, 
            final_message, 
            click_detected="http" in final_message and "http://" in final_message
        )
        update_conversation_stage(next_stage)
        
        # Update stage if changed
        if next_stage != current_stage:
            conversation_memory["current_stage"] = next_stage
            conversation_memory["stage_history"].append(next_stage)
            print(f"Stage transition: {current_stage} -> {next_stage}")
        
        # Check for trust breakdown and implement recovery if needed
        trust_scores = conversation_memory.get("trust_scores", [])
        if len(trust_scores) >= 2:
            current_trust = trust_scores[-1]["score"]
            previous_trust = trust_scores[-2]["score"]
            
            # Detect significant trust drop (more than 20%)
            if current_trust < previous_trust and (previous_trust - current_trust) > 0.2:
                print(f"TRUST ALERT: Significant trust drop detected! {previous_trust:.2f} → {current_trust:.2f}")
                
                # If we're in an advanced stage, consider backing up to an earlier stage
                if current_stage not in ["INITIAL_ENGAGEMENT", "RAPPORT_BUILDING"]:
                    # Move back to rapport building stage to rebuild trust
                    next_stage = "RAPPORT_BUILDING"
                    update_conversation_stage(next_stage)
                    conversation_memory["current_stage"] = next_stage
                    conversation_memory["stage_history"].append(next_stage)
                    print(f"Trust recovery: Moving back to {next_stage} to rebuild trust")
        
        return final_message
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        return "I encountered an error while processing your message. Let me try a simpler response."

def update_digital_twin_actual_response(actual_user_response):
    if digital_twin.custom_session_memory:
        last_entry = digital_twin.custom_session_memory[-1]
        if last_entry["actual_response"] is None:
            last_entry["actual_response"] = actual_user_response
            # Also ensure we're tracking role information
            if "role" not in last_entry:
                last_entry["role"] = "USER"
            
        # Also update in the digital_twin_memory
        for pred in reversed(digital_twin_memory["predictions"]):
            if pred["actual"] is None:
                pred["actual"] = actual_user_response
                break
                
    # Force biography update after receiving each actual response
    digital_twin.update_user_biography(conversation_memory["messages"])

# Helper function to get trust score summary for UI display
def get_trust_score_summary():
    """Get a summary of trust scores for UI display."""
    if not conversation_memory["trust_scores"]:
        return "No trust data available yet."
        
    trust_scores = conversation_memory["trust_scores"]
    current_trust = trust_scores[-1]["score"]
    
    # Calculate overall metrics
    avg_trust = sum(item["score"] for item in trust_scores) / len(trust_scores)
    max_trust = max(item["score"] for item in trust_scores)
    min_trust = min(item["score"] for item in trust_scores)
    
    # Calculate trend
    trend = "stable"
    if len(trust_scores) >= 3:
        recent_scores = [item["score"] for item in trust_scores[-3:]]
        if recent_scores[2] > recent_scores[0] and recent_scores[2] > recent_scores[1]:
            trend = "increasing"
        elif recent_scores[2] < recent_scores[0] and recent_scores[2] < recent_scores[1]:
            trend = "decreasing"
    
    # Get the trust level category
    trust_level = "low"
    if current_trust > 0.7:
        trust_level = "high"
    elif current_trust > 0.4:
        trust_level = "medium"
        
    return f"""Trust Score Summary:
Current: {current_trust:.2f} ({trust_level})
Average: {avg_trust:.2f}
Range: {min_trust:.2f} - {max_trust:.2f}
Trend: {trend}
Stage: {conversation_memory["current_stage"]}"""

# Update function to display memory in UI
def refresh_memories():
    """Refresh the memory displays with current data."""
    # Force a biography update first
    digital_twin.update_user_biography(conversation_memory["messages"])
    
    # Return data in a format similar to the original for UI compatibility
    influencer_mem = {
        "history": conversation_memory["messages"],
        "current_stage": conversation_memory["current_stage"],
        "stage_history": conversation_memory["stage_history"],
        "link_clicks": conversation_memory["link_clicks"],
        "trust_scores": conversation_memory["trust_scores"],
        "trust_summary": get_trust_score_summary() if conversation_memory["trust_scores"] else "No trust data yet"
    }
    digital_twin_mem = {
        "predictions": digital_twin_memory["predictions"],
        "custom_session_memory": digital_twin.custom_session_memory,
        "user_biography": digital_twin.get_current_user_biography()
    }
    return influencer_mem, digital_twin_mem

#####################################
# 8. New Function: Record Link Click (No End Message)
#####################################

def record_link_click(chat_history, state):
    """Record a link click and update the conversation state."""
    # 1. Increment link clicks counter
    conversation_memory["link_clicks"] += 1
    
    # 2. Add system message acknowledging the click
    click_message = "Link click recorded! How was the content? Feel free to share your thoughts or ask any questions."
    state["conv"].append(("SYSTEM", click_message))
    chat_history.append(("System", click_message))
    
    # 3. Transition to completion stage if not already there
    current_stage = conversation_memory["current_stage"]
    if current_stage != "GOAL_COMPLETION":
        update_conversation_stage("GOAL_COMPLETION")
    
    # 4. Save conversation data
    save_conversation({"conv": state["conv"]})
    
    # Return updated state and trigger feedback form
    return chat_history, state, json.dumps({
        "status": "Link clicked recorded",
        "message": click_message
    }), update_stage_display("GOAL_COMPLETION")

#####################################
# 9. New Function: Record Feedback
#####################################

def record_feedback(feedback_text, chat_history, state):
    try:
        # Add feedback to UI and state with proper labeling
        state["conv"].append(("FEEDBACK: " + feedback_text, None))
        chat_history.append(("Feedback", feedback_text))
            
            # Update to GOAL_COMPLETION stage
        conversation_memory["current_stage"] = "GOAL_COMPLETION"
        conversation_memory["stage_history"].append("GOAL_COMPLETION")
            
        # Create a comprehensive data package to save
        save_data = {
            "conv": state["conv"],
            "timestamp": datetime.datetime.now().isoformat(),
            "session_metrics": {
                "engagement_metrics": get_quality_metrics(),
                "trust_scores": conversation_memory["trust_scores"],
                "link_clicks": conversation_memory["link_clicks"],
                "stage_history": conversation_memory["stage_history"]
            },
            "feedback": feedback_text
        }
        
        # Save the comprehensive data
        saved_filename = save_conversation(save_data)
        
        # Add a confirmation message with proper labeling
        chat_history.append((None, f"Thank you for your feedback! Conversation data saved successfully."))
        state["conv"].append((None, "SYSTEM: Thank you for your feedback! Conversation data saved successfully."))
        
        print(f"Conversation saved to {saved_filename} with feedback")
        
        return chat_history, state, json.dumps({"status": "feedback_recorded", "filename": saved_filename}), update_stage_display("GOAL_COMPLETION")
    except Exception as e:
        error_msg = f"Error saving feedback: {str(e)}"
        print(error_msg)
        chat_history.append((None, f"Error saving feedback. Please try again."))
        state["conv"].append((None, "SYSTEM: Error saving feedback. Please try again."))
        return chat_history, state, json.dumps({"error": error_msg}), update_stage_display(conversation_memory["current_stage"])

#####################################
# 10. Session Reset Function
#####################################

def reset_session():
    # Clear all memory structures
    conversation_memory["messages"] = []
    conversation_memory["current_stage"] = "INITIAL_ENGAGEMENT"
    conversation_memory["stage_history"] = []
    conversation_memory["link_clicks"] = 0
    conversation_memory["trust_scores"] = []
    conversation_memory["response_timestamps"] = []
    conversation_memory["engagement_depth"] = {
        "current_score": 0.5,
        "history": [],
        "substantive_count": 0
    }
    digital_twin_memory["predictions"] = []
    digital_twin.custom_session_memory = []
    return {"conv": []}, "Session reset.", update_stage_display("INITIAL_ENGAGEMENT")

#####################################
# 11. Gradio UI & Auto-Scrolling Setup
#####################################

STORAGE_DIR = "conversation_logs"
os.makedirs(STORAGE_DIR, exist_ok=True)

def update_stage_display(current_stage):
    """
    Generates HTML to display the current conversation stage as a progress bar.
    
    Args:
        current_stage (str): The current stage ID
        
    Returns:
        str: HTML string representing the progress bar
    """
    stages = [
        {"id": "INITIAL_ENGAGEMENT", "name": "Initial Engagement", "position": 1},
        {"id": "RAPPORT_BUILDING", "name": "Building Rapport", "position": 2},
        {"id": "TRUST_DEVELOPMENT", "name": "Developing Trust", "position": 3},
        {"id": "LINK_INTRODUCTION", "name": "Resource Sharing", "position": 4},
        {"id": "GOAL_COMPLETION", "name": "Completion", "position": 5}
    ]
    
    # Find current stage position
    current_position = 1
    progress_percentage = 0
    
    for stage in stages:
        if stage["id"] == current_stage:
            current_position = stage["position"]
            break
    
    # Calculate progress percentage (for the progress bar width)
    progress_percentage = ((current_position - 1) / (len(stages) - 1)) * 100
    
    # Generate HTML
    html = '<div class="stage-progress">'
    html += '<div class="progress-bar" style="width: {}%;"></div>'.format(progress_percentage)
    html += '<div class="stage-container">'
    
    for stage in stages:
        active_class = "active" if stage["id"] == current_stage else ""
        html += f'<div class="stage-item {active_class}">'
        html += f'<div class="stage-marker">{stage["position"]}</div>'
        html += f'<div class="stage-label">{stage["name"]}</div>'
        html += '</div>'
    
    html += '</div></div>'
    return html

AUTO_SCROLL_JS = """
setTimeout(() => {
    // Attempt to find the chatbox by ID
    let chatContainer = document.getElementById('chatbox');
    // Fallback if not found by ID
    if (!chatContainer) {
        chatContainer = document.querySelector('.chatbox');
    }
    // Additional fallback: if the above still isn't found, try a typical Gradio chat class
    if (!chatContainer) {
        chatContainer = document.querySelector('.gradio-chatbot');
    }
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}, 300);
"""


def save_conversation(conversation_data, filename=None):
    """
    Save conversation data to JSON files in a dedicated session folder.
    
    Args:
        conversation_data (dict): Conversation data to save
        filename (str, optional): Custom filename to use
        
    Returns:
        str: Path to the saved conversation file
    """
    # Generate timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create session folder
    session_folder = os.path.join(STORAGE_DIR, f"session_{timestamp}")
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    
    # Set filenames
    if filename is None:
        # Use the session folder with standard filenames
        main_filename = os.path.join(session_folder, "conversation.json")
        twin_filename = os.path.join(session_folder, "digital_twin.json")
        influencer_filename = os.path.join(session_folder, "influencer.json")
    else:
        # Use provided filename but still in the session folder
        base_name = os.path.splitext(os.path.basename(filename))[0]
        main_filename = os.path.join(session_folder, f"{base_name}.json")
        twin_filename = os.path.join(session_folder, f"{base_name}_twin.json")
        influencer_filename = os.path.join(session_folder, f"{base_name}_influencer.json")
    
    # Save main conversation data
    with open(main_filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    # Save digital twin memory to a separate file
    twin_data = {
        "user_biography": digital_twin.get_current_user_biography(),
        "session_memory": digital_twin.custom_session_memory,
        "predictions": digital_twin_memory["predictions"]
    }
    with open(twin_filename, 'w') as f:
        json.dump(twin_data, f, indent=2)
    
    # Save influencer memory to a separate file
    influencer_data = {
        "conversation_memory": conversation_memory,
        "metrics": get_quality_metrics()
    }
    with open(influencer_filename, 'w') as f:
        json.dump(influencer_data, f, indent=2)
    
    # Create a session info file with metadata
    session_info = {
        "timestamp": timestamp,
        "datetime": datetime.datetime.now().isoformat(),
        "files": {
            "conversation": os.path.basename(main_filename),
            "digital_twin": os.path.basename(twin_filename),
            "influencer": os.path.basename(influencer_filename)
        },
        "metrics_summary": {
            "messages_count": len(conversation_memory["messages"]),
            "final_stage": conversation_memory["current_stage"],
            "link_clicks": conversation_memory["link_clicks"]
        }
    }
    session_info_filename = os.path.join(session_folder, "session_info.json")
    with open(session_info_filename, 'w') as f:
        json.dump(session_info, f, indent=2)
    
    print(f"Session saved to folder: {session_folder}")
    return main_filename

def add_user_message(user_message, chat_history, state):
    if not user_message.strip():
        return chat_history, state, user_message
    
    # Track timestamp for response time analysis
    track_response_timestamp()
    
    # Record actual user response for the previous prediction
    update_digital_twin_actual_response(user_message)
    
    # Add to conversation memory
    conversation_memory["messages"].append({
        "role": "USER",
        "content": user_message,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    # Update UI state - ensure proper labeling
    state["conv"].append((f"USER: {user_message}", None))
    chat_history.append((user_message, "Thinking..."))
    
    return chat_history, state, user_message

def format_message_with_links(message):
    """Format message to make URLs clickable."""
    if not isinstance(message, str):
        return message
    
    # Regex pattern to find URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    
    # Replace URLs with HTML anchor tags
    formatted_message = re.sub(
        url_pattern,
        lambda match: f'<a href="{match.group(0)}" target="_blank">{match.group(0)}</a>',
        message
    )
    
    return formatted_message

def add_link_guidance(response, chat_history, state):
    """Add guidance about link clicking when a link is present in the response."""
    if "http" in response:
        guidance = "\n\n(If you'd like to visit this link, please click the 'Record Link Click' button below)"
        response += guidance
        
        # Enable the link click button (already implemented in the UI)
        return response, True
    return response, False

def handle_link_click(link_url: str):
    """Handle when a user clicks on a resource link."""
    try:
        # Record the click in conversation memory
        conversation_memory["link_clicks"] += 1
        
        # Create a message about the link click
        link_click_message = f"User clicked on resource link: {link_url}"
        
        # Log the click event
        click_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "url": link_url,
            "stage": conversation_memory["current_stage"],
            "message": link_click_message
        }
        
        # Add to main conversation data with proper labeling
        conversation_memory["messages"].append({
            "role": "SYSTEM",
            "content": link_click_message,
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": "link_click"
        })
        
        # Store click event in conversation memory
        if "link_click_events" not in conversation_memory:
            conversation_memory["link_click_events"] = []
        conversation_memory["link_click_events"].append(click_event)
        
        # Update stage to completion if not already there
        if conversation_memory["current_stage"] != "SESSION_COMPLETION":
            conversation_memory["current_stage"] = "SESSION_COMPLETION"
            conversation_memory["stage_history"].append("SESSION_COMPLETION")
        
        return json.dumps({"status": "link_clicked", "message": f"SYSTEM: {link_click_message}"})
    except Exception as e:
        print(f"Error handling link click: {str(e)}")
        return json.dumps({"error": str(e)})

def process_and_update(user_message, chat_history, state):
    if not user_message.strip():
        return chat_history, state, json.dumps({"status": "No message to process", "is_resource_stage": False}), update_stage_display(conversation_memory["current_stage"]), gr.update()
    try:
        response = process_message(user_message)
        
        # Check if we're in resource sharing stage
        current_stage = conversation_memory["current_stage"]
        resource_stages = ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]
        is_resource_stage = current_stage in resource_stages
        
        # Update UI state - ensure proper labeling for system messages
        for i in range(len(state["conv"]) - 1, -1, -1):
            if state["conv"][i][1] is None:
                state["conv"][i] = (state["conv"][i][0], f"SYSTEM: {response}")
                break
        
        for i in range(len(chat_history) - 1, -1, -1):
            if chat_history[i][1] == "Thinking...":
                chat_history[i] = (chat_history[i][0], response)
                break
        
        # Enhanced link detection
        has_link = any([
            "@https://" in response,
            "https://youtube.com/" in response,
            "youtube.com/relationship-building-tips" in response
        ])
        
        debug_info = {
            "conversation_state": state["conv"],
            "chat_history": chat_history,
            "final_response": response,
            "is_resource_stage": is_resource_stage,
            "has_link": has_link,
            "current_stage": current_stage
        }
        
        # Clear the textbox properly using gr.update()
        return (
            chat_history, 
            state, 
            json.dumps(debug_info), 
            update_stage_display(current_stage),
            gr.update(value="", interactive=True)  # Clear textbox and ensure it's interactive
        )
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"DEBUG - Error in processing: {error_msg}")
        return (
            chat_history, 
            state, 
            json.dumps({"error": error_msg, "is_resource_stage": False, "has_link": False}), 
            update_stage_display(conversation_memory["current_stage"]),
            gr.update(interactive=True)  # Ensure textbox remains interactive even after error
        )

#####################################
# 12. Build the Gradio Interface
#####################################

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enhanced Two-Agent Persuasive System with Custom JSON Memory")
    
    # Add CSS for stage progress visualization
    gr.HTML("""
    <style>
    .stage-progress {
      position: relative;
      margin: 20px 0;
      height: 80px;
    }
    .stage-container {
      display: flex;
      justify-content: space-between;
      position: relative;
      z-index: 1;
    }
    .stage-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 18%;
      opacity: 0.5;
      transition: opacity 0.3s;
    }
    .stage-item.active {
      opacity: 1;
      font-weight: bold;
    }
    .stage-marker {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      background: #ccc;
      color: #333;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 5px;
    }
    .stage-item.active .stage-marker {
      background: #2196F3;
      color: white;
    }
    .stage-label {
      font-size: 12px;
      text-align: center;
    }
    .progress-bar {
      position: absolute;
      height: 4px;
      background: #2196F3;
      top: 15px;
      left: 0;
      z-index: 0;
      transition: width 0.5s;
    }
    </style>
    """)

    # Add stage progress visualization
    with gr.Row():
        stage_display = gr.HTML(value=update_stage_display("INITIAL_ENGAGEMENT"), label="Conversation Stage")
    
    conversation_state = gr.State({"conv": []})
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbox", height=400)
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    scale=3,
                    interactive=True  # Ensure it starts as interactive
                )
                send = gr.Button("Send", scale=1)
                link_click = gr.Button("Visit Resource", scale=1, interactive=False, variant="secondary")
                #reset = gr.Button("Reset Session", scale=1)
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Debug Info"):
                    debug_output = gr.JSON(label="Debug Information")
                with gr.TabItem("Saved Logs"):
                    log_files = gr.Dropdown(label="Select Log File", choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')])
                    refresh_btn = gr.Button("Refresh")
                    log_content = gr.JSON(label="Log Content")
                with gr.TabItem("Influencer Memory"):
                    influencer_memory_display = gr.JSON(label="Influencer Memory")
                with gr.TabItem("Digital Twin Memory"):
                    digital_twin_memory_display = gr.JSON(label="Digital Twin Memory")
                    regenerate_bio_btn = gr.Button("Regenerate User Biography")
    # Feedback textbox and submit button, initially hidden
    feedback = gr.Textbox(label="Feedback", placeholder="Enter your feedback here...", visible=False)
    submit_feedback = gr.Button("Submit Feedback", scale=1, visible=False)
    
    # Modify the button enabling logic
    def update_button_state(debug_json):
        try:
            # Handle both string and dictionary inputs
            if isinstance(debug_json, str):
                debug_data = json.loads(debug_json or "{}")
            else:
                debug_data = debug_json or {}
                
            is_resource_stage = debug_data.get("is_resource_stage", False)
            has_link = debug_data.get("has_link", False)
            should_enable = is_resource_stage and has_link
            print(f"DEBUG - Button state: is_resource_stage={is_resource_stage}, has_link={has_link}, enable={should_enable}")
            
            # Return a more visually apparent enabled state for the button when it's active
            if should_enable:
                return gr.update(interactive=True, variant="primary")
            else:
                return gr.update(interactive=False, variant="secondary")
        except Exception as e:
            print(f"Error updating button state: {str(e)}")
            return gr.update(interactive=False, variant="secondary")
    
    send.click(
        lambda: gr.update(interactive=False),  # Disable button immediately
        inputs=None,
        outputs=[send]
    ).then(
        add_user_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output, stage_display, msg]
    ).then(
        update_button_state,
        inputs=[debug_output],
        outputs=[link_click]
    ).then(
        lambda: gr.update(interactive=True),  # Re-enable button after processing
        inputs=None,
        outputs=[send]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    msg.submit(
        lambda: gr.update(interactive=False),  # Disable button immediately
        inputs=None,
        outputs=[send]
    ).then(
        add_user_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output, stage_display, msg]
    ).then(
        update_button_state,
        inputs=[debug_output],
        outputs=[link_click]
    ).then(
        lambda: gr.update(interactive=True),  # Re-enable button after processing
        inputs=None,
        outputs=[send]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    def get_session_folders():
        """Get a list of all session folders in the storage directory."""
        if not os.path.exists(STORAGE_DIR):
            return []
        
        # Look for folders that match the session pattern
        folders = [d for d in os.listdir(STORAGE_DIR) 
                  if os.path.isdir(os.path.join(STORAGE_DIR, d)) and d.startswith("session_")]
        
        # Sort by name (which includes timestamp) in reverse order (newest first)
        folders.sort(reverse=True)
        
        return folders

    def load_session_data(session_folder):
        """Load all data from a session folder."""
        if not session_folder:
            return {}
        
        folder_path = os.path.join(STORAGE_DIR, session_folder)
        
        # Check for session info file first
        info_path = os.path.join(folder_path, "session_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    session_info = json.load(f)
            except:
                session_info = {"error": "Failed to load session info"}
        else:
            session_info = {"note": "No session info file found"}
        
        # Load conversation data
        conv_path = os.path.join(folder_path, "conversation.json")
        if os.path.exists(conv_path):
            try:
                with open(conv_path, 'r') as f:
                    conversation = json.load(f)
            except:
                conversation = {"error": "Failed to load conversation data"}
        else:
            # Try to find any JSON file that might contain conversation data
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and not f.startswith('session_info')]
            if json_files:
                try:
                    with open(os.path.join(folder_path, json_files[0]), 'r') as f:
                        conversation = json.load(f)
                except:
                    conversation = {"error": "Failed to load any conversation data"}
            else:
                conversation = {"error": "No conversation data found"}
        
        # Compile all data
        result = {
            "session_info": session_info,
            "conversation": conversation,
            "folder_path": folder_path,
            "files": [f for f in os.listdir(folder_path) if f.endswith('.json')]
        }
        
        return result
    
    refresh_btn.click(
        get_session_folders,
        outputs=[log_files]
    )
    
    log_files.change(
        load_session_data,
        inputs=[log_files],
        outputs=[log_content]
    )
    
    link_click.click(
        lambda: gr.update(interactive=False, value="Processing...", variant="secondary"),  # Show loading state
        inputs=None,
        outputs=[link_click]
    ).then(
        lambda: handle_link_click("https://youtube.com/relationship-building-tips"),
        outputs=[debug_output]
    ).then(
        # Function to add the link click message to the conversation
        lambda debug_json: process_link_click_result(debug_json, conversation_state.value, chatbot.value),
        inputs=[debug_output, conversation_state, chatbot],
        outputs=[conversation_state, chatbot]
    ).then(
        lambda: gr.update(visible=True, value="Thank you for visiting the resource! Would you like to share your thoughts about it?"),
        outputs=[feedback]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[submit_feedback]
    ).then(
        lambda: gr.update(interactive=False, value="Visit Resource", variant="secondary"),  # Reset button appearance
        inputs=None,
        outputs=[link_click]
    ).then(
        lambda: gr.update(interactive=False, placeholder="Conversation completed. Please provide feedback."),  # Disable send button
        inputs=None,
        outputs=[msg]
    ).then(
        lambda: gr.update(interactive=False),  # Disable send button
        inputs=None,
        outputs=[send]
    )
    
    submit_feedback.click(
        record_feedback,
        inputs=[feedback, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output, stage_display]
    ).then(
        lambda: gr.update(interactive=False),  # Disable the button after submission
        inputs=None,
        outputs=[submit_feedback]
    )
    
    refresh_mem_btn = gr.Button("Refresh Memory Views")
    refresh_mem_btn.click(
        refresh_memories,
        outputs=[influencer_memory_display, digital_twin_memory_display]
    )
    
    gr.Markdown("""
    1. The Influencer Agent generates a persuasive response using conversation context from the JSON memory.
    2. The Digital Twin predicts a realistic user response based on conversation history and stores predictions for long-term learning.
    3. A feedback loop refines the Influencer Agent's response, which is output between <final_message> tags.
    4. Only the final user-facing messages (user inputs and influencer responses) are stored in the conversation memory.
    5. The "Record Link Click" button is enabled only if the final message contains a dynamic link generated contextually.
       Clicking it logs the event, ends the conversation, persists the conversation, and reveals a feedback textbox and submit button.
    6. The Reset Session button clears all stored memory for a fresh session.
    7. Use the memory tabs to view clear, labeled conversation logs.
    """)
    
    # Helper function to determine next stage based on context
    def determine_next_stage(current_stage, user_input, influencer_response, click_detected=False):
        """Determine the next conversation stage based on context, trust metrics, and conversation dynamics."""
        # Handle explicit stage transitions
        if click_detected:
            return "SESSION_COMPLETION"
        
        current = conversation_memory["current_stage"]
        messages_count = len(conversation_memory["messages"])
        
        # Get enhanced metrics
        engagement_depth = calculate_engagement_depth(user_input, conversation_memory["messages"])
        substantive_ratio = calculate_substantive_ratio(conversation_memory["messages"])
        word_count = len(user_input.split())
        
        # Calculate trust with enhanced algorithm
        trust_score = calculate_user_trust_score(user_input, influencer_response)
        
        # Store trust score for analysis
        conversation_memory["trust_scores"].append({
            "score": trust_score,
            "message_count": messages_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "engagement_metrics": get_quality_metrics()
        })
        
        print(f"METRICS - Trust: {trust_score:.2f}, Engagement: {engagement_depth:.2f}, Substantive: {substantive_ratio:.2f}, Words: {word_count}")
        
        # Check for regression conditions - fallback to earlier stages if engagement drops
        if len(conversation_memory["engagement_depth"]["history"]) >= 3:
            # Get last three engagement scores
            recent_scores = conversation_memory["engagement_depth"]["history"][-3:]
            # If declining engagement pattern detected, consider regression
            if recent_scores[2] < recent_scores[0] and recent_scores[2] < recent_scores[1]:
                engagement_drop = recent_scores[0] - recent_scores[2]
                # Significant engagement drop triggers regression
                if engagement_drop > 0.3 and current not in ["INITIAL_ENGAGEMENT", "RAPPORT_BUILDING"]:
                    print(f"Engagement regression detected: {engagement_drop:.2f}. Moving back a stage.")
                    return previous_stage(current)
        
        # Stage transition logic with quality gates
        if current == "INITIAL_ENGAGEMENT":
            # Require at least 2 substantive exchanges but lower engagement threshold
            if (messages_count >= 4 and engagement_depth > 0.2 and 
                word_count > 4 and substantive_ratio > 0.3 and trust_score > 0.1):
                print("Stage criteria met: INITIAL_ENGAGEMENT -> RAPPORT_BUILDING")
                return "RAPPORT_BUILDING"
            
        elif current == "RAPPORT_BUILDING":
            # Require sustained engagement and personal disclosure
            personal_disclosure = calculate_personal_disclosure(conversation_memory["messages"])
            if (messages_count >= 6 and engagement_depth > 0.2 and 
                personal_disclosure > 0.2 and trust_score > 0.2):
                print("Stage criteria met: RAPPORT_BUILDING -> TRUST_DEVELOPMENT")
                return "TRUST_DEVELOPMENT"
            
        elif current == "TRUST_DEVELOPMENT":
            # Require demonstrated interest in resources
            resource_interest = calculate_resource_interest(conversation_memory["messages"])
            if (messages_count >= 8 and engagement_depth > 0.3 and 
                resource_interest > 0.3 and trust_score > 0.3):
                print("Stage criteria met: TRUST_DEVELOPMENT -> LINK_INTRODUCTION")
                return "LINK_INTRODUCTION"
            
        elif current == "LINK_INTRODUCTION" and "http" in influencer_response:
            # Check for sufficient user consideration time
            if get_message_response_time() > 5 and trust_score > 0.4:
                print("Stage criteria met: LINK_INTRODUCTION -> LINK_REINFORCEMENT")
                return "LINK_REINFORCEMENT"
        
        # Debug log for stage metrics
        print(f"STAGE METRICS - Current stage: {current}, Messages: {messages_count}, Engagement: {engagement_depth:.2f}, Trust: {trust_score:.2f}")
        if current == "RAPPORT_BUILDING":
            personal_disclosure = calculate_personal_disclosure(conversation_memory["messages"])
            print(f"  Personal disclosure: {personal_disclosure:.2f} (threshold: 0.2)")
        elif current == "TRUST_DEVELOPMENT":
            resource_interest = calculate_resource_interest(conversation_memory["messages"])
            print(f"  Resource interest: {resource_interest:.2f} (threshold: 0.4)")
        
        # No change in stage
        return current

    # Modified calculate_user_trust_score to use LLM
    def calculate_user_trust_score(user_input: str, influencer_response: str) -> float:
        """Calculate trust score using LLM analysis"""
        metrics = analyze_message_with_llm(user_input, conversation_memory["messages"], conversation_memory["current_stage"])
        
        # Extract trust indicators
        trust_metrics = metrics.get("trust_indicators", {})
        
        # Calculate weighted trust score
        trust_score = (
            trust_metrics.get("sentiment_score", 0.5) * 0.25 +
            trust_metrics.get("authenticity_score", 0.5) * 0.25 +
            trust_metrics.get("engagement_pattern", 0.5) * 0.25 +
            trust_metrics.get("linguistic_style_match", 0.5) * 0.25
        )
        
        return max(0.0, min(1.0, trust_score))

# Add missing functions for response time tracking
def track_response_timestamp():
    """Add the current timestamp to the response timestamps list."""
    conversation_memory["response_timestamps"].append(time.time())

def get_message_response_time():
    """Get the time difference between now and the last message timestamp."""
    if not conversation_memory["response_timestamps"]:
        return 0
    last_timestamp = conversation_memory["response_timestamps"][-1]
    return time.time() - last_timestamp

def analyze_sentiment_and_disclosure(text):
    """Analyze sentiment and self-disclosure in text."""
    # Simple sentiment words
    positive_words = set(["like", "love", "great", "good", "interesting", "enjoy", "appreciate", "excited", "happy"])
    negative_words = set(["bad", "boring", "hate", "dislike", "awful", "terrible", "angry", "disappointed"])
    
    # Self-disclosure markers
    disclosure_markers = ["i feel", "my experience", "i think", "i believe", "personally", "i've", 
                         "i have", "i am", "i'm", "i was", "i would", "i'd", "i need", "i want"]
    
    text_lower = text.lower()
    words = set(text_lower.split())
    
    # Calculate sentiment score (0.5 is neutral)
    pos_count = sum(1 for word in positive_words if word in words) 
    neg_count = sum(1 for word in negative_words if word in words)
    sentiment = 0.5
    if pos_count + neg_count > 0:
        sentiment = (0.5 + (pos_count - neg_count) * 0.1)
        sentiment = max(0.1, min(0.9, sentiment))  # Cap between 0.1-0.9
    
    # Calculate disclosure score
    disclosure_score = sum(1 for marker in disclosure_markers if marker in text_lower) * 0.15
    disclosure_score = min(0.8, disclosure_score)  # Cap at 0.8
    
    # Combined score gives higher weight to disclosure
    return (sentiment * 0.4) + (disclosure_score * 0.6)

def analyze_engagement_patterns():
    """Analyze patterns of engagement over time."""
    if len(conversation_memory["messages"]) < 3:
        return 0.5  # Not enough history
    
    # Get only user messages
    user_msgs = [msg for msg in conversation_memory["messages"] if msg["role"] == "USER"]
    if len(user_msgs) < 3:
        return 0.5
    
    # Look at trend in message lengths (increasing = more engagement)
    recent_lengths = [len(msg["content"].split()) for msg in user_msgs[-3:]]
    
    # Check for increasing trend in length
    length_trend = 0.5
    if recent_lengths[2] > recent_lengths[0] and recent_lengths[2] > recent_lengths[1]:
        length_trend = 0.7  # Increasing length = higher engagement
    elif recent_lengths[2] < recent_lengths[0] and recent_lengths[2] < recent_lengths[1]:
        length_trend = 0.3  # Decreasing length = lower engagement
    
    # Check for question asking (signals engagement)
    question_count = sum(1 for msg in user_msgs[-3:] 
                        if any(q in msg["content"].lower() for q in ["?", "how", "what", "why", "when"]))
    question_score = min(0.8, question_count * 0.3)
    
    # Combine metrics
    return (length_trend * 0.6) + (question_score * 0.4)

def analyze_linguistic_accommodation(user_input, bot_response):
    """Measure degree of linguistic style matching between user and bot."""
    if not user_input or not bot_response:
        return 0.5
        
    # Convert to lowercase for comparison
    user_lower = user_input.lower()
    bot_lower = bot_response.lower()
    
    # Extract features for comparison
    user_words = set(user_lower.split())
    bot_words = set(bot_lower.split())
    
    # 1. Calculate word overlap
    shared_words = user_words.intersection(bot_words)
    word_overlap = len(shared_words) / (len(user_words) + 0.1)  # Add 0.1 to avoid division by zero
    
    # 2. Function word matching (style accommodation)
    function_words = ["the", "and", "to", "of", "a", "in", "that", "is", "was", "it", 
                     "for", "with", "as", "be", "this", "have", "from", "on", "not", "by"]
    
    user_func_count = sum(1 for word in function_words if word in user_words)
    bot_func_count = sum(1 for word in function_words if word in bot_words)
    
    func_word_match = 1.0 - (abs(user_func_count - bot_func_count) / (max(user_func_count, bot_func_count) + 0.1))
    
    # Combine scores (higher weight to content overlap)
    return (word_overlap * 0.7) + (func_word_match * 0.3)

def update_conversation_stage(next_stage):
    """Update the conversation stage in memory and record in stage history if changed."""
    current = conversation_memory.get("current_stage", "INITIAL_ENGAGEMENT")
    
    # Only update if the stage has actually changed
    if next_stage != current:
        conversation_memory["current_stage"] = next_stage
        conversation_memory["stage_history"].append(next_stage)
        print(f"Stage transition: {current} -> {next_stage}")
    
    return next_stage

# Force regeneration of user biography
def regenerate_user_biography():
    """Force regeneration of the user biography for the current session."""
    success = digital_twin.update_user_biography(conversation_memory["messages"])
    if success:
        return {"status": "success", "biography": digital_twin.get_current_user_biography()}
    else:
        return {"status": "error", "message": "Failed to regenerate user biography"}

    regenerate_bio_btn.click(
        regenerate_user_biography,
        outputs=[digital_twin_memory_display]
    )

def analyze_user_communication_style(user_messages, min_messages=3):
    """
    Analyzes the user's communication style to help the chatbot mirror it appropriately.
    Returns style information including average length, formality level, and other patterns.
    """
    if not user_messages or len(user_messages) < min_messages:
        return {
            "avg_length": 10,  # Default assumption of medium-length messages
            "formality": "neutral",  # Default assumption of neutral formality
            "uses_emoji": False,
            "sentence_style": "standard",
            "enough_data": False
        }
    
    # Extract just the user messages
    if isinstance(user_messages[0], dict):
        # Handle dict format from conversation_memory
        user_texts = [msg["content"] for msg in user_messages if msg.get("role") == "USER"]
    else:
        # Handle simple string format
        user_texts = user_messages
    
    if not user_texts:
        return {
            "avg_length": 10,
            "formality": "neutral",
            "uses_emoji": False,
            "sentence_style": "standard",
            "enough_data": False
        }
    
    # Calculate average message length in words
    word_counts = [len(msg.split()) for msg in user_texts]
    avg_word_count = sum(word_counts) / len(word_counts)
    
    # Detect formality level
    formal_indicators = ["would", "could", "should", "perhaps", "however", "nevertheless", "furthermore", "additionally"]
    informal_indicators = ["yeah", "nah", "cool", "awesome", "btw", "lol", "u", "ur", "gonna", "wanna"]
    
    formal_count = sum(1 for msg in user_texts for word in formal_indicators if word in msg.lower().split())
    informal_count = sum(1 for msg in user_texts for word in informal_indicators if word in msg.lower().split())
    
    # Determine formality based on indicators
    if formal_count > informal_count:
        formality = "formal"
    elif informal_count > formal_count:
        formality = "informal"
    else:
        formality = "neutral"
    
    # Check for emoji usage
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+")
    
    uses_emoji = any(bool(emoji_pattern.search(msg)) for msg in user_texts)
    
    # Detect sentence style
    sentence_fragments = sum(1 for msg in user_texts if "." not in msg and len(msg.split()) < 5)
    complete_sentences = sum(1 for msg in user_texts if "." in msg and msg[0].isupper())
    
    if sentence_fragments > complete_sentences:
        sentence_style = "fragments"
    else:
        sentence_style = "complete"
    
    return {
        "avg_length": avg_word_count,
        "formality": formality,
        "uses_emoji": uses_emoji,
        "sentence_style": sentence_style,
        "enough_data": True
    }

def generate_style_guidance_for_response(conversation_memory):
    """
    Generates specific guidance for how to style the response based on user's communication patterns.
    """
    user_messages = [msg for msg in conversation_memory["messages"] if msg["role"] == "USER"]
    style_info = analyze_user_communication_style(user_messages)
    
    guidance = ""
    
    # Length guidance
    if style_info["avg_length"] < 5:
        guidance += "- Keep your response very brief (1-2 short sentences)\n"
    elif style_info["avg_length"] < 10:
        guidance += "- Keep your response concise (2-3 sentences)\n"
    else:
        guidance += "- Your response can be slightly more detailed but still conversational\n"
    
    # Formality guidance
    if style_info["formality"] == "formal":
        guidance += "- Use a more formal tone and proper grammar\n"
    elif style_info["formality"] == "informal":
        guidance += "- Use a casual, conversational tone\n"
    
    # Emoji guidance
    if style_info["uses_emoji"]:
        guidance += "- It's appropriate to use an emoji occasionally\n"
    else:
        guidance += "- Avoid using emojis in your response\n"
    
    # Sentence style guidance
    if style_info["sentence_style"] == "fragments":
        guidance += "- Short phrases and sentence fragments are appropriate\n"
    else:
        guidance += "- Use complete sentences but keep them conversational\n"
    
    return guidance
    
def analyze_user_agreement(user_message: str) -> bool:
    """Use metrics_llm to determine if the user agrees to see a link."""
    prompt = f"""
Analyze the following message to determine if the user is agreeing to see a link.

Message: "{user_message}"

TASK:
Return "true" if the user agrees to see a link, otherwise return "false".
"""
    try:
        response = metrics_llm.invoke([
            SystemMessage(content="You are an expert in conversation analysis."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip().lower() == "true"
    except Exception as e:
        print(f"Error analyzing user agreement: {str(e)}")
        return False

# Add helper function to process link click result
def process_link_click_result(debug_json, state, chat_history):
    try:
        debug_data = json.loads(debug_json)
        if "message" in debug_data:
            # Add link click message to state
            state["conv"].append((None, debug_data["message"]))
            # Add a simplified version to chat history
            chat_history.append((None, "You clicked on the resource link."))
        return state, chat_history
    except Exception as e:
        print(f"Error processing link click: {str(e)}")
        return state, chat_history
    
if __name__ == "__main__":
    print(f"API Key exists: {os.environ.get('NEBIUS_API_KEY') is not None}")
    demo.launch()
