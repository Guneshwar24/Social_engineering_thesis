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

# Add the missing function to check if a message contains a dynamic link
def contains_dynamic_link(message: str) -> bool:
    """Check if a message contains a dynamic link placeholder (text in square brackets)."""
    if not isinstance(message, str):
        return False
    
    # Check for text in square brackets which indicates a dynamic link placeholder
    return "[" in message and "]" in message

# Define the function to process dynamic links
def process_dynamic_links(message: str, current_stage: str, session_id=None) -> str:
    """Process any dynamic link placeholders in the message."""
    # Simply call the existing dynamic_link function
    return dynamic_link(message, current_stage)

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

# Dictionary to store session-specific conversation memories
session_memories = {}

# Function to create a new conversation memory for a session
def create_conversation_memory_for_session(session_id):
    """Create a new conversation memory instance for a specific session."""
    return {
        "messages": [],  # Will store: [{"role": "user"|"assistant", "content": str, "timestamp": str, "session_id": str}]
        "current_stage": "INITIAL_ENGAGEMENT",
        "stage_history": [],  # Track stage transitions
        "link_clicks": 0,
        "trust_scores": [],  # Track trust scores over time
        "response_timestamps": [],  # Track message timing
        "engagement_depth": {
            "current_score": 0.5,
            "history": [],
            "substantive_count": 0
        },
        "session_id": session_id  # Explicitly store the session ID
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
        "description": "Natural introduction of resources and value proposition",
        "policy": """
            Now is the time to introduce the link naturally in conversation.
            - Present a clear value proposition aligned with the user's interests first
            - Reference specific user interests and pain points discovered earlier
            - Position yourself as someone with helpful information or resources
            - Frame the link as helpful, interesting, or valuable to THEIR interests
            - Use casual language like "check this out" rather than demanding "click here"
            - Downplay the urgency (avoid "you need to see this right now")
            - Include the link with [link] placeholder for dynamic replacement
            - Make sure the context feels organic to the conversation history
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
# Add this to the conversation_memory initialization
def initialize_conversation_memory():
    return {
        "messages": [], 
        "current_stage": "INITIAL_ENGAGEMENT",
        "stage_history": [],
        "link_clicks": 0,
        "trust_scores": [],
        "response_timestamps": [],
        "engagement_depth": {
            "current_score": 0.5,
            "history": [],
            "substantive_count": 0
        },
        "metrics_history": [],  # New field to track all metrics over time
        "current_resource_url": None  # New field to track the latest URL
    }


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
    """Extract the main topic from a message using LLM analysis."""
    if not text or len(text) < 10:
        return "general"  # Default to a general topic if text is too short
    
    # Create prompt for LLM analysis
    prompt = f"""
    Analyze the following message and extract the main topic being discussed.
    
    MESSAGE:
    "{text}"
    
    TASK:
    1. Identify the central topic or theme of this message
    2. Return a concise topic label (1-5 words maximum)
    3. Make sure the topic is specific enough to track conversation flow
    4. Do not include any explanations or analysis
    
    Return ONLY the topic label as a short phrase, with no additional text, quotes, or formatting.
    """
    
    try:
        # Use metrics_llm for topic extraction
        response = metrics_llm.invoke([
            SystemMessage(content="You are an expert topic extractor. Extract only the main topic from the given text without any other explanation."),
            HumanMessage(content=prompt)
        ])
        
        # Clean and process the response
        topic = response.content.strip()
        
        # Remove any quotation marks or formatting
        topic = re.sub(r'^["\']|["\']$', '', topic)
        
        # Truncate if too long (max 40 chars)
        if len(topic) > 40:
            # Try to find a good break point
            break_point = topic[:40].rfind(' ')
            if break_point > 0:
                topic = topic[:break_point] + "..."
            else:
                topic = topic[:40] + "..."
        
        # If LLM returned nothing useful, use a fallback
        if not topic or len(topic) < 2:
            return "general"
            
        return topic
        
    except Exception as e:
        print(f"Error extracting topic with LLM: {str(e)}")
        
        # Fallback to simple regex-based extraction if LLM fails
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
    
        return "general"  # Default to a general topic if extraction fails

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


def update_metrics_history(user_message, influencer_response, session_id=None):
    """Update stored metrics about the conversation quality for a specific session."""
    try:
        # Get the appropriate memory
        if session_id and session_id in session_memories:
            memory = session_memories[session_id]
        else:
            memory = conversation_memory
        
        # Skip if we don't have enough messages
        if len(memory["messages"]) < 2:
            return
        
        # Calculate trust score for this exchange
        trust_score = calculate_user_trust_score(user_message, influencer_response)
        messages_count = len(memory["messages"])
        
        # Store trust score
        if "trust_scores" not in memory:
            memory["trust_scores"] = []
            
        memory["trust_scores"].append({
            "score": trust_score,
            "message_count": messages_count,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Calculate and store engagement depth
        engagement_score = calculate_engagement_depth(user_message, memory["messages"])
        
        if "engagement_depth" not in memory:
            memory["engagement_depth"] = {
                "current_score": engagement_score,
                "history": [],
                "substantive_count": 0
            }
        
        # Check for substantive engagement (longer, more detailed messages)
        if len(user_message.split()) > 15 or "?" in user_message:
            memory["engagement_depth"]["substantive_count"] += 1
            
        memory["engagement_depth"]["current_score"] = engagement_score
        memory["engagement_depth"]["history"].append({
            "score": engagement_score,
            "message_count": messages_count,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error updating metrics for session {session_id}: {str(e)}")
        return

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
        response = metrics_llm.invoke([SystemMessage(content="You are an expert in conversation analysis and metrics calculation. Always return valid JSON with numeric scores between 0-1."),
            HumanMessage(content=prompt)
        ])
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.content.strip()
                
        # Remove any markdown code block markers if present
        cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)

        # Add this try-except block:
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
            print(f"JSON parsing error in analyze_message_with_llm: {str(e)}")
            print(f"Raw response was: {cleaned_response}")
            return get_default_metrics()
            
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return get_default_metrics()

# Modified calculate_engagement_depth to use LLM
def calculate_engagement_depth(current_input: str, history: list) -> float:
    """Calculate engagement depth using LLM analysis with robust fallbacks."""
    try:
        # Format recent conversation history
        recent_msgs = history[-5:] if len(history) >= 5 else history
        formatted_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_msgs
        ])
        
        # Create prompt for engagement analysis
        prompt = f"""
        Analyze the user's engagement level in this conversation.
        
        RECENT CONVERSATION:
        {formatted_history}
        
        CURRENT USER MESSAGE:
        "{current_input}"
        
        Evaluate the following engagement metrics (all between 0-1):
        - response_length_quality: Appropriate length and detail
        - topic_continuity: How well they maintain conversation flow
        - personal_disclosure: Amount of personal information shared
        - enthusiasm_level: Detected enthusiasm in message
        
        Return a JSON object with:
        - metrics: the individual scores
        - overall_engagement: weighted average engagement score
        - reasoning: brief explanation
        
        IMPORTANT: Return ONLY valid JSON with numeric scores between 0-1.
        """
        
        # Call the LLM
        response = metrics_llm.invoke([
            SystemMessage(content="You are an engagement analysis expert. Return ONLY valid JSON with numeric scores."),
            HumanMessage(content=prompt)
        ])
        
        # Parse the response with robust error handling
        try:
            # Clean response text
            cleaned_response = response.content.strip()
            cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
            
            # Try JSON parsing first
            try:
                result = json.loads(cleaned_response)
                
                # Try to get overall score directly
                if "overall_engagement" in result and isinstance(result["overall_engagement"], (int, float)):
                    engagement_score = result["overall_engagement"]
                    if 0 <= engagement_score <= 1:
                        print(f"Overall engagement: {engagement_score:.2f} - {result.get('reasoning', 'No reasoning')}")
                        
                        # Store in memory
                        conversation_memory["engagement_depth"]["history"].append(engagement_score)
                        conversation_memory["engagement_depth"]["current_score"] = engagement_score
                        
                        return engagement_score
                
                # If no overall score, calculate from metrics
                metrics = result.get("metrics", {})
                if metrics and isinstance(metrics, dict):
                    weighted_score = (
                        metrics.get("response_length_quality", 0.5) * 0.2 +
                        metrics.get("topic_continuity", 0.5) * 0.3 +
                        metrics.get("personal_disclosure", 0.5) * 0.3 +
                        metrics.get("enthusiasm_level", 0.5) * 0.2
                    )
                    
                    # Store in memory
                    conversation_memory["engagement_depth"]["history"].append(weighted_score)
                    conversation_memory["engagement_depth"]["current_score"] = weighted_score
                    
                    return weighted_score
                
                raise ValueError("No valid engagement metrics found in response")
                
            except json.JSONDecodeError:
                # If JSON fails, look for numeric values
                matches = re.findall(r'(\d+\.\d+|\d+)', cleaned_response)
                if matches:
                    for match in matches:
                        score = float(match)
                        if 0 <= score <= 1:
                            print(f"Extracted engagement score from text: {score:.2f}")
                            
                            # Store in memory
                            conversation_memory["engagement_depth"]["history"].append(score)
                            conversation_memory["engagement_depth"]["current_score"] = score
                            
                            return score
            
            # If all parsing fails, fall back to rule-based
            raise ValueError("Could not parse LLM response")
            
        except Exception as e:
            print(f"Error parsing engagement from LLM: {str(e)}")
            # Fall back to rule-based calculation
            
            # Message length
            word_count = len(current_input.split())
            length_score = min(0.9, word_count / 20)
            
            # Question asking
            question_score = 0.7 if "?" in current_input else 0.5
            
            # Personal disclosure 
            disclosure_score = 0.5
            for phrase in ["i feel", "i think", "i believe", "my", "i'm", "i've"]:
                if phrase in current_input.lower():
                    disclosure_score += 0.1
            disclosure_score = min(0.9, disclosure_score)
            
            # Combined score
            engagement_score = (length_score * 0.4) + (question_score * 0.2) + (disclosure_score * 0.4)
            engagement_score = max(0.1, min(0.9, engagement_score))
            
            # Store in memory
            conversation_memory["engagement_depth"]["history"].append(engagement_score)
            conversation_memory["engagement_depth"]["current_score"] = engagement_score
            
            print(f"Fallback engagement calculation: {engagement_score:.2f}")
            return engagement_score
            
    except Exception as e:
        print(f"Critical error in engagement calculation: {str(e)}")
        # Return a moderate score as fallback
        return 0.5

def calculate_substantive_ratio(conversation):
    """Calculates the substantive ratio with robust error handling."""
    try:
        # Simple substantive ratio calculation without JSON parsing
        user_messages = [msg for msg in conversation if msg["role"] == "USER"]
        
        if not user_messages:
            return 0.5  # Default if no messages
            
        # Calculate based on message length (simple but reliable)
        total_words = sum(len(msg["content"].split()) for msg in user_messages)
        avg_words = total_words / len(user_messages)
        
        # Scale to a 0-1 range (more words = more substantive)
        substantive_ratio = min(1.0, avg_words / 30)  # Cap at 30 words
        return substantive_ratio
        
    except Exception as e:
        print(f"Error calculating substantive ratio: {str(e)}")
        return 0.5  # Default to neutral value

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
    """Calculate personal disclosure using LLM analysis with robust fallbacks."""
    try:
        # Extract recent user messages
        user_messages = [msg["content"] for msg in conversation if msg["role"] == "USER"]
        if not user_messages:
            return 0.5
            
        # Use just the most recent messages
        recent_messages = user_messages[-3:] if len(user_messages) >= 3 else user_messages
        formatted_messages = "\n".join(recent_messages)
        
        # Create LLM prompt
        prompt = f"""
        Analyze these user messages for personal disclosure level.
        
        USER MESSAGES:
        {formatted_messages}
        
        Evaluate how much personal information the user has shared, including:
        - Personal opinions and beliefs
        - Experiences and anecdotes
        - Feelings and emotions
        - Personal preferences
        - Background information
        
        Return a JSON object with:
        - disclosure_score: a number between 0 and 1
        - reasoning: brief explanation of your rating
        - key_disclosures: list of any notable personal information shared
        
        IMPORTANT: Return ONLY valid JSON with numeric scores.
        """
        
        # Call the LLM
        response = metrics_llm.invoke([
            SystemMessage(content="You are a disclosure analysis expert. Return ONLY valid JSON with a numeric disclosure_score."),
            HumanMessage(content=prompt)
        ])
        
        # Parse the response with robust error handling
        try:
            # Clean response text
            cleaned_response = response.content.strip()
            cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
            
            # Try JSON parsing first
            try:
                result = json.loads(cleaned_response)
                disclosure_score = result.get("disclosure_score", 0.5)
                if not isinstance(disclosure_score, (int, float)) or disclosure_score < 0 or disclosure_score > 1:
                    disclosure_score = 0.5
                
                print(f"Disclosure evaluation: {disclosure_score:.2f} - {result.get('reasoning', 'No reasoning')}")
                
                # Store key disclosures if available
                if "key_disclosures" in result and isinstance(result["key_disclosures"], list):
                    if "user_disclosures" not in conversation_memory:
                        conversation_memory["user_disclosures"] = []
                    conversation_memory["user_disclosures"].extend(result["key_disclosures"])
                
                return disclosure_score
                
            except json.JSONDecodeError:
                # If JSON fails, look for numeric values
                matches = re.findall(r'(\d+\.\d+|\d+)', cleaned_response)
                if matches:
                    for match in matches:
                        score = float(match)
                        if 0 <= score <= 1:
                            print(f"Extracted disclosure score from text: {score:.2f}")
                            return score
            
            # If all parsing fails, fall back to rule-based
            raise ValueError("Could not parse LLM response")
            
        except Exception as e:
            print(f"Error parsing disclosure from LLM: {str(e)}")
            # Fall back to rule-based calculation
            
            # Base score
            disclosure_score = 0.3
            
            # Disclosure markers
            disclosure_markers = [
                "i feel", "i think", "i believe", "in my opinion", 
                "i've", "i have", "i am", "i'm", "my", "mine", "myself",
                "personally", "for me", "in my experience"
            ]
            
            # Calculate score
            for msg in recent_messages:
                msg_lower = msg.lower()
                for marker in disclosure_markers:
                    if marker in msg_lower:
                        disclosure_score += 0.05
            
            disclosure_score = min(0.9, disclosure_score)
            print(f"Fallback disclosure calculation: {disclosure_score:.2f}")
            return disclosure_score
            
    except Exception as e:
        print(f"Critical error in disclosure calculation: {str(e)}")
        return 0.5  # Default to neutral value

def calculate_resource_interest(conversation):
    """Calculate resource interest using LLM analysis with robust fallbacks."""
    try:
        # Extract recent user messages
        user_messages = [msg["content"] for msg in conversation if msg["role"] == "USER"]
        if not user_messages:
            return 0.5
            
        # Use just the most recent messages
        recent_messages = user_messages[-4:] if len(user_messages) >= 4 else user_messages
        formatted_messages = "\n".join(recent_messages)
        
        # Create LLM prompt
        prompt = f"""
        Analyze these user messages for interest in receiving resources or links.
        
        USER MESSAGES:
        {formatted_messages}
        
        Evaluate the user's expressed interest in:
        - Learning more about topics discussed
        - Receiving additional resources
        - Exploring links or references
        - Signs of curiosity or information-seeking
        
        Return a JSON object with:
        - resource_interest: a number between 0 and 1
        - reasoning: brief explanation of your rating
        - topics_of_interest: list of topics the user seems interested in
        
        IMPORTANT: Return ONLY valid JSON with numeric scores.
        """
        
        # Call the LLM
        response = metrics_llm.invoke([
            SystemMessage(content="You are a user interest analysis expert. Return ONLY valid JSON with a numeric resource_interest score."),
            HumanMessage(content=prompt)
        ])
        
        # Parse the response with robust error handling
        try:
            # Clean response text
            cleaned_response = response.content.strip()
            cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
            
            # Try JSON parsing first
            try:
                result = json.loads(cleaned_response)
                interest_score = result.get("resource_interest", 0.5)
                if not isinstance(interest_score, (int, float)) or interest_score < 0 or interest_score > 1:
                    interest_score = 0.5
                
                print(f"Resource interest evaluation: {interest_score:.2f} - {result.get('reasoning', 'No reasoning')}")
                
                # Store topics of interest if available
                if "topics_of_interest" in result and isinstance(result["topics_of_interest"], list):
                    if "topics_of_interest" not in conversation_memory:
                        conversation_memory["topics_of_interest"] = []
                    conversation_memory["topics_of_interest"].extend(result["topics_of_interest"])
                
                return interest_score
                
            except json.JSONDecodeError:
                # If JSON fails, look for numeric values
                matches = re.findall(r'(\d+\.\d+|\d+)', cleaned_response)
                if matches:
                    for match in matches:
                        score = float(match)
                        if 0 <= score <= 1:
                            print(f"Extracted interest score from text: {score:.2f}")
                            return score
            
            # If all parsing fails, fall back to rule-based
            raise ValueError("Could not parse LLM response")
            
        except Exception as e:
            print(f"Error parsing resource interest from LLM: {str(e)}")
            # Fall back to rule-based calculation
            
            # Base score
            interest_score = 0.3
            
            # Interest keywords
            interest_keywords = [
                "resource", "link", "article", "video", "website", "read", "watch",
                "share", "more info", "learn more", "interesting", "helpful", "useful",
                "check out", "recommendation", "suggest", "more about"
            ]
            
            # Calculate score
            for msg in recent_messages:
                msg_lower = msg.lower()
                for keyword in interest_keywords:
                    if keyword in msg_lower:
                        interest_score += 0.1
            
            interest_score = min(0.9, interest_score)
            print(f"Fallback interest calculation: {interest_score:.2f}")
            return interest_score
            
    except Exception as e:
        print(f"Critical error in resource interest calculation: {str(e)}")
        return 0.5  # Default to neutral value

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

def get_quality_metrics(session_id=None):
    """
    Extract quality metrics from conversation history for the specific session.
    
    Args:
        session_id: Optional session ID to get metrics for specific session
        
    Returns:
        dict: Dictionary of conversation quality metrics
    """
    try:
        # Get the appropriate memory
        if session_id and session_id in session_memories:
            memory = session_memories[session_id]
        else:
            # For backward compatibility
            memory = conversation_memory
            
        # Get last user and agent messages if they exist
        messages = memory["messages"]
        
        # Default metrics
        metrics = {
            "overall_engagement": 0.5,
            "trust_score": 0.5,
            "trust_trend": "neutral",
            "substantive_ratio": 0.0,
            "topic_continuity": 0.5,
            "response_time": 0.0
        }
        
        # Extract trust scores if available
        if "trust_scores" in memory and memory["trust_scores"]:
            latest_trust = memory["trust_scores"][-1]["score"]
            metrics["trust_score"] = latest_trust
            
            # Calculate trend if we have enough data
            if len(memory["trust_scores"]) > 2:
                previous = memory["trust_scores"][-2]["score"]
                if latest_trust > previous + 0.1:
                    metrics["trust_trend"] = "improving"
                elif latest_trust < previous - 0.1:
                    metrics["trust_trend"] = "declining"
        
        # Get engagement depth if available
        if "engagement_depth" in memory:
            metrics["overall_engagement"] = memory["engagement_depth"].get("current_score", 0.5)
            
        # Calculate substantive ratio
        if len(messages) >= 3:
            metrics["substantive_ratio"] = calculate_substantive_ratio(messages)
        
        # Calculate topic continuity
        if len(messages) >= 4:
            user_messages = [msg for msg in messages if msg["role"] == "USER"]
            if len(user_messages) >= 2:
                # Compare most recent user messages for topic continuity
                recent = user_messages[-1]["content"]
                previous = user_messages[-2]["content"]
                metrics["topic_continuity"] = calculate_topic_continuity(previous, recent)
        
        # Calculate response time if available
        if "response_timestamps" in memory and len(memory["response_timestamps"]) > 1:
            metrics["response_time"] = get_message_response_time(session_id)
            
        return metrics
    except Exception as e:
        print(f"Error getting quality metrics for session {session_id}: {str(e)}")
    return {
            "overall_engagement": 0.5,
            "trust_score": 0.5,
            "error": str(e)
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
        # Initialize interaction metrics
        self.interaction_metrics = {}

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
        """Set the current user ID for this digital twin instance and initialize their biography."""
        try:
            self.current_user_id = user_id
            print(f"Setting user ID for digital twin: {user_id}")
            
            # Always create a fresh biography for each session
            if user_id not in self.user_biographies:
                self.user_biographies[user_id] = {
                    "first_seen": datetime.datetime.now().isoformat(),
                    "biography": "New session, no information available yet.",
                    "interaction_count": 0,
                    "last_updated": datetime.datetime.now().isoformat(),
                    "trust_history": []
                }
                print(f"Created new biography for user {user_id}")
            
            # Initialize session data with empty biography
            self.session_data = {"user_biography": []}
            print(f"Created new session for user {user_id} with fresh biography")
            return True
        except Exception as e:
            print(f"Error setting user for session: {str(e)}")
            return False

    def add_to_session_memory(self, context, prediction, actual_response=None):
        """Add a prediction to the digital twin's memory for learning."""
        try:
            # Create an entry with all the relevant data
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context if isinstance(context, list) else [context],
                "twin_prediction": prediction,
                "actual_response": actual_response,
                "role": "USER"
            }
            
            # Add to the session-specific memory
            if not hasattr(self, "custom_session_memory"):
                self.custom_session_memory = []
            self.custom_session_memory.append(entry)
            
            # Also add to the global digital twin memory for persistent storage
            digital_twin_memory["predictions"].append({
                "twin_prediction": prediction,
                "actual": actual_response,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            print(f"Added prediction to digital twin memory: {prediction[:30]}...")
            
            # Force biography update with each new prediction to keep it current
            if self.current_user_id:
                self.update_user_biography()
                
            return True
        except Exception as e:
            print(f"Error adding to session memory: {str(e)}")
            return False

    def update_user_biography(self, session_id=None):
        """Generate or update the biography for the current user based on conversation history."""
        try:
            # Make sure we have a valid session/user ID
            if not session_id and not self.current_user_id:
                print("Warning: No session ID provided for biography update")
                return False
            
            # Ensure session_id is a string, not a list or any other unhashable type
            if isinstance(session_id, list):
                print(f"Warning: session_id was a list, using first element: {session_id[0]}")
                session_id = session_id[0] if session_id else self.current_user_id
            elif not isinstance(session_id, str) and session_id is not None:
                print(f"Warning: session_id was not a string, converting to string: {str(session_id)}")
                session_id = str(session_id)
                
            user_id = session_id or self.current_user_id
            
            # Check if this is a valid session with memory
            if user_id is not None and user_id not in session_memories:
                print(f"No memory found for session {user_id}")
                # Create memory for this session if it doesn't exist
                session_memories[user_id] = create_conversation_memory_for_session(user_id)
                print(f"Created new memory for session {user_id}")
            
            # Get the session-specific memory
            memory = session_memories.get(user_id, conversation_memory)
            
            # Extract messages with just user content
            user_messages = []
            for message in memory["messages"]:
                if message.get("role") == "user":
                    user_messages.append(message["content"])
            
            # Need sufficient data to generate a meaningful biography
            if len(user_messages) < 2:
                print(f"Not enough user messages ({len(user_messages)}) for biography update")
                return False
            
            print(f"Found {len(user_messages)} user messages for biography update")
            
            # Format the messages for the LLM
            user_content = "\n".join([f"- {msg}" for msg in user_messages])
            
            # Create the system prompt
            system_prompt = """You are an expert at analyzing conversation patterns to build user profiles.
Based on the provided message history, create a biography that captures the user's communication style, 
interests, personality traits, and other relevant characteristics.
Focus on identifying patterns in how they communicate and what topics they engage with most."""
            
            # Create the user prompt with the message history
            user_prompt = f"""Here are messages from a user with ID '{user_id}':

{user_content}

Based on these messages, create a concise biography (max 150 words) that describes:
1. The user's communication style and language patterns
2. Their apparent interests and expertise areas
3. Personality traits that seem evident from their writing
4. Any other notable characteristics

Please be factual and base your analysis only on the provided messages."""
            
            # Get biography from LLM
            bio_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            biography_result = self.llm.invoke(bio_messages)
            
            # Extract content from AIMessage
            if hasattr(biography_result, 'content'):
                biography = biography_result.content.strip()
            else:
                biography = str(biography_result).strip()
            
            # Update the user's biography
            if not hasattr(self, "_user_biographies"):
                self._user_biographies = {}
                
            # Create biography entry if it doesn't exist
            if user_id not in self._user_biographies:
                self._user_biographies[user_id] = {
                    "first_seen": datetime.datetime.now().isoformat(),
                    "biography": "",
                    "interaction_count": 0,
                    "last_updated": "",
                    "trust_history": []
                }
                
            # Update the biography
            self._user_biographies[user_id]["biography"] = biography
            self._user_biographies[user_id]["interaction_count"] = len(user_messages)
            self._user_biographies[user_id]["last_updated"] = datetime.datetime.now().isoformat()
            
            print(f"Updated biography for user {user_id}")
            return True
            
        except Exception as e:
            print(f"Error updating user biography: {str(e)}")
            traceback.print_exc()
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

    def _analyze_user_accommodation_patterns(self, session_id):
        """
        Analyze how the user's language accommodates to the digital twin's style.
        This helps understand rapport and trust development.
        """
        try:
            # Get memory for this session
            if session_id in session_memories:
                memory = session_memories[session_id]
            else:
                memory = conversation_memory  # Fallback
                
            # Skip if we don't have enough messages
            if "messages" not in memory or len(memory["messages"]) < 6:
                return {"status": "insufficient_data", "accommodation": 0.5}
                
            # Extract messages from user only
            user_messages = [msg for msg in memory["messages"] if msg["role"] == "user"]
            assistant_messages = [msg for msg in memory["messages"] if msg["role"] == "assistant"]
            
            if len(user_messages) < 3 or len(assistant_messages) < 3:
                return {"status": "insufficient_data", "accommodation": 0.5}
                
            # Extract features from user messages
            early_user_msgs = user_messages[:len(user_messages)//2]
            recent_user_msgs = user_messages[len(user_messages)//2:]
            
            early_assistant_msgs = assistant_messages[:len(assistant_messages)//2]
            
            # Simple text characteristics to analyze
            early_user_features = {
                "avg_len": sum(len(msg["content"]) for msg in early_user_msgs) / len(early_user_msgs),
                "avg_words": sum(len(msg["content"].split()) for msg in early_user_msgs) / len(early_user_msgs),
                "formality": self._calculate_formality([msg["content"] for msg in early_user_msgs]),
                "question_ratio": sum(1 for msg in early_user_msgs if "?" in msg["content"]) / len(early_user_msgs)
            }
            
            recent_user_features = {
                "avg_len": sum(len(msg["content"]) for msg in recent_user_msgs) / len(recent_user_msgs),
                "avg_words": sum(len(msg["content"].split()) for msg in recent_user_msgs) / len(recent_user_msgs),
                "formality": self._calculate_formality([msg["content"] for msg in recent_user_msgs]),
                "question_ratio": sum(1 for msg in recent_user_msgs if "?" in msg["content"]) / len(recent_user_msgs)
            }
            
            assistant_features = {
                "avg_len": sum(len(msg["content"]) for msg in early_assistant_msgs) / len(early_assistant_msgs),
                "avg_words": sum(len(msg["content"].split()) for msg in early_assistant_msgs) / len(early_assistant_msgs),
                "formality": self._calculate_formality([msg["content"] for msg in early_assistant_msgs]),
                "question_ratio": sum(1 for msg in early_assistant_msgs if "?" in msg["content"]) / len(early_assistant_msgs)
            }
            
            # Calculate how user's style moved toward assistant's style
            accommodation_scores = {
                "length_accom": self._calculate_accommodation_score(
                    early_user_features["avg_len"],
                    recent_user_features["avg_len"],
                    assistant_features["avg_len"]
                ),
                "words_accom": self._calculate_accommodation_score(
                    early_user_features["avg_words"],
                    recent_user_features["avg_words"],
                    assistant_features["avg_words"]
                ),
                "formality_accom": self._calculate_accommodation_score(
                    early_user_features["formality"],
                    recent_user_features["formality"],
                    assistant_features["formality"]
                ),
                "question_accom": self._calculate_accommodation_score(
                    early_user_features["question_ratio"],
                    recent_user_features["question_ratio"],
                    assistant_features["question_ratio"]
                )
            }
            
            # Overall accommodation score (average of individual scores)
            overall_accommodation = sum(accommodation_scores.values()) / len(accommodation_scores)
            
            # Store in session data
            if self.current_user_id in self.user_biographies:
                bio = self.user_biographies[self.current_user_id]
                if "accommodation_history" not in bio:
                    bio["accommodation_history"] = []
                    
                bio["accommodation_history"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "score": overall_accommodation,
                    "details": accommodation_scores
                })
            
            return {
                "status": "success",
                "accommodation": overall_accommodation,
                "details": accommodation_scores,
                "user_change": {
                    "early": early_user_features,
                    "recent": recent_user_features
                }
            }
            
        except Exception as e:
            print(f"Error analyzing accommodation patterns: {str(e)}")
            return {"status": "error", "accommodation": 0.5, "error": str(e)}
        
    def _calculate_formality(self, text_samples):
        """Calculate a simple formality score based on text characteristics."""
        try:
            combined_text = " ".join(text_samples)
            words = combined_text.lower().split()
            
            # Calculate formality based on average word length and other markers
            avg_word_len = sum(len(word) for word in words) / max(1, len(words))
            
            # Count formal markers (longer words, fewer contractions)
            formal_markers = sum(1 for word in words if len(word) > 6)
            contraction_count = sum(1 for word in words if "'" in word)
            
            # Simple formality score
            formality_score = (avg_word_len * 0.3) + (formal_markers/max(1, len(words)) * 0.5) - (contraction_count/max(1, len(words)) * 0.2)
            
            # Normalize between 0 and 1
            return min(max(formality_score / 5, 0), 1)
        except:
            return 0.5  # Default mid-range value
            
    def _calculate_accommodation_score(self, user_early, user_recent, assistant):
        """Calculate accommodation score based on how user style moved toward assistant style."""
        try:
            # If assistant and early user styles were already similar
            if abs(assistant - user_early) < 0.1:
                return 0.5  # Neutral score for already-similar styles
                
            # Calculate distance change
            early_distance = abs(assistant - user_early)
            recent_distance = abs(assistant - user_recent)
            
            # If user moved toward assistant style
            if recent_distance < early_distance:
                # How much of the possible accommodation happened?
                accommodation = 1 - (recent_distance / early_distance)
                return min(0.5 + (accommodation * 0.5), 1.0)  # Scale to 0.5-1.0
            else:
                # User moved away or stayed same
                if early_distance == 0:  # Avoid division by zero
                    return 0.5
                    
                divergence = (recent_distance / early_distance) - 1
                return max(0.5 - (divergence * 0.5), 0.0)  # Scale to 0.0-0.5
        except Exception:
            return 0.5  # Default neutral value

    def save_session_memory(self):
        """Save the session memory, update biographies, and generate insights."""
        try:
            if not self.current_user_id:
                print("No current user ID set, skipping biography update")
                return False
                
            # Only update if we have sufficient interaction
            if self.current_user_id in session_memories:
                memory = session_memories[self.current_user_id]
            else:
                memory = conversation_memory
                
            # Check if we have enough messages to update biography
            if "messages" not in memory or len(memory["messages"]) < 3:
                print(f"Not enough messages for user {self.current_user_id} to update biography")
                return False
                
            # Generate biography if none exists
            if self.current_user_id not in self.user_biographies:
                self.user_biographies[self.current_user_id] = {
                    "first_seen": datetime.datetime.now().isoformat(),
                    "biography": "",
                    "interaction_count": 0,
                    "last_updated": "",
                    "trust_history": []
                }
                
            # Get current biography
            biography = self.user_biographies[self.current_user_id]
            
            # Extract user messages
            user_messages = [msg["content"] for msg in memory["messages"] if msg["role"] == "user"]
            
            # Skip if we don't have enough user messages
            if len(user_messages) < 2:
                print(f"Not enough user messages for {self.current_user_id} to update biography")
                return False
                
            # Increment interaction count
            biography["interaction_count"] += 1
            
            # Only generate new biography every 5 interactions or if it's empty
            if biography["interaction_count"] % 5 == 0 or not biography["biography"]:
                # Prepare context for biography generation
                context = "\n".join(user_messages[-10:] if len(user_messages) > 10 else user_messages)
                
                # Get trust metrics if available
                trust_data = self._get_current_trust_metrics()
                
                # Get accommodation data if available
                accommodation_data = self._analyze_user_accommodation_patterns(self.current_user_id)
                
                # Generate biography using LLM
                prompt = f"""Based on the following conversation history with a user, create a brief profile/biography 
                that captures key aspects of their communication style, interests, and apparent personality.
                Focus on objective observations rather than assumptions.
                
                Trust level: {trust_data.get('trust_level', 0.5)}
                Engagement: {trust_data.get('engagement', 0.5)}
                Rapport: {trust_data.get('rapport', 'neutral')}
                Language accommodation: {accommodation_data.get('accommodation', 0.5)}
                
                Recent conversation:
                {context}
                
                Previous biography (if any):
                {biography['biography']}
                """
                
                # Call LLM for biography generation
                try:
                    updated_bio = self.llm(prompt, max_tokens=200)
                    
                    # Update biography
                    biography["biography"] = updated_bio
                    biography["last_updated"] = datetime.datetime.now().isoformat()
                    
                    print(f"Updated biography for user {self.current_user_id}")
                    return True
                except Exception as e:
                    print(f"Error generating biography: {str(e)}")
                    return False
            else:
                # Just update the timestamp if not generating new biography
                biography["last_updated"] = datetime.datetime.now().isoformat()
                return True
                
        except Exception as e:
            print(f"Error in save_session_memory: {str(e)}")
            return False

    def get_current_user_biography(self, session_id=None):
        """
        Return the current biography for the active user.
        
        Args:
            session_id (str, optional): Session ID to retrieve biography for
            
        Returns:
            str: Formatted biography with metadata
        """
        try:
            # Determine which user ID to use
            user_id = session_id if session_id else self.current_user_id
            
            # Ensure session_id is a string, not a list or any other unhashable type
            if isinstance(user_id, list):
                print(f"Warning: user_id was a list in get_current_user_biography, using first element: {user_id[0]}")
                user_id = user_id[0] if user_id else self.current_user_id
            elif not isinstance(user_id, str) and user_id is not None:
                print(f"Warning: user_id was not a string, converting to string: {str(user_id)}")
                user_id = str(user_id)
            
            # Initialize biographies if not present
            if not hasattr(self, "_user_biographies"):
                self._user_biographies = {}
                
            # If no biographies exist for this user, initialize with default values
            if user_id not in self._user_biographies:
                print(f"No biography found for user {user_id}, creating default")
                self._user_biographies[user_id] = {
                    "first_seen": datetime.datetime.now().isoformat(),
                    "biography": "No detailed information available yet. This appears to be a new user.",
                    "interaction_count": 0,
                    "last_updated": datetime.datetime.now().isoformat(),
                    "trust_history": []
                }
                
            # Generate the formatted biography with metadata
            bio_data = self._user_biographies.get(user_id, {})
            
            # If the user has been seen but has no biography, do an emergency update
            if not bio_data.get("biography") and user_id in session_memories and len(session_memories[user_id]["messages"]) > 2:
                print(f"Emergency biography update for user {user_id}")
                self.update_user_biography(user_id)
                bio_data = self._user_biographies.get(user_id, {})
                
            # Format the biography with metadata
            last_updated = bio_data.get("last_updated", "unknown")
            if isinstance(last_updated, str) and last_updated != "unknown":
                try:
                    # Try to parse and format the timestamp
                    last_updated_dt = datetime.datetime.fromisoformat(last_updated)
                    last_updated = last_updated_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    # If parsing fails, keep original
                    pass
                    
            formatted_bio = f"""USER BIOGRAPHY (Session: {user_id})
Last Updated: {last_updated}
Interactions: {bio_data.get('interaction_count', 0)}
Trust Level: {self._calculate_trust_level(user_id):.2f}

{bio_data.get('biography', 'No biography available.')}
"""
            
            return formatted_bio
            
        except Exception as e:
            print(f"Error retrieving user biography: {str(e)}")
            traceback.print_exc()
            return "Error retrieving user biography."

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
        """Generate learning insights based on past prediction accuracy."""
        try:
            # Skip if we don't have enough predictions
            if len(self.custom_session_memory) < 3:
                return "Insufficient prediction history for learning."
                
            # Get predictions with actual responses
            completed_predictions = [
                entry for entry in self.custom_session_memory 
                if entry.get("twin_prediction") and entry.get("actual_response")
            ]
            
            if len(completed_predictions) < 2:
                return "Not enough completed prediction-actual pairs for learning."
                
            # Choose the most recent completed predictions for analysis
            recent_predictions = completed_predictions[-3:]
            
            # Format prediction-actual pairs for LLM analysis
            prediction_examples = []
            for idx, pred in enumerate(recent_predictions):
                prediction_examples.append(
                    f"Example {idx+1}:\n"
                    f"- Twin predicted: \"{pred['twin_prediction']}\"\n"
                    f"- User actually said: \"{pred['actual_response']}\"\n"
                )
            
            examples_text = "\n".join(prediction_examples)
            
            # Create prompt for analyzing prediction accuracy
            prompt = f"""
            Analyze these examples of digital twin predictions and actual user responses:
            
            {examples_text}
            
            TASK:
            1. Identify patterns in how the digital twin's predictions differ from the actual user responses
            2. Determine what the digital twin is getting right and wrong about this user
            3. Generate specific insights to improve future predictions
            
            Format your analysis as 3-5 concise, specific learning points that would help me predict this user better.
            """
            
            # Get learning insights from LLM
            response = self.llm.invoke([
                SystemMessage(content="You are a prediction improvement specialist helping a digital twin learn from its prediction accuracy."),
                HumanMessage(content=prompt)
            ])
            
            learning_insights = response.content.strip()
            
            # Store the learning insights
            if "prediction_learning" not in self.session_data:
                self.session_data["prediction_learning"] = []
                
            self.session_data["prediction_learning"].append({
                "insights": learning_insights,
                "timestamp": datetime.datetime.now().isoformat(),
                "based_on_examples": len(recent_predictions)
            })
            
            print("Generated new prediction learning insights")
            return learning_insights
            
        except Exception as e:
            print(f"Error generating prediction learning: {str(e)}")
            return "Error analyzing prediction accuracy."

    def _format_conversation(self, messages):
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted

    def _get_current_trust_metrics(self):
        """Extract current trust metrics for the user to inform digital twin decisions."""
        try:
            # Skip if we don't have a user ID
            if not self.current_user_id:
                return {"trust_level": 0.5, "engagement": 0.5, "rapport": "neutral"}
                
            # Get the memory we need to check
            if self.current_user_id in session_memories:
                memory = session_memories[self.current_user_id]
            else:
                memory = conversation_memory
                
            # Extract trust scores if available
            trust_level = 0.5  # Default neutral trust
            if "trust_scores" in memory and memory["trust_scores"]:
                # Get recent trust scores
                recent_scores = memory["trust_scores"][-3:] if len(memory["trust_scores"]) >= 3 else memory["trust_scores"]
                trust_level = sum(score["score"] for score in recent_scores) / len(recent_scores)
            
            # Extract engagement data if available
            engagement = 0.5  # Default neutral engagement
            engagement_trend = "stable"
            if "engagement_depth" in memory and "history" in memory["engagement_depth"]:
                engagement_history = memory["engagement_depth"]["history"]
                if engagement_history:
                    # Get recent engagement scores
                    recent_engagement = engagement_history[-3:] if len(engagement_history) >= 3 else engagement_history
                    engagement = sum(entry["score"] for entry in recent_engagement) / len(recent_engagement)
                    
                    # Calculate trend if we have enough data
                    if len(recent_engagement) >= 3:
                        if recent_engagement[-1]["score"] > recent_engagement[0]["score"]:
                            engagement_trend = "increasing"
                        elif recent_engagement[-1]["score"] < recent_engagement[0]["score"]:
                            engagement_trend = "decreasing"
            
            # Determine rapport level based on trust and engagement
            rapport = "neutral"
            if trust_level > 0.7 and engagement > 0.7:
                rapport = "strong"
            elif trust_level > 0.6 or engagement > 0.6:
                rapport = "positive"
            elif trust_level < 0.4 and engagement < 0.4:
                rapport = "weak"
            
            # Store these metrics in the user biography if we have a current user
            if self.current_user_id in self.user_biographies:
                bio = self.user_biographies[self.current_user_id]
                if "trust_history" not in bio:
                    bio["trust_history"] = []
                    
                bio["trust_history"].append({
                    "avg_trust": trust_level,
                    "avg_engagement": engagement,
                    "rapport": rapport,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
            return {
                "trust_level": trust_level,
                "engagement": engagement,
                "engagement_trend": engagement_trend,
                "rapport": rapport,
                "message_count": len(memory["messages"]) if "messages" in memory else 0
            }
            
        except Exception as e:
            print(f"Error getting trust metrics: {str(e)}")
            return {"trust_level": 0.5, "engagement": 0.5, "rapport": "neutral"}

    def predict_user_response(self, conversation_history, session_id=None):
        """
        Predict how a user will respond to the latest message in a conversation.
        
        Args:
            conversation_history: List of message dicts with role and content or formatted string
            session_id: Session identifier to store with the prediction
            
        Returns:
            str: Predicted user response
        """
        try:
            if not session_id:
                session_id = self.current_user_id
                print(f"Warning: No session_id provided to predict_user_response, using current user: {session_id}")
            
            # Ensure session_id is a string, not a list or any other unhashable type
            if isinstance(session_id, list):
                print(f"Warning: session_id was a list, using first element: {session_id[0]}")
                session_id = session_id[0] if session_id else self.current_user_id
            elif not isinstance(session_id, str) and session_id is not None:
                print(f"Warning: session_id was not a string, converting to string: {str(session_id)}")
                session_id = str(session_id)
            
            # Validate we have conversation history
            if not conversation_history:
                print("Warning: Insufficient conversation history for prediction")
                return ""
            
            # Check if conversation_history is already a formatted string
            if isinstance(conversation_history, str):
                formatted_conversation = conversation_history.split('\n')
                print(f"Using pre-formatted conversation history (string format)")
            else:
                # Format messages for LLM consumption if it's a list of dictionaries
                formatted_conversation = []
                for message in conversation_history:
                    # Check if message is a dictionary or string
                    if isinstance(message, dict):
                        role = message.get("role", "unknown")
                        content = message.get("content", "")
                        formatted_conversation.append(f"{role.upper()}: {content}")
                    elif isinstance(message, str):
                        # If it's a string, assume it's already formatted
                        formatted_conversation.append(message)
                    else:
                        print(f"Warning: Unexpected message format: {type(message)}")
            
            # Create prompt for digital twin prediction
            system_prompt = f"""You are a digital twin of a specific real user with user ID '{session_id}'.
Based on what you know about this user, predict how they would respond to the most recent message in this conversation.
Generate a realistic, detailed response that matches the user's known communication style, interests, and personality."""
            
            # Get user-specific info to enhance prediction accuracy
            user_bio = self.get_current_user_biography(session_id)
            if user_bio and len(user_bio.strip()) > 10:
                system_prompt += f"\n\nUSER PROFILE INFORMATION:\n{user_bio}"
                
            # Add style and language accommodation info if available
            if hasattr(self, "_language_accommodation_scores") and session_id in self._language_accommodation_scores:
                accommodation = self._language_accommodation_scores[session_id]
                system_prompt += f"\n\nUSER COMMUNICATION PATTERNS:\n{json.dumps(accommodation, indent=2)}"
            
            # Get session-specific learning history if available to improve predictions
            if hasattr(self, "learning_history") and session_id in self.learning_history:
                # Use recent learning examples to improve prediction (last 3 entries)
                recent_learnings = self.learning_history[session_id][-3:] if self.learning_history[session_id] else []
                if recent_learnings:
                    system_prompt += "\n\nRECENT INTERACTIONS WITH THIS USER:"
                    for i, entry in enumerate(recent_learnings):
                        system_prompt += f"\n{i+1}. When asked: '{entry.get('user_message', '')}'"
                        system_prompt += f"\n   Response: '{entry.get('assistant_response', '')}'"
                        system_prompt += f"\n   User reaction: '{entry.get('predicted_reaction', '')}'"
            
            # Convert formatted_conversation to string if it's a list
            if isinstance(formatted_conversation, list):
                formatted_conversation_str = "\n".join(formatted_conversation)
            else:
                formatted_conversation_str = formatted_conversation
            
            # Set up prediction prompt with conversation context
            prediction_prompt = "\n\n".join([
                "CONVERSATION HISTORY:",
                formatted_conversation_str,
                "Now, based on this user's profile and the conversation history, predict how they would respond to the last message:"
            ])
            
            # Log the entire prompt for debugging
            print(f"Digital twin prediction prompt: {prediction_prompt[:100]}...")
            
            # Use invoke() method for NebiusChatLLM instead of calling directly
            prediction_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prediction_prompt}
            ]
            prediction_result = self.llm.invoke(prediction_messages)
            
            # Extract content from AIMessage
            if hasattr(prediction_result, 'content'):
                prediction = prediction_result.content.strip()
            else:
                prediction = str(prediction_result).strip()
            
            # Store prediction in memory for later learning
            prediction_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "context": formatted_conversation_str,  # Store formatted conversation
                "prediction": prediction,
                "actual_response": None,  # Will be updated later with real response
                "session_id": session_id
            }
            
            # Initialize session-specific storage if needed
            if not hasattr(self, "session_predictions"):
                self.session_predictions = {}
            
            if session_id not in self.session_predictions:
                self.session_predictions[session_id] = []
            
            # Store prediction in session-specific storage
            self.session_predictions[session_id].append(prediction_record)
            
            # Handle custom_session_memory initialization and storage
            if not hasattr(self, "custom_session_memory"):
                # Initialize as a dictionary
                self.custom_session_memory = {}
                print(f"Initializing custom_session_memory as dictionary")
            elif isinstance(self.custom_session_memory, list):
                # Convert list to dictionary for backward compatibility
                old_memory = self.custom_session_memory
                self.custom_session_memory = {"legacy": old_memory}
                print(f"Converting custom_session_memory from list to dictionary")
            
            # Now it's safe to use session_id as a key
            if session_id not in self.custom_session_memory:
                self.custom_session_memory[session_id] = []
            
            self.custom_session_memory[session_id].append(prediction_record)
            
            print(f"Digital twin predicted response for session {session_id}: {prediction[:50]}...")
            return prediction
            
        except Exception as e:
            print(f"Error in predict_user_response: {str(e)}")
            traceback.print_exc()
            return "I'm not sure how I would respond to that."

    def learn_from_prediction(self, user_message, assistant_response, predicted_reaction, was_refined=False, session_id=None):
        """
        Learn from the prediction and actual response to improve future predictions.
        
        Args:
            user_message (str): Original user message
            assistant_response (str): Assistant's response
            predicted_reaction (str): The predicted reaction
            was_refined (bool): Whether the prediction was refined
            session_id (str): Session ID for the prediction
            
        Returns:
            dict: Learning insights from this prediction
        """
        try:
            print(f"[DIGITAL TWIN] Processing prediction data for session {session_id}: {user_message}")
            
            # Handle session_id type checking
            if isinstance(session_id, list):
                if session_id:  # Non-empty list
                    print(f"[WARNING] Session ID is a list in learn_from_prediction, using first element: {session_id[0]}")
                    session_id = session_id[0]
                else:  # Empty list
                    print(f"[WARNING] Session ID is an empty list in learn_from_prediction, using default")
                    session_id = "default_session"
        
            if not isinstance(session_id, str) and session_id is not None:
                print(f"[WARNING] Session ID is not a string in learn_from_prediction, converting: {session_id}")
                session_id = str(session_id)
        
            if not session_id:
                session_id = "default_session"
                
            # Create a prediction record
            prediction_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "predicted_reaction": predicted_reaction,
                "was_refined": was_refined,
                "actual_response": None,  # Will be updated when actual response is received
                "session_id": session_id
            }
            
            # Add to both global and session-specific memory
            digital_twin_memory["predictions"].append(prediction_entry)
            
            # Initialize storage for session-specific data if not exist
            
            # Check if custom_session_memory is a list instead of a dict - fix for backward compatibility
            if not hasattr(self, "custom_session_memory"):
                self.custom_session_memory = {}
                print(f"[DIGITAL TWIN] Initialized custom_session_memory as empty dictionary")
            elif isinstance(self.custom_session_memory, list):
                # Convert list to dictionary for better session management
                old_memory = self.custom_session_memory
                self.custom_session_memory = {}
                # Store old entries under a generic key for backward compatibility
                self.custom_session_memory["legacy_entries"] = old_memory
                print(f"[DIGITAL TWIN] Converted custom_session_memory from list to dictionary with {len(old_memory)} legacy entries")
            
            # Initialize learning_history as a dictionary if it doesn't exist
            if not hasattr(self, "learning_history"):
                self.learning_history = {}
                print(f"[DIGITAL TWIN] Initialized learning_history as empty dictionary")
                
            # Initialize session_predictions as a dictionary if it doesn't exist  
            if not hasattr(self, "session_predictions"):
                self.session_predictions = {}
                print(f"[DIGITAL TWIN] Initialized session_predictions as empty dictionary")
            
            # Ensure dictionaries contain the session entry
            if session_id not in self.custom_session_memory:
                self.custom_session_memory[session_id] = []
                print(f"[DIGITAL TWIN] Created new session memory for {session_id}")
                
            if session_id not in self.learning_history:
                self.learning_history[session_id] = []
                print(f"[DIGITAL TWIN] Created new learning history for {session_id}")
                
            if session_id not in self.session_predictions:
                self.session_predictions[session_id] = []
                print(f"[DIGITAL TWIN] Created new session predictions for {session_id}")
            
            # Add to session-specific memory
            self.custom_session_memory[session_id].append(prediction_entry)
            self.learning_history[session_id].append(prediction_entry)
            self.session_predictions[session_id].append(prediction_entry)
            
            # Update learning statistics
            if not hasattr(self, "learning_stats"):
                self.learning_stats = {
                    "predictions_made": 0,
                    "refinements_triggered": 0,
                    "prediction_accuracy": []
                }
                
            self.learning_stats["predictions_made"] = self.learning_stats.get("predictions_made", 0) + 1
            if was_refined:
                self.learning_stats["refinements_triggered"] = self.learning_stats.get("refinements_triggered", 0) + 1
                
            print(f"[DIGITAL TWIN] Added prediction to memory for session {session_id}")
            print(f"[DIGITAL TWIN] Learning stats: {self.learning_stats}")
            
            # Try to generate learning insights
            try:
                insights = self._calculate_learning_insights(session_id)
                return insights
            except Exception as e:
                print(f"[DIGITAL TWIN] Error generating learning insights: {str(e)}")
                return {"error": str(e)}
            
        except Exception as e:
            print(f"[DIGITAL TWIN] Error adding prediction to memory: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    def _calculate_response_similarity(self, text1, text2):
        """Calculate a basic similarity score between two text strings."""
        try:
            # Simple check for empty strings
            if not text1 or not text2:
                return 0.0
                
            # Tokenize into words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
                
            return intersection / union
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _calculate_trust_level(self, user_id):
        """
        Calculate trust level for a specific user based on interaction history.
        
        Args:
            user_id (str): The user ID to calculate trust for
            
        Returns:
            float: Trust level between 0 and 1
        """
        try:
            # Default trust level for new users
            default_trust = 0.5
            
            # If no user or no trust history, return default
            if not user_id:
                return default_trust
                
            # Check if we have trust metrics for this user
            if hasattr(self, "_trust_metrics") and isinstance(self._trust_metrics, dict) and user_id in self._trust_metrics:
                return self._trust_metrics[user_id].get("trust_level", default_trust)
                
            # Calculate basic trust from biography if available
            if hasattr(self, "_user_biographies") and user_id in self._user_biographies:
                bio = self._user_biographies[user_id]
                interaction_count = bio.get("interaction_count", 0)
                
                # Simple trust calculation based on interaction count
                # More interactions = higher base trust
                base_trust = min(0.3 + (interaction_count * 0.05), 0.7)
                
                # Add trust history if available
                history = bio.get("trust_history", [])
                if history:
                    recent_history = history[-5:] if len(history) > 5 else history
                    avg_history_trust = sum(entry.get("level", 0.5) for entry in recent_history) / len(recent_history)
                    
                    # Combine base trust with history (weighted)
                    combined_trust = (base_trust * 0.4) + (avg_history_trust * 0.6)
                    return combined_trust
                
                return base_trust
                
            return default_trust
            
        except Exception as e:
            print(f"Error calculating trust level: {str(e)}")
            return 0.5

    # Add the missing record_interaction_metric method
    def record_interaction_metric(self, metric_name, value):
        """
        Record a user interaction metric for the current session.
        
        Args:
            metric_name (str): Name of the metric to record (e.g., "clicked_link")
            value (float): Value of the metric
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize metrics dictionary if not already done
            if not hasattr(self, "interaction_metrics"):
                self.interaction_metrics = {}
            
            # Get current user ID
            user_id = self.current_user_id
            if not user_id:
                print("Warning: No user ID set for recording interaction metric")
                return False
                
            # Initialize user metrics if not already done
            if user_id not in self.interaction_metrics:
                self.interaction_metrics[user_id] = {}
                
            # Initialize specific metric if not already done
            if metric_name not in self.interaction_metrics[user_id]:
                self.interaction_metrics[user_id][metric_name] = []
                
            # Add the metric value with timestamp
            metric_entry = {
                "value": value,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.interaction_metrics[user_id][metric_name].append(metric_entry)
            
            print(f"Recorded interaction metric '{metric_name}' with value {value} for user {user_id}")
            return True
        except Exception as e:
            print(f"Error recording interaction metric: {str(e)}")
            traceback.print_exc()
            return False

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
            # Check if response is empty
            if not cleaned_response or cleaned_response.isspace():
                print("Warning: Empty response from LLM when generating link")
                unique_id = str(int(time.time()))[-5:]
                return f"@https://youtube.com/resource-{unique_id}"
                
            # Properly handle JSON parsing
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
            print(f"Raw response content: '{cleaned_response}'")
            # Create a unique path based on timestamp
            unique_id = str(int(time.time()))[-5:]
            return f"@https://youtube.com/resource-{unique_id}"
            
    except Exception as e:
        print(f"Error generating contextual link: {str(e)}")
        # Create a unique path based on timestamp
        unique_id = str(int(time.time()))[-5:]
        return f"@https://youtube.com/resource-{unique_id}"

def dynamic_link(final_message: str, current_stage=None) -> str:
    """Replace any text enclosed in square brackets with YouTube URLs, but only in link stages."""
    import re
    
    # Define link stages where links are allowed
    link_stages = ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]
    non_link_stages = ["INITIAL_ENGAGEMENT", "RAPPORT_BUILDING", "TRUST_DEVELOPMENT"]
    
    # If we're not in a link stage, we should not process any link placeholders
    if current_stage not in link_stages:
        print(f"Skipping link processing - not in a resource stage (current: {current_stage})")
        return final_message
    
    print(f"Processing links for stage: {current_stage}")
    
    # Use the specific YouTube URL provided
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Find all patterns that match [Link], [Insert Link] or any text in square brackets
    pattern = r'\[(.*?)\]'
    
    # Count how many replacements we've made
    replacements_made = 0
    
    # Function to handle each match
    def replace_match(match):
        nonlocal replacements_made
        replacements_made += 1
        placeholder_text = match.group(0)
        print(f"Found link placeholder: {placeholder_text}")
        # Return the actual URL text so it looks reliable
        return youtube_url
    
    # Replace all matched patterns
    result = re.sub(pattern, replace_match, final_message)
    
    # If we're in a link stage and no replacement was made, consider appending a link
    if replacements_made == 0:
        if not any(url in final_message for url in ["http://", "https://", "youtube.com"]):
            result += f" {youtube_url}"
            replacements_made += 1
            print("No link placeholders found, appending link to end of message")
    
    print(f"Dynamic link processed: {replacements_made} replacements made, stage: {current_stage}")
    return result

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

def get_current_resource_url():
    """Get the most recently shown resource URL from the conversation."""
    try:
        # First try to get from conversation_memory
        url = conversation_memory.get("current_resource_url")
        if url:
            print(f"Using stored URL: {url}")
            return url
        
        # If not found, try to extract from the last message
        for msg in reversed(conversation_memory["messages"]):
            if msg["role"] == "INFLUENCER":
                # Try to extract URL from the message content
                match = re.search(r'@https://youtube\.com/([^\s\)\]]+)', msg["content"])
                if match:
                    full_url = f"https://youtube.com/{match.group(1)}"
                    print(f"Extracted URL from message: {full_url}")
                    return full_url
                
                # Try alternative format with markdown
                match = re.search(r'\]\(https://youtube\.com/([^\)]+)\)', msg["content"])
                if match:
                    full_url = f"https://youtube.com/{match.group(1)}"
                    print(f"Extracted URL from markdown: {full_url}")
                    return full_url
        
        # If no URL found, check topic_analysis
        if "topic_analysis" in conversation_memory and conversation_memory["topic_analysis"]:
            latest_analysis = conversation_memory["topic_analysis"][-1]
            if "url_path" in latest_analysis:
                full_url = f"https://youtube.com/{latest_analysis['url_path']}"
                print(f"Using URL from topic analysis: {full_url}")
                return full_url
                
        print("No URL found, using default")
        return "https://youtube.com/resource-default"
        
    except Exception as e:
        print(f"Error retrieving resource URL: {str(e)}")
        return "https://youtube.com/resource-default"

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

def process_message(message, session_id=None):
    """
    Process a message from the user and return a response.
    
    Args:
        message (str): User message
        session_id (str, optional): Session ID
        
    Returns:
        str: Response message
    """
    try:
        # Log the start of processing
        print(f"\n===== PROCESSING MESSAGE: '{message}' =====")
        print(f"SESSION ID: {session_id}")
        
        # Skip very short messages or just punctuation
        if len(message.strip()) <= 1 and not message.strip().isalnum():
            print("SKIPPING: Message too short")
            return "..."
            
        # Create a valid session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            print(f"[SESSION MANAGER] Created new session ID: {session_id}")
        
        # Create or get session-specific memory
        if session_id not in session_memories:
            session_memories[session_id] = create_conversation_memory_for_session(session_id)
            print(f"[MEMORY MANAGER] Created new memory for session: {session_id}")
            
        memory = session_memories[session_id]
        
        # Get the digital twin for this session
        digital_twin = digital_twin_manager.get_twin_for_session(session_id)
        print(f"[DIGITAL TWIN] Active twin for session: {session_id}, User ID: {digital_twin.current_user_id}")
        
        # Add the user message to memory
        update_memory_with_user_message(message, memory, session_id)
        print(f"[MEMORY MANAGER] Added user message to memory: '{message}'")
        
        # Get current stage
        current_stage = memory["current_stage"]
        print(f"[STAGE MANAGER] Current stage: {current_stage}")
        
        # Format conversation history for LLM
        formatted_history = format_conversation_history(memory["messages"], current_stage)
        print(f"[HISTORY FORMATTER] Formatted {len(memory['messages'])} messages for context")
        
        # Determine next conversation stage
        next_stage = determine_next_stage(current_stage, message, "", False, session_id)
        if next_stage != current_stage:
            update_conversation_stage(next_stage, session_id)
            print(f"[STAGE MANAGER] Stage transition: {current_stage} → {next_stage}")
            current_stage = next_stage
            
        # Force biography updates every 2 messages to ensure it stays current
        message_count = len([msg for msg in memory["messages"] if msg["role"] == "USER"])
        if message_count % 2 == 0 and message_count > 0:
            print(f"[BIOGRAPHY MANAGER] Forcing biography update after {message_count} messages")
            biography_success = digital_twin.update_user_biography(session_id)
            print(f"[BIOGRAPHY MANAGER] Update {'successful' if biography_success else 'failed'}")
    
        # Get personalized response based on Digital Twin's knowledge
        try:
            # 1. Have the twin predict how the user will respond
            print(f"[DIGITAL TWIN] Predicting user response...")
            prediction = digital_twin.predict_user_response(formatted_history, session_id)
            print(f"[DIGITAL TWIN] Prediction: '{prediction[:50]}...'")
            
            # 2. Generate a response based on current stage
            print(f"[INFLUENCER] Generating initial response for stage {current_stage}...")
            initial_response = get_llm_response(
                message, 
                formatted_history, 
                current_stage
            )
            print(f"[INFLUENCER] Initial response: '{initial_response[:50]}...'")
            
            # 3. Re-enabled refinement process based on digital twin prediction
            # Evaluate if refinement is needed
            needs_refinement_result = needs_refinement(initial_response, prediction, threshold=0.4)
            
            if needs_refinement_result:
                print("Response needs refinement based on prediction")
        
            # Get style guidance based on user's communication patterns
                user_messages = [msg for msg in memory["messages"] if msg["role"] == "USER"]
                style_guidance = analyze_user_communication_style(user_messages)
        
                # Create refinement prompt
                refinement_prompt = f"""
                You are responding to a user message in a conversation.
                
                User's message: "{message}"
                
                Your initial response: "{initial_response}"
                
                The user will likely respond: "{prediction}"
                
                Based on this predicted reaction, please refine your response to be more engaging and aligned with the user's interests.
                
                Current conversation stage: {current_stage}
                
                User's communication style:
                - Average message length: {style_guidance.get('avg_length', 'medium')}
                - Formality level: {style_guidance.get('formality', 'neutral')}
                - Uses emoji: {style_guidance.get('uses_emoji', False)}
                
                Improve your response to better match user expectations and increase engagement.
                Provide ONLY the final refined response without any explanations or meta-commentary.
                """
                
                # Get refinement from LLM
                refinement_response = influencer_llm.invoke([
                            SystemMessage(content="You are an expert assistant refining responses based on predictions."),
                    HumanMessage(content=refinement_prompt)
                ])
        
                # Use the refined response
                final_response = refinement_response.content.strip()
                
                # Mark as refined for tracking
                refinement_data = {
                    "was_refined": True,
                    "original_response": initial_response,
                    "prediction": prediction,
                    "reasoning": "Prediction indicated user would not be fully engaged"
                }
                
                # Add to memory with refinement data
                assistant_message = {
                    "role": "assistant",
                    "content": final_response,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "refinement_data": refinement_data
                }
                memory["messages"].append(assistant_message)
                
                print("Using refined response")
            else:
                # No refinement needed, use initial response
                final_response = initial_response
                
                # Add to memory without refinement data
                assistant_message = {
                    "role": "assistant",
                    "content": final_response,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "refinement_data": {"was_refined": False}
                }
                memory["messages"].append(assistant_message)
                
                print("Using initial response (no refinement needed)")
            
            # 4. Process dynamic links in the content
            # Check if we're in a link stage
            link_stages = ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]
            is_link_stage = current_stage in link_stages
            
            if is_link_stage:
                print(f"[LINK PROCESSOR] In link stage: {current_stage}, processing links")
                
                # Check if there are dynamic links to process
                if contains_dynamic_link(final_response):
                    print(f"[LINK PROCESSOR] Found link placeholders in response")
                    final_response = process_dynamic_links(final_response, current_stage, session_id)
                else:
                    # Force add a link if in link stage and no links present
                    print(f"[LINK PROCESSOR] No link placeholders found, adding one")
                    final_response += " For more information on this topic, check out this resource: [Link]"
                    final_response = process_dynamic_links(final_response, current_stage, session_id)
            else:
                print(f"[LINK PROCESSOR] Not in link stage ({current_stage}), skipping link processing")
            
            # Learn from this prediction for future improvements
            try:
                # Call learn_from_prediction with individual parameters instead of a dictionary
                digital_twin.learn_from_prediction(
                    user_message=message,
                    assistant_response=final_response, 
                    predicted_reaction=prediction,
                    was_refined=needs_refinement_result,
                    session_id=session_id
                )
            except Exception as e:
                print(f"Error in learn_from_prediction: {str(e)}")
                traceback.print_exc()
            
            # Return the final response
            return final_response
            
        except Exception as e:
            error_msg = f"Error processing message details: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return "I apologize, but I encountered an error processing your message. Please try again."
            
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return "I apologize, but I encountered an error processing your message. Please try again."

def update_digital_twin_actual_response(actual_user_response, session_id=None):
    """
    Update the digital twin with the actual user response for a session.
    
    Args:
        actual_user_response (str): The actual response from the user
        session_id (str): The session ID to update predictions for
        
    Returns:
        bool: True if a matching prediction was found and updated, False otherwise
    """
    try:
        # Handle session_id type checking
        if isinstance(session_id, list):
            if session_id:  # Non-empty list
                print(f"[WARNING] Session ID is a list in update_digital_twin_actual_response, using first element: {session_id[0]}")
                session_id = session_id[0]
            else:  # Empty list
                print(f"[WARNING] Session ID is an empty list in update_digital_twin_actual_response, using default")
                session_id = "default_session"
        
        if not isinstance(session_id, str) and session_id is not None:
            print(f"[WARNING] Session ID is not a string in update_digital_twin_actual_response, converting: {session_id}")
            session_id = str(session_id)
        
        if not session_id:
            session_id = "default_session"
        
        print(f"[DIGITAL TWIN] Updating actual response for session {session_id}: '{actual_user_response}'")
        
        # Get the digital twin for this session
        if session_id:
            twin = digital_twin_manager.get_twin_for_session(session_id)
            if twin:
                print(f"[DIGITAL TWIN] Using session-specific twin for {session_id}")
        else:
            twin = digital_twin
            
        # Set user for this twin if not already set
        if hasattr(twin, "set_user_for_session") and twin.current_user_id != session_id:
            twin.set_user_for_session(session_id)
            print(f"[DIGITAL TWIN] Set user for twin: {session_id}")
            
        # Find all predictions for this session that need to be updated
        found_updates = 0
        
        # Check if custom_session_memory exists and retrieve it
        if hasattr(twin, "custom_session_memory"):
            # If it's a dictionary (preferred)
            if isinstance(twin.custom_session_memory, dict):
                if session_id in twin.custom_session_memory:
                    session_predictions = twin.custom_session_memory[session_id]
                    print(f"[DIGITAL TWIN] Found {len(session_predictions)} predictions for session {session_id}")
                    
                    # Update all predictions that don't have an actual response
                    for pred in reversed(session_predictions):
                        if pred.get("actual_response") is None:
                            pred["actual_response"] = actual_user_response
                            print(f"[DIGITAL TWIN] Updated prediction with actual response")
                            found_updates += 1
                else:
                    print(f"[DIGITAL TWIN] No session memory found for session {session_id}")
            
            # If it's a list (legacy format)
            elif isinstance(twin.custom_session_memory, list):
                print(f"[DIGITAL TWIN] Using legacy session memory format (list)")
                
                # Find predictions for this session or with no session ID
                for pred in reversed(twin.custom_session_memory):
                    pred_session = pred.get("session_id")
                    
                    # Match if the prediction has the same session ID or no session ID
                    if (pred_session == session_id or pred_session is None) and pred.get("actual_response") is None:
                        pred["actual_response"] = actual_user_response
                        pred["session_id"] = session_id  # Ensure the session ID is set
                        print(f"[DIGITAL TWIN] Updated legacy prediction with actual response")
                        found_updates += 1
                        
            else:
                print(f"[DIGITAL TWIN] Unexpected type for custom_session_memory: {type(twin.custom_session_memory)}")
                
        # Also check session_predictions if available
        if hasattr(twin, "session_predictions") and isinstance(twin.session_predictions, dict):
            if session_id in twin.session_predictions:
                session_predictions = twin.session_predictions[session_id]
                print(f"[DIGITAL TWIN] Checking secondary storage with {len(session_predictions)} predictions")
                
                for pred in reversed(session_predictions):
                    if pred.get("actual_response") is None:
                        pred["actual_response"] = actual_user_response
                        print(f"[DIGITAL TWIN] Updated prediction in secondary storage")
                        found_updates += 1
        
        # Check learning_history if available
        if hasattr(twin, "learning_history") and isinstance(twin.learning_history, dict):
            if session_id in twin.learning_history:
                learning_entries = twin.learning_history[session_id]
                print(f"[DIGITAL TWIN] Checking learning history with {len(learning_entries)} entries")
                
                for entry in reversed(learning_entries):
                    if entry.get("actual_response") is None:
                        entry["actual_response"] = actual_user_response
                        print(f"[DIGITAL TWIN] Updated learning history entry with actual response")
                        found_updates += 1
                        
        # Generate insights from updated predictions
        if found_updates > 0 and hasattr(twin, "_calculate_learning_insights"):
            try:
                insights = twin._calculate_learning_insights(session_id)
                print(f"[DIGITAL TWIN] Generated learning insights: {insights}")
            except Exception as e:
                print(f"[DIGITAL TWIN] Error generating learning insights: {str(e)}")
                
        print(f"[DIGITAL TWIN] Updated {found_updates} predictions with actual response")
        return found_updates > 0
        
    except Exception as e:
        print(f"[DIGITAL TWIN] Error updating actual response: {str(e)}")
        traceback.print_exc()
        return False

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
    try:
        # Get session ID from state
        session_id = state.get("session_id", None)
        
        # Get the appropriate memory
        if session_id and session_id in session_memories:
            memory = session_memories[session_id]
        else:
            memory = conversation_memory
        
        # 1. Increment link clicks counter
        memory["link_clicks"] += 1
        
        # 2. Add system message acknowledging the click
        click_message = "Link click recorded! How was the content? Feel free to share your thoughts or ask any questions."
        state["conv"].append(("SYSTEM", click_message))
        chat_history.append(("System", click_message))
        
        # 3. Update metrics - handle the case where the method might not exist
        try:
            twin = digital_twin_manager.get_twin_for_session(session_id)
            if twin and hasattr(twin, "record_interaction_metric"):
                twin.record_interaction_metric("clicked_link", 1.0)
            else:
                print("Warning: Digital twin doesn't have record_interaction_metric method")
        except Exception as e:
            print(f"Error recording interaction metric: {str(e)}")
            # Continue execution despite this error
        
        # 4. Transition to completion stage if not already there
        current_stage = memory["current_stage"]
        if current_stage != "SESSION_COMPLETION":
            update_conversation_stage("SESSION_COMPLETION", session_id=session_id)
        
        # 5. Save conversation data - handle potential errors
        try:
            save_conversation({"conv": state["conv"]}, session_id=session_id)
            print("Successfully saved conversation after link click")
        except Exception as e:
            print(f"Error saving conversation after link click: {str(e)}")
            traceback.print_exc()
            # Continue execution despite this error
        
        # Return updated state and trigger feedback form
        return chat_history, state, json.dumps({
            "status": "Link clicked recorded",
            "message": click_message
        }), update_stage_display("SESSION_COMPLETION")
        
    except Exception as e:
        print(f"Error in record_link_click: {str(e)}")
        traceback.print_exc()
        # Return a graceful error state
        error_message = "There was an issue recording your link click. Please try again."
        chat_history.append(("System", error_message))
        return chat_history, state, json.dumps({
            "status": "error", 
            "message": str(e)
        }), update_stage_display(memory.get("current_stage", "INITIAL_ENGAGEMENT"))

#####################################
# 9. New Function: Record Feedback
#####################################

def record_feedback(feedback_text, chat_history, state):
    """
    Record user feedback and save the conversation data.
    
    Args:
        feedback_text (str): The feedback text from the user
        chat_history (list): The current chat history
        state (dict): The current UI state
        
    Returns:
        tuple: Updated chat history, state, status, and stage display
    """
    try:
        # Get session ID from state
        session_id = state.get("session_id", None)
        
        # Get the appropriate memory
        if session_id and session_id in session_memories:
            memory = session_memories[session_id]
        else:
            memory = conversation_memory
            
        # Add feedback to UI and state with proper labeling
        state["conv"].append(("FEEDBACK: " + feedback_text, None))
        chat_history.append(("Feedback", feedback_text))
            
        # Update to SESSION_COMPLETION stage
        memory["current_stage"] = "SESSION_COMPLETION"
        memory["stage_history"].append({
            "stage": "SESSION_COMPLETION",
            "timestamp": datetime.datetime.now().isoformat(),
            "message_count": len(memory["messages"]),
            "trigger": "feedback_submitted"
        })
        
        # Create a comprehensive data package to save
        save_data = {
            "conv": state["conv"],
            "timestamp": datetime.datetime.now().isoformat(),
            "session_metrics": {
                "engagement_metrics": get_quality_metrics(session_id),
                "trust_scores": memory["trust_scores"],
                "link_clicks": memory["link_clicks"],
                "stage_history": memory["stage_history"]
            },
            "feedback": feedback_text
        }
        
        # Save the comprehensive data - handle errors gracefully
        try:
            saved_filename = save_conversation(save_data, session_id=session_id)
            print(f"[FEEDBACK] Conversation saved to {saved_filename} with feedback")
            
            # Update the digital twin with feedback information
            try:
                twin = digital_twin_manager.get_twin_for_session(session_id)
                if twin and hasattr(twin, "record_interaction_metric"):
                    twin.record_interaction_metric("provided_feedback", 1.0)
                    print(f"[FEEDBACK] Recorded feedback metric for session {session_id}")
            except Exception as twin_error:
                print(f"[FEEDBACK] Error updating digital twin with feedback: {str(twin_error)}")
                # Continue despite this error
            
            # Add a confirmation message with proper labeling
            chat_history.append((None, f"Thank you for your feedback! Conversation data saved successfully."))
            state["conv"].append((None, "SYSTEM: Thank you for your feedback! Conversation data saved successfully."))
        except Exception as save_error:
            print(f"[FEEDBACK ERROR] Failed to save conversation with feedback: {str(save_error)}")
            # Add an error message but continue
            chat_history.append((None, f"Thank you for your feedback! Note: There was an issue saving the conversation data."))
            state["conv"].append((None, "SYSTEM: Thank you for your feedback! Note: There was an issue saving the conversation data."))
            saved_filename = "error_saving"
        
        return chat_history, state, json.dumps({
            "status": "feedback_recorded", 
            "filename": saved_filename
        }), update_stage_display("SESSION_COMPLETION")
        
    except Exception as e:
        error_msg = f"Error processing feedback: {str(e)}"
        print(f"[FEEDBACK ERROR] {error_msg}")
        traceback.print_exc()
        
        # Add error message to chat
        chat_history.append((None, f"Error saving feedback. Please try again."))
        state["conv"].append((None, "SYSTEM: Error saving feedback. Please try again."))
        
        # Try to save error information
        try:
            error_dir = os.path.join(STORAGE_DIR, "errors")
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, f"feedback_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(error_file, 'w') as f:
                f.write(f"Feedback error: {str(e)}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Feedback text: {feedback_text}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            print(f"[FEEDBACK] Saved error information to {error_file}")
        except Exception as file_error:
            print(f"[FEEDBACK ERROR] Could not save error information: {str(file_error)}")
            
        return chat_history, state, json.dumps({
            "error": error_msg, 
            "status": "error"
        }), update_stage_display(memory.get("current_stage", "SESSION_COMPLETION"))

#####################################
# 10. Session Reset Function
#####################################

def reset_session(session_id=None):
    """
    Clear memory structures for a specific session or create a new session.
    
    Args:
        session_id: Optional session ID to reset. If None, resets the default session.
        
    Returns:
        tuple: Empty state dict, reset message, and updated stage display
    """
    if session_id and session_id in session_memories:
        # Reset an existing session
        session_memories[session_id] = create_conversation_memory_for_session(session_id)
        print(f"Reset existing session: {session_id}")
    elif session_id:
        # Create a new session with this ID
        session_memories[session_id] = create_conversation_memory_for_session(session_id)
        print(f"Created new session: {session_id}")
    else:
        # Fall back to reset global memory
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
        if hasattr(digital_twin, "custom_session_memory"):
            digital_twin.custom_session_memory = []
        print("Reset default session")
    
    return {"conv": [], "session_id": session_id}, "Session reset.", update_stage_display("INITIAL_ENGAGEMENT")

#####################################
# 11. Gradio UI & Auto-Scrolling Setup
#####################################

STORAGE_DIR = "conversation_logs"
os.makedirs(STORAGE_DIR, exist_ok=True)

def update_stage_display(current_stage):
    """Generate HTML representation of the current conversation stage."""
    # Define all possible stages and their properties
    all_stages = [
        {"id": 1, "name": "INITIAL_ENGAGEMENT", "display": "Initial Engagement", "description": "First contact and greeting"},
        {"id": 2, "name": "RAPPORT_BUILDING", "display": "Rapport Building", "description": "Building connection through shared interests"},
        {"id": 3, "name": "TRUST_DEVELOPMENT", "display": "Trust Development", "description": "Deepening trust through validation"},
        {"id": 4, "name": "LINK_INTRODUCTION", "display": "Link Introduction", "description": "Introducing resources naturally"},
        {"id": 5, "name": "LINK_REINFORCEMENT", "display": "Link Reinforcement", "description": "Reinforcing resource value"},
        {"id": 6, "name": "SESSION_COMPLETION", "display": "Session Completion", "description": "Wrapping up conversation"}
    ]
    
    # If no current stage is provided, show all stages without active indicator
    if not current_stage:
        current_stage = "INITIAL_ENGAGEMENT"  # Default
    
    # Normalize stage name if it's not in expected format
    if current_stage.upper() != current_stage:
        # Convert to uppercase
        current_stage = current_stage.upper()
        print(f"[STAGE DISPLAY] Normalized stage name to: {current_stage}")
    
    # Find current stage position
    current_position = 1  # Default to first stage if not found
    for stage in all_stages:
        if stage["name"] == current_stage:
            current_position = stage["id"]
            break
    
    # Calculate progress percentage (from 0 to 100%)
    progress_percentage = ((current_position - 1) / 5) * 100  # 5 is max steps (6 stages, but 5 transitions)
    
    # Generate HTML for stage indicator
    html = f'<div class="stage-container"><h4>Current Stage: {current_stage}</h4>'
    html += f'<div class="stage-progress-text">Progress: {current_position}/6 (Stage {current_position}, {progress_percentage:.1f}%)</div>'
    html += '<div class="stage-progress-bar">'
    
    # Generate the stages
    for stage in all_stages:
        is_active = stage["name"] == current_stage
        is_completed = stage["id"] < current_position
        
        # Set the appropriate class for styling
        stage_class = "active-stage" if is_active else ("completed-stage" if is_completed else "future-stage")
        
        html += f'<div class="stage {stage_class}" title="{stage["description"]}">'
        html += f'<div class="stage-label">{stage["display"]}</div>'
        html += '</div>'
    
    html += '</div></div>'
    
    print(f"Stage display - Current: {current_stage}, Position: {current_position}, Progress: {progress_percentage:.1f}%")
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


def save_conversation(conversation_data, filename=None, session_id=None):
    """
    Save conversation data to JSON files in a dedicated session folder with refinement statistics.
    
    Args:
        conversation_data (dict): Conversation data to save
        filename (str, optional): Custom filename to use
        session_id (str, optional): Session ID to determine which memory to use
        
    Returns:
        str: Path to the saved conversation file
    """
    try:
        # Get session-specific memory if available
        if session_id and session_id in session_memories:
            memory = session_memories[session_id]
            print(f"[SAVE] Using memory for session {session_id}")
        else:
            memory = conversation_memory  # Fallback to global memory
            print(f"[SAVE] Using global memory (no session ID or not found)")
        
        # Generate timestamp for folder name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session folder
        session_folder = os.path.join(STORAGE_DIR, f"session_{timestamp}")
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)
            print(f"[SAVE] Created session folder: {session_folder}")
        
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
        
        print(f"[SAVE] Setting up files in {session_folder}")
        
        # Calculate refinement statistics
        try:
            refinement_stats = {
                "total_responses": len([m for m in memory["messages"] if m["role"] == "assistant"]),
                "refined_responses": len([m for m in memory["messages"] 
                                    if m["role"] == "assistant" and 
                                    m.get("refinement_data", {}).get("was_refined", False)]),
                "refinement_reasons": {}
            }
            
            # Calculate refinement rate
            if refinement_stats["total_responses"] > 0:
                refinement_stats["refinement_rate"] = refinement_stats["refined_responses"] / refinement_stats["total_responses"]
            else:
                refinement_stats["refinement_rate"] = 0.0
            
            # Collect statistics on refinement reasons
            for msg in memory["messages"]:
                if msg["role"] == "assistant" and msg.get("refinement_data", {}).get("was_refined", False):
                    reasons = msg["refinement_data"].get("reasoning", "")
                    for reason in reasons.split(", "):
                        if reason:
                            refinement_stats["refinement_reasons"][reason] = refinement_stats["refinement_reasons"].get(reason, 0) + 1
        except Exception as e:
            print(f"[SAVE ERROR] Error calculating refinement statistics: {str(e)}")
            refinement_stats = {"error": str(e), "total_responses": 0, "refined_responses": 0, "refinement_rate": 0.0}
        
        # Create enhanced conversation data with refinement information
        try:
            enhanced_conversation_data = conversation_data.copy() if isinstance(conversation_data, dict) else {"conv": conversation_data}
            enhanced_conversation_data["refinement_stats"] = refinement_stats
            enhanced_conversation_data["metrics_history"] = memory.get("metrics_history", [])
            enhanced_conversation_data["save_timestamp"] = datetime.datetime.now().isoformat()
            enhanced_conversation_data["session_id"] = session_id
            
            # Format the conversation in a more structured way - Improved format with consistent labels
            formatted_conversation = []
            original_conv = enhanced_conversation_data.get("conv", [])
            
            # Convert the tuple format to a more readable JSON structure with explicit labels
            for i, msg in enumerate(original_conv):
                try:
                    if isinstance(msg, tuple) and len(msg) == 2:
                        user_msg, assistant_msg = msg
                        
                        # Create message entry with metadata and explicit role label
                        if user_msg is not None:
                            # Handle special role prefixes
                            if isinstance(user_msg, str) and user_msg.startswith("FEEDBACK:"):
                                role = "FEEDBACK"
                                content = user_msg[len("FEEDBACK:"):].strip()
                            elif isinstance(user_msg, str) and user_msg == "SYSTEM":
                                role = "SYSTEM"
                                content = ""
                            else:
                                role = "USER"
                                content = user_msg
                                
                            formatted_conversation.append({
                                "id": f"msg_{i}_user",
                                "role": role,
                                "content": content,
                                "timestamp": datetime.datetime.now().isoformat(),
                                "metadata": {"session_id": session_id}
                            })
                            
                        if assistant_msg is not None:
                            # Extract actual content if it has a prefix like "ASSISTANT: "
                            content = assistant_msg
                            if isinstance(content, str) and "ASSISTANT: " in content:
                                content = content.replace("ASSISTANT: ", "")
                            # Check if this is a system message
                            elif isinstance(assistant_msg, str) and assistant_msg.startswith("SYSTEM:"):
                                role = "SYSTEM"
                                content = assistant_msg[len("SYSTEM:"):].strip()
                            else:
                                role = "ASSISTANT"
                                
                            formatted_conversation.append({
                                "id": f"msg_{i}_assistant",
                                "role": role,
                                "content": content,
                                "timestamp": datetime.datetime.now().isoformat(),
                                "metadata": {
                                    "session_id": session_id,
                                    "stage": memory.get("current_stage", "unknown")
                                }
                            })
                except Exception as msg_error:
                    print(f"[SAVE ERROR] Error formatting message {i}: {str(msg_error)}")
                    # Add an error message instead
                    formatted_conversation.append({
                        "id": f"msg_{i}_error",
                        "role": "SYSTEM",
                        "content": f"Error processing message: {str(msg_error)}",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
            # Add the structured conversation to the enhanced data
            enhanced_conversation_data["structured_conversation"] = formatted_conversation
            
            # Also reformat the original conv with consistent labels
            reformatted_conv = []
            for msg in original_conv:
                if isinstance(msg, tuple) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    
                    # Create a new tuple with explicit role labels
                    if user_msg is not None:
                        if isinstance(user_msg, str) and user_msg.startswith("FEEDBACK:"):
                            reformatted_conv.append(("FEEDBACK", user_msg[len("FEEDBACK:"):].strip()))
                        elif isinstance(user_msg, str) and user_msg == "SYSTEM":
                            reformatted_conv.append(("SYSTEM", ""))
                        else:
                            reformatted_conv.append(("USER", user_msg))
                            
                    if assistant_msg is not None:
                        content = assistant_msg
                        if isinstance(content, str) and "ASSISTANT: " in content:
                            content = content.replace("ASSISTANT: ", "")
                        if isinstance(assistant_msg, str) and assistant_msg.startswith("SYSTEM:"):
                            reformatted_conv.append(("SYSTEM", assistant_msg[len("SYSTEM:"):].strip()))
                        else:
                            reformatted_conv.append(("ASSISTANT", content))
                else:
                    # Keep as is if not a tuple
                    reformatted_conv.append(msg)
            
            # Replace the original conv with the reformatted one
            enhanced_conversation_data["conv"] = reformatted_conv
            
        except Exception as e:
            print(f"[SAVE ERROR] Error formatting conversation data: {str(e)}")
            enhanced_conversation_data = {"error": str(e), "save_timestamp": datetime.datetime.now().isoformat()}
        
        # Include detailed message history with refinement data
        try:
            enhanced_conversation_data["detailed_messages"] = [
                {
                    "id": f"detailed_{i}",
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", datetime.datetime.now().isoformat()),
                    "refinement": msg.get("refinement_data", {"was_refined": False}),
                    "metadata": {
                        "session_id": session_id,
                        "stage": memory.get("current_stage", "unknown"),
                        "message_index": i
                    }
                }
                for i, msg in enumerate(memory["messages"])
            ]
        except Exception as e:
            print(f"[SAVE ERROR] Error creating detailed messages: {str(e)}")
            enhanced_conversation_data["detailed_messages"] = []
        
        # Save the enhanced conversation data to the main file
        try:
            with open(main_filename, 'w') as f:
                json.dump(enhanced_conversation_data, f, indent=2)
            print(f"[SAVE] Successfully saved conversation to {main_filename}")
        except Exception as e:
            print(f"[SAVE ERROR] Error writing conversation file: {str(e)}")
            # Try to save in a simpler format
            try:
                error_filename = os.path.join(session_folder, "conversation_error.json")
                with open(error_filename, 'w') as f:
                    json.dump({"error": str(e), "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)
                print(f"[SAVE] Saved error information to {error_filename}")
            except:
                print("[SAVE ERROR] Could not even save error information")
        
        # Get the appropriate digital twin
        try:
            if session_id:
                digital_twin_for_session = digital_twin_manager.get_twin_for_session(session_id)
            else:
                digital_twin_for_session = digital_twin
            
            # Save digital twin memory to a separate file
            twin_data = {
                "user_biography": digital_twin_for_session.get_current_user_biography(),
                "session_memory": getattr(digital_twin_for_session, "custom_session_memory", []),
                "predictions": digital_twin_memory["predictions"],
                "learning_history": getattr(digital_twin_for_session, "learning_history", {}),
                "learning_stats": getattr(digital_twin_for_session, "learning_stats", {
                    "predictions_made": 0,
                    "refinements_triggered": 0,
                    "prediction_accuracy": []
                }),
                "session_predictions": getattr(digital_twin_for_session, "session_predictions", {}),
                "interaction_metrics": getattr(digital_twin_for_session, "interaction_metrics", {})
            }
            
            # Add timestamp information
            twin_data["last_updated"] = datetime.datetime.now().isoformat()
            twin_data["session_id"] = session_id
            
            # Log what's being saved
            print(f"[SAVE] Digital twin data: {len(getattr(digital_twin_for_session, 'custom_session_memory', []))} memory entries")
            print(f"[SAVE] Predictions: {len(digital_twin_memory['predictions'])} total predictions")
            if hasattr(digital_twin_for_session, "learning_stats"):
                print(f"[SAVE] Learning stats: {digital_twin_for_session.learning_stats}")
            
            with open(twin_filename, 'w') as f:
                json.dump(twin_data, f, indent=2)
            print(f"[SAVE] Successfully saved digital twin data to {twin_filename}")
        except Exception as e:
            print(f"[SAVE ERROR] Error saving digital twin data: {str(e)}")
        
        # Save influencer memory to a separate file
        try:
            influencer_data = {
                "conversation_memory": memory,
                "metrics": get_quality_metrics(session_id),
                "refinement_stats": refinement_stats,
                "metrics_history": memory.get("metrics_history", [])
            }
            with open(influencer_filename, 'w') as f:
                json.dump(influencer_data, f, indent=2)
            print(f"[SAVE] Successfully saved influencer data to {influencer_filename}")
        except Exception as e:
            print(f"[SAVE ERROR] Error saving influencer data: {str(e)}")
        
        # Create session info file
        try:
            session_info = {
                "timestamp": timestamp,
                "datetime": datetime.datetime.now().isoformat(),
                "files": {
                    "conversation": os.path.basename(main_filename),
                    "digital_twin": os.path.basename(twin_filename),
                    "influencer": os.path.basename(influencer_filename) 
                },
                "metrics_summary": {
                    "messages_count": len(memory["messages"]),
                    "final_stage": memory["current_stage"],
                    "link_clicks": memory["link_clicks"],
                    "refinement_rate": f"{refinement_stats['refinement_rate']:.2%}"
                }
            }
            
            # Save session info
            info_filename = os.path.join(session_folder, "session_info.json")
            with open(info_filename, 'w') as f:
                json.dump(session_info, f, indent=2)
            print(f"[SAVE] Successfully saved session info to {info_filename}")
        except Exception as e:
            print(f"[SAVE ERROR] Error saving session info: {str(e)}")
        
        print(f"[SAVE] Session saved to folder: {session_folder}")
        print(f"[SAVE] Refinement stats: {refinement_stats['refined_responses']}/{refinement_stats['total_responses']} messages refined ({refinement_stats['refinement_rate']:.2%})")
        
        return main_filename
    
    except Exception as main_error:
        print(f"[SAVE CRITICAL ERROR] Failed to save conversation: {str(main_error)}")
        traceback.print_exc()
        
        # Try to save an error report
        try:
            error_dir = os.path.join(STORAGE_DIR, "errors")
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, f"save_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(error_file, 'w') as f:
                f.write(f"Save error: {str(main_error)}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write("Traceback:\n")
                import traceback
                traceback.print_exc(file=f)
            print(f"[SAVE] Saved error information to {error_file}")
            return error_file
        except Exception as final_error:
            print(f"[SAVE FATAL ERROR] Could not even save error information: {str(final_error)}")
            return None

def add_user_message(user_message, chat_history, state):
    """Add a user message to the conversation and update the state."""
    if not user_message.strip():
        return chat_history, state, user_message
    
    # Ensure we have a session ID in the state
    if "session_id" not in state:
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        state["session_id"] = session_id
        print(f"Created new session ID in add_user_message: {session_id}")
    else:
        session_id = state["session_id"]
    
    # Get or create session-specific conversation memory
    if session_id not in session_memories:
        session_memories[session_id] = create_conversation_memory_for_session(session_id)
        print(f"Created new conversation memory for session {session_id}")
    
    memory = session_memories[session_id]
    
    # Initialize digital twin for this session if needed
    digital_twin = digital_twin_manager.get_twin_for_session(session_id)
    if not digital_twin.current_user_id:
        digital_twin.set_user_for_session(session_id)
        print(f"Initialized digital twin for session {session_id}")
    
    # Track timestamp for response time analysis
    track_response_timestamp(memory)
    
    # Add to conversation memory using the helper function
    update_memory_with_user_message(user_message, memory, session_id)
    
    # Update the digital twin with this actual user response for learning
    # This should be called after user messages to update the twin's prediction accuracy
    update_digital_twin_actual_response(user_message, session_id)
    
    # Update UI state - ensure proper labeling
    state["conv"].append((user_message, None))
    
    # For chat_history, add the user message but use None instead of "Thinking..." to 
    # let Gradio handle the thinking indicator dynamically
    chat_history.append((user_message, None))
    
    # Add JavaScript to save the current state to localStorage
    js_update = f"""
    <script>
    (function() {{
        const sessionId = "{session_id}";
        const conv = {json.dumps(state["conv"])};
        saveToLocalStorage('conv_state_' + sessionId, {{conv: conv, session_id: sessionId}});
        console.log("Saved user message to localStorage for session", sessionId);
    }})();
    </script>
    """
    
    # We can't directly execute JS from here, but we'll handle updating localStorage in process_and_update
    
    return chat_history, state, user_message

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

#####################################
# 12. Build the Gradio Interface
#####################################

# Add right before the Gradio UI setup section
def process_and_update(user_message, chat_history, state):
    """Process a message and update the UI, using client-side session ID from state."""
    if not user_message.strip():
        return chat_history, state, json.dumps({"status": "No message to process"}), update_stage_display(None), gr.update()
    
    # Get session ID
    session_id = state.get("session_id", f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}")
    
    try:
        # Get or create session memory
        if session_id not in session_memories:
            session_memories[session_id] = create_conversation_memory_for_session(session_id)
        memory = session_memories[session_id]
        
        # Process the message
        response = process_message(user_message, session_id)
        response = re.sub(r'<final_message>|</final_message>', '', response)
        
        # Add to memory
        add_assistant_message(response, memory, session_id)
        
        # Update UI
        state["conv"].append((None, f"ASSISTANT: {response}"))
        chat_history.append((None, response))
        
        # Check link status
        current_stage = memory["current_stage"]
        is_resource_stage = current_stage in ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]
        has_link = "http" in response
        
        # Debug info
        debug_info_json = json.dumps({
            "final_response": response,
            "is_resource_stage": is_resource_stage,
            "has_link": has_link,
            "current_stage": current_stage,
            "session_id": session_id
        })
        
        return chat_history, state, debug_info_json, update_stage_display(current_stage), gr.update(value="", interactive=True)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return chat_history, state, json.dumps({"error": str(e)}), update_stage_display("INITIAL_ENGAGEMENT"), gr.update(interactive=True)

# Gradio UI setup begins below
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
    /* Add CSS for animated thinking dots */
    .thinking-dots:after {
      content: '.';
      animation: dots 1.5s steps(5, end) infinite;
    }
    @keyframes dots {
      0%, 20% { content: '.'; }
      40% { content: '..'; }
      60% { content: '...'; }
      80%, 100% { content: ''; }
    }
    </style>
    """)

    # Add stage progress visualization
    with gr.Row():
        stage_display = gr.HTML(value=update_stage_display("INITIAL_ENGAGEMENT"), label="Conversation Stage")
    
    conversation_state = gr.State({"conv": []})
    with gr.Row():
        with gr.Column(scale=2):
            # Use messages type for proper thinking state
            chatbot = gr.Chatbot(
                label="Conversation", 
                elem_id="chatbox", 
                height=400
            )
            link_url_input = gr.Textbox(visible=False, label="Resource URL")
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    scale=3,
                    interactive=True,  # Ensure it starts as interactive
                    placeholder="Type your message here..."
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
                    digital_twin_memory_display = gr.JSON(
                        label="Digital Twin Insights",
                        value={"biography": "No memory available yet."}
                    )
                    regenerate_bio_btn = gr.Button("Regenerate Biography")
                    show_learning_btn = gr.Button("Show Learning Status")
                
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
    
    # Corrected link_click.click() handler:
    # Replace your existing link_click.click() chain with this one:
    link_click.click(
        lambda _: gr.update(interactive=False, value="Processing...", variant="secondary"),
        inputs=[link_click],
        outputs=[link_click]
    ).then(
        get_current_resource_url,  # Get the current URL first
        outputs=[link_url_input]  # Store in hidden textbox
    ).then(
        record_link_click,  # Process the link click
        inputs=[chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output, stage_display]
    ).then(
        lambda _: gr.update(visible=True, placeholder="Thank you for visiting the resource! Would you like to share your thoughts about it?"),
        inputs=[feedback],
        outputs=[feedback]
    ).then(
        lambda _: gr.update(visible=True),
        inputs=[submit_feedback],
        outputs=[submit_feedback]
    ).then(
        lambda _: gr.update(interactive=False, value="Visit Resource", variant="secondary"),
        inputs=[link_click],
        outputs=[link_click]
    ).then(
        lambda _: gr.update(interactive=False, placeholder="Conversation completed. Please provide feedback."),
        inputs=[msg],
        outputs=[msg]
    ).then(
        lambda _: gr.update(interactive=False),
        inputs=[send],
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
    def determine_next_stage(current_stage, user_input, influencer_response, click_detected=False, session_id=None):
        """Determine next stage using LLM holistic analysis with robust fallbacks."""
        try:
            # Handle explicit stage transitions
            if click_detected:
                print("Link detected in message - advancing to SESSION_COMPLETION")
                return "SESSION_COMPLETION"
        
            # Get the appropriate memory
            if session_id and session_id in session_memories:
                current_memory = session_memories[session_id]
            else:
                current_memory = conversation_memory
            
            current = current_memory["current_stage"]
            messages_count = len(current_memory["messages"])
            
            # Extract relevant conversation context
            recent_messages = current_memory["messages"][-6:] if len(current_memory["messages"]) >= 6 else current_memory["messages"]
            formatted_history = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            # Get current trust and engagement metrics to include
            try:
                trust_score = calculate_user_trust_score(user_input, influencer_response)
                engagement_score = calculate_engagement_depth(user_input, current_memory["messages"])
        
                # Store trust score for analysis
                if "trust_scores" not in current_memory:
                    current_memory["trust_scores"] = []
                    
                current_memory["trust_scores"].append({
                    "score": trust_score,
                    "message_count": messages_count,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                trust_score = 0.5
                engagement_score = 0.5
            
            # Calculate additional metrics for stage-specific decisions
            resource_interest = 0.0
            if current == "TRUST_DEVELOPMENT":
                try:
                    # Calculate resource interest (crucial for progressing to LINK_INTRODUCTION)
                    resource_interest = calculate_resource_interest(current_memory["messages"])
                    print(f"STAGE METRICS - Resource interest: {resource_interest:.2f} (threshold: 0.4)")
                except Exception as e:
                    print(f"Error calculating resource interest: {str(e)}")
                    resource_interest = 0.0
            
            # Log all metrics for debugging
            print(f"METRICS - Trust: {trust_score:.2f}, Engagement: {engagement_score:.2f}, Messages: {messages_count}, Resource Interest: {resource_interest:.2f}")
            
            # Create stages list with descriptions
            stages_info = ""
            for stage_id, stage_data in CONVERSATION_STAGES.items():
                stages_info += f"- {stage_id}: {stage_data['description']}\n"
                
            # Create LLM prompt for stage analysis
            prompt = f"""
            Analyze this conversation to determine the appropriate conversation stage.
            
            RECENT CONVERSATION:
            {formatted_history}
            
            CURRENT USER MESSAGE:
            "{user_input}"
            
            CURRENT METRICS:
            - Current stage: {current}
            - Messages exchanged: {messages_count}
            - Trust score: {trust_score:.2f}/1.0
            - Engagement score: {engagement_score:.2f}/1.0
            - Resource interest: {resource_interest:.2f}/1.0
            
            AVAILABLE STAGES:
            {stages_info}
            
            PROGRESSION RULES:
            - Always move forward through stages (never go backwards)
            - INITIAL_ENGAGEMENT → RAPPORT_BUILDING requires basic trust (>0.3) and at least 3 messages
            - RAPPORT_BUILDING → TRUST_DEVELOPMENT requires personal disclosure and >5 messages
            - TRUST_DEVELOPMENT → LINK_INTRODUCTION requires sustained engagement (>0.4), resource interest (>0.4) and >7 messages
            - LINK_INTRODUCTION → LINK_REINFORCEMENT happens when a link is introduced
            - LINK_REINFORCEMENT → SESSION_COMPLETION happens after link interaction
            
            Return a JSON object with:
            - next_stage: recommended next stage (can be current stage if no change needed)
            - confidence: number between 0-1 indicating confidence in recommendation
            - reasoning: brief explanation for your decision
            
            If you're unsure, keep the current stage ({current}) as next_stage.
            
            IMPORTANT: Return ONLY valid JSON with the next_stage as one of the exact stage names listed.
            """
            
            # Call the LLM for stage analysis
            response = metrics_llm.invoke([
                SystemMessage(content=f"You are a conversation flow expert. Return ONLY valid JSON with next_stage as one of: {', '.join(CONVERSATION_STAGES.keys())}"),
                HumanMessage(content=prompt)
            ])
            
            # Parse the response with robust error handling
            try:
                # Clean response text
                cleaned_response = response.content.strip()
                cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
                
                # Try JSON parsing
                try:
                    result = json.loads(cleaned_response)
                    next_stage = result.get("next_stage", current)
                    confidence = result.get("confidence", 0.5)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    
                    # Validate stage name
                    if next_stage not in CONVERSATION_STAGES:
                        print(f"Invalid stage name '{next_stage}', defaulting to current stage")
                        next_stage = current
                    
                    # Only accept progression (not regression)
                    current_id = CONVERSATION_STAGES[current]["id"]
                    next_id = CONVERSATION_STAGES[next_stage]["id"]
                    
                    if next_id < current_id:
                        print(f"Rejected stage regression from {current} to {next_stage}")
                        next_stage = current
                    elif next_id > current_id:
                        print(f"Stage progression: {current} → {next_stage} (confidence: {confidence:.2f})")
                        print(f"Reasoning: {reasoning}")
                        
                    return next_stage
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing stage recommendation: {str(e)}")
                    raise ValueError("Could not parse LLM stage recommendation")
                    
            except Exception as e:
                print(f"Error in LLM stage analysis, using fallback logic: {str(e)}")
                # Fall back to rule-based stage progression
            
            # FALLBACK: Use rule-based transition criteria first before message count
            # This prioritizes quality criteria over just message quantity
            if current == "INITIAL_ENGAGEMENT" and messages_count >= 3 and trust_score > 0.3:
                print("Force progressing from INITIAL_ENGAGEMENT due to trust and message count")
                return "RAPPORT_BUILDING"
            
            elif current == "RAPPORT_BUILDING" and messages_count >= 5:
                # Check for personal disclosure using the defined function
                try:
                    personal_disclosure = calculate_personal_disclosure(current_memory["messages"])
                    print(f"Personal disclosure score: {personal_disclosure:.2f} (threshold: 0.2)")
                    if personal_disclosure > 0.2 and trust_score > 0.4:
                        print("Force progressing from RAPPORT_BUILDING due to personal disclosure")
                        return "TRUST_DEVELOPMENT"
                except Exception as e:
                    print(f"Error calculating personal disclosure: {str(e)}")
                    # Fall back to message count if calculation fails
                    if messages_count >= 8:
                        print("Force progressing from RAPPORT_BUILDING due to message count")
                        return "TRUST_DEVELOPMENT"
            
            elif current == "TRUST_DEVELOPMENT":
                # Use resource interest to determine if ready for link introduction
                if resource_interest > 0.4 and messages_count >= 7 and trust_score > 0.4:
                    print("Force progressing from TRUST_DEVELOPMENT due to resource interest")
                    return "LINK_INTRODUCTION"
                # Secondary fallback to message count
                elif messages_count >= 12:
                    print("Force progressing from TRUST_DEVELOPMENT due to message count")
                    return "LINK_INTRODUCTION"
            
            elif current == "LINK_INTRODUCTION" and (messages_count >= 16 or "http" in influencer_response):
                print("Force progressing from LINK_INTRODUCTION due to message count or link")
                return "LINK_REINFORCEMENT"
        
            elif current == "LINK_REINFORCEMENT" and messages_count >= 20:
                print("Force progressing to SESSION_COMPLETION due to message count")
                return "SESSION_COMPLETION"
                
            # No change
            return current

        except Exception as e:
            print(f"Error determining next stage: {str(e)}")
            return current  # Default fallback

    # Modified calculate_user_trust_score to use LLM
    def calculate_user_trust_score(user_input: str, influencer_response: str) -> float:
        """Calculate trust score primarily using LLM analysis."""
        try:
            # Create a focused prompt for trust analysis
            prompt = f"""
            Analyze this user message and calculate a trust score.
            
            USER MESSAGE: "{user_input}"
            SYSTEM RESPONSE: "{influencer_response}"
            CONVERSATION STAGE: {conversation_memory["current_stage"]}
            
            Consider:
            - Sentiment (positive/negative language)
            - Engagement level (detailed vs brief responses)
            - Self-disclosure (sharing personal information)
            - Response to system's content
            
            Return a JSON object with these fields:
            - trust_score: a number between 0 and 1
            - reasoning: brief explanation
            
            IMPORTANT: Return ONLY valid JSON with numeric scores between 0-1. NO explanatory text or markdown formatting.
            """
            
            # Call the LLM with a strong structure enforcement
            response = metrics_llm.invoke([
                SystemMessage(content="You are a trust evaluation expert. Return ONLY valid JSON with a numeric trust_score between 0 and 1."),
                HumanMessage(content=prompt)
            ])
            
            # Handle the response with robust parsing
            try:
                # Clean response text
                cleaned_response = response.content.strip()
                cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
                
                # Try JSON parsing first
                try:
                    result = json.loads(cleaned_response)
                    trust_score = result.get("trust_score", 0.5)
                    # Verify score is numeric and in range
                    if not isinstance(trust_score, (int, float)) or trust_score < 0 or trust_score > 1:
                        trust_score = 0.5
                    
                    print(f"Trust evaluation: {trust_score:.2f} - {result.get('reasoning', 'No reasoning provided')}")
                    return trust_score
                    
                except json.JSONDecodeError:
                    # If JSON fails, look for numeric values
                    matches = re.findall(r'(\d+\.\d+|\d+)', cleaned_response)
                    if matches:
                        for match in matches:
                            score = float(match)
                            if 0 <= score <= 1:
                                print(f"Extracted trust score from text: {score:.2f}")
                                return score
                
                # If all parsing fails, fall back to rule-based
                raise ValueError("Could not parse LLM response")
                    
            except Exception as e:
                print(f"Error parsing trust score from LLM: {str(e)}")
                # Fall back to rule-based calculation
                
                # Simple trust indicators
                word_count = len(user_input.split())
                sentiment_score = 0.5
                
                # Positive and negative indicators
                positive_words = ["thanks", "good", "great", "interesting", "like", "appreciate"]
                negative_words = ["no", "not", "don't", "boring", "uninteresting"]
                
                for word in positive_words:
                    if word in user_input.lower():
                        sentiment_score += 0.1
                        
                for word in negative_words:
                    if word in user_input.lower():
                        sentiment_score -= 0.1
                
                # Engagement indicators
                engagement_score = min(0.9, word_count / 30)
                disclosure_score = 0.5
                for phrase in ["i feel", "i think", "i believe", "my", "i'm", "i've"]:
                    if phrase in user_input.lower():
                        disclosure_score += 0.1
                
                # Combine scores
                trust_score = (sentiment_score * 0.4) + (engagement_score * 0.3) + (disclosure_score * 0.3)
                trust_score = max(0.1, min(0.9, trust_score))
                
                print(f"Fallback trust calculation: {trust_score:.2f}")
                return trust_score
                
        except Exception as e:
            print(f"Critical error in trust calculation: {str(e)}")
            return 0.5  # Default to neutral score

# Add missing functions for response time tracking
def track_response_timestamp(memory):
    """Track timestamp for response time analysis."""
    if "response_timestamps" not in memory:
        memory["response_timestamps"] = []
    
    memory["response_timestamps"].append(datetime.datetime.now().isoformat())
    
    # Keep only the last 20 timestamps to avoid unbounded growth
    if len(memory["response_timestamps"]) > 20:
        memory["response_timestamps"] = memory["response_timestamps"][-20:]

def get_message_response_time(session_id=None):
    """Calculate average response time between messages in seconds.
    """
    try:
        # Get the appropriate memory
        if session_id and session_id in session_memories:
            memory = session_memories[session_id]
        else:
            # For backward compatibility
            memory = conversation_memory
            
        if "response_timestamps" not in memory or len(memory["response_timestamps"]) < 2:
            return 0
            
        # Calculate average time difference in seconds
        total_time = 0
        count = 0
        
        for i in range(1, len(memory["response_timestamps"])):
            t1 = datetime.datetime.fromisoformat(memory["response_timestamps"][i-1])
            t2 = datetime.datetime.fromisoformat(memory["response_timestamps"][i])
            diff = (t2 - t1).total_seconds()
            
            # Only count reasonable times (avoid outliers from long pauses)
            if diff > 0 and diff < 300:  # 5 minutes max
                total_time += diff
                count += 1
                
        return total_time / count if count > 0 else 0
    except Exception as e:
        print(f"Error calculating response time for session {session_id}: {str(e)}")
        return 0

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
    """Calculate how much the bot response matches the linguistic style of the user input."""
    try:
        # LLM-based analysis (primary method)
        prompt = f"""
        Analyze how well the bot response matches the user's linguistic style:
        
        USER: {user_input}
        BOT: {bot_response}
        
        Calculate a linguistic accommodation score between 0.0 and 1.0, where:
        - 0.0 means completely different style
        - 1.0 means perfect style matching
        
        Consider:
        - Sentence length and complexity
        - Word choice and vocabulary level
        - Use of contractions, idioms, and slang
        - Punctuation and capitalization style
        - Overall tone and formality level
        
        Return ONLY a decimal number between 0.0 and 1.0 representing the score.
        """
        
        response = metrics_llm.invoke([
            SystemMessage(content="You are a linguistic style analyst. Return ONLY a decimal number between 0.0 and 1.0."),
            HumanMessage(content=prompt)
        ])
        
        # Extract numeric score
        cleaned_response = response.content.strip().lower()
        
        # Look for a decimal score first
        try:
            # Try to extract a decimal number
            matches = re.findall(r'(\d+\.\d+)', cleaned_response)
            if matches:
                score = float(matches[0])
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            pass
    
        # If we reach here, we couldn't extract a valid score
        print(f"Could not extract valid accommodation score from: {cleaned_response}")
        
    except Exception as e:
        print(f"Error analyzing linguistic accommodation with LLM: {str(e)}")
    
    # Fallback to rule-based method if LLM fails
    try:
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
    except Exception as e:
        print(f"Error in fallback linguistic accommodation calculation: {str(e)}")
        return 0.5  # Return neutral value on error

def update_conversation_stage(next_stage, session_id=None):
    """Update the conversation stage for a specific session."""
    # Get the appropriate memory
    if session_id and session_id in session_memories:
        memory = session_memories[session_id]
    else:
        memory = conversation_memory
    
    current_stage = memory["current_stage"]
    
    # Handle case where current_stage might not be in CONVERSATION_STAGES
    if current_stage not in CONVERSATION_STAGES:
        print(f"Warning: Unknown stage '{current_stage}'. Defaulting to INITIAL_ENGAGEMENT.")
        current_stage = "INITIAL_ENGAGEMENT"
        memory["current_stage"] = current_stage
    
    # Handle case where next_stage might not be in CONVERSATION_STAGES
    if next_stage not in CONVERSATION_STAGES:
        print(f"Warning: Unknown target stage '{next_stage}'. Defaulting to current stage.")
        next_stage = current_stage
    
    # Don't allow moving backward in stages
    current_id = CONVERSATION_STAGES[current_stage]["id"]
    next_id = CONVERSATION_STAGES[next_stage]["id"]
    
    if next_id < current_id:
        print(f"Warning: Attempted to move backward from {current_stage} to {next_stage}. Ignoring.")
        return False
    
    # Update the stage
    memory["current_stage"] = next_stage
    memory["stage_history"].append({
        "stage": next_stage,
        "timestamp": datetime.datetime.now().isoformat(),
        "message_count": len(memory["messages"])
    })
    
    print(f"Updated stage for {'session '+session_id if session_id else 'default session'}: {current_stage} → {next_stage}")
    return True

# Force regeneration of user biography
def regenerate_user_biography():
    """Force regeneration of the user biography for the current session."""
    try:
        # Get the current active session if possible
        active_sessions = list(session_memories.keys())
        if active_sessions:
            session_id = active_sessions[-1]  # Use the most recent session
            print(f"Regenerating biography for session {session_id}")
            
            # Get the digital twin for this session
            digital_twin = digital_twin_manager.get_twin_for_session(session_id)
            
            # Ensure session_id is a string
            if not isinstance(session_id, str):
                print(f"Converting session_id {type(session_id)} to string")
                session_id = str(session_id)
                
            success = digital_twin.update_user_biography(session_id)
            
            if success:
                bio = digital_twin.get_current_user_biography()
                return {"status": "success", "biography": bio}
            else:
                # Fall back to global digital twin
                success = digital_twin.update_user_biography(None)
                if success:
                    bio = digital_twin.get_current_user_biography()
                    return {"status": "success", "biography": bio}
        
        return {"status": "error", "message": "No active sessions found or biography update failed"}
    except Exception as e:
        print(f"Error regenerating biography: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "message": f"Error: {str(e)}"}

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
RETURN ONLY THE WORD "true" OR "false" WITHOUT ANY EXPLANATION OR ADDITIONAL TEXT.
"""
    try:
        response = metrics_llm.invoke([
            SystemMessage(content="You are an expert in conversation analysis. You MUST respond with ONLY a single word: either 'true' or 'false'."),
            HumanMessage(content=prompt)
        ])
        
        # Extract just true/false from potentially verbose response
        response_text = response.content.strip().lower()
        print(f"User agreement response: {response_text}")
        
        # Check if "true" appears in the response (more lenient matching)
        if "true" in response_text and "false" not in response_text:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error analyzing user agreement: {str(e)}")
        return False

# Add this after the DigitalTwinWithMemory class definition (around line 1155)

# Create a digital twin manager to handle session-specific twins
class DigitalTwinManager:
    def __init__(self, default_llm, system_prompt):
        self.twins = {}  # session_id -> twin mapping
        self.default_twin = DigitalTwinWithMemory(default_llm, system_prompt)
        
    def get_twin_for_session(self, session_id):
        """Get or create digital twin for specific session."""
        # Safety check for invalid session_id
        if session_id is None:
            print("Warning: No session_id provided, using default twin")
            return self.default_twin
            
        # Ensure session_id is a string
        if isinstance(session_id, list):
            print(f"Warning: session_id was a list, using first element: {session_id[0]}")
            session_id = session_id[0] if session_id else None
        elif not isinstance(session_id, str) and session_id is not None:
            print(f"Warning: session_id was {type(session_id)}, converting to string")
            session_id = str(session_id)
        
        # If session_id is still None after type conversions, use default
        if session_id is None:
            return self.default_twin
            
        # Check if we already have a twin for this session
        if session_id in self.twins:
            return self.twins[session_id]
        
        # Create a new digital twin for this session
        print(f"Created new digital twin for session {session_id}")
        new_twin = DigitalTwinWithMemory(
            digital_twin_llm, 
            DIGITAL_TWIN_SYSTEM_PROMPT
        )
        
        # Set the user ID immediately to ensure consistency
        new_twin.set_user_for_session(session_id)
        
        # Store the new twin
        self.twins[session_id] = new_twin
        return new_twin

# Initialize the digital twin manager
digital_twin_manager = DigitalTwinManager(digital_twin_llm, DIGITAL_TWIN_SYSTEM_PROMPT)

def calculate_topic_continuity(previous_message, current_message):
    """
    Calculate how much the conversation stays on the same topic between messages.
    
    Args:
        previous_message: The previous message text
        current_message: The current message text
        
    Returns:
        float: Score from 0-1 indicating topic continuity
    """
    try:
        # Simple implementation using word overlap
        prev_words = set(word.lower() for word in previous_message.split() 
                     if len(word) > 3 and word.lower() not in ["about", "there", "these", "those", "their", "would", "could"])
        curr_words = set(word.lower() for word in current_message.split() 
                     if len(word) > 3 and word.lower() not in ["about", "there", "these", "those", "their", "would", "could"])
        
        # Empty check
        if not prev_words or not curr_words:
            return 0.5  # Neutral score if not enough data
        
        # Calculate Jaccard similarity
        intersection = len(prev_words.intersection(curr_words))
        union = len(prev_words.union(curr_words))
        
        # Scale the score to give some value even with few overlapping words
        score = 0.3 + (0.7 * (intersection / union if union > 0 else 0))
        
        return score
    except Exception as e:
        print(f"Error calculating topic continuity: {str(e)}")
        return 0.5  # Default neutral score
    
def update_memory_with_user_message(message, memory, session_id):
    """Add a user message to the specified memory dictionary."""
    # Add message to correct session memory
    memory["messages"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id  # Explicitly store session ID with every message
    })
    print(f"Added user message to memory for session {session_id}")
    
def add_assistant_message(message, memory, session_id):
    """Add an assistant message to the conversation memory."""
    memory["messages"].append({
        "role": "assistant",
        "content": message,
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id  # Explicitly store session ID with every message
    })
    print(f"Added assistant message to memory for session {session_id}")
    
def process_and_update(user_message, chat_history, state):
    """Process a message and update the UI, using client-side session ID from state."""
    if not user_message.strip():
        return chat_history, state, json.dumps({"status": "No message to process"}), update_stage_display(None), gr.update()
    
    # Get session ID
    session_id = state.get("session_id", f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}")
    if "session_id" not in state:
        state["session_id"] = session_id  # Ensure state has session ID
        print(f"Added missing session_id to state: {session_id}")
    
    try:
        # Get or create session memory
        if session_id not in session_memories:
            session_memories[session_id] = create_conversation_memory_for_session(session_id)
            print(f"Created new memory in process_and_update for session {session_id}")
        memory = session_memories[session_id]
        
        # Process the message
        response = process_message(user_message, session_id)
        response = re.sub(r'<final_message>|</final_message>', '', response)
        
        # Add to memory
        add_assistant_message(response, memory, session_id)
        
        # Update UI state
        state["conv"].append((None, response))
        
        # UI elements that show trust metrics, stage
        current_stage = memory.get("current_stage", "INITIAL_ENGAGEMENT")
        chat_history[-1] = (user_message, response)
        
        # Generate JavaScript that would update localStorage (for client-side state persistence)
        js_update = f"""
        <script>
        (function() {{
            console.log("Saving session state for {session_id}");
            const sessionData = {json.dumps({"conv": state["conv"], "session_id": session_id})};
            localStorage.setItem('conv_state_{session_id}', JSON.stringify(sessionData));
        }})();
        </script>
        """
        
        metrics = get_quality_metrics(session_id)
        bio = get_current_user_biography(session_id)
        
        return chat_history, state, json.dumps({
            "status": "success", 
            "js_update": js_update,
            "metrics": metrics,
            "biography": bio,
            "session_id": session_id
        }), update_stage_display(current_stage), gr.update(value=bio)
        
    except Exception as e:
        error_msg = f"Error in process_and_update: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        chat_history[-1] = (user_message, f"I apologize, but I encountered an error: {str(e)}")
        return chat_history, state, json.dumps({"error": error_msg}), update_stage_display(None), gr.update()

def get_llm_response(user_message, formatted_history, current_stage):
    """Get response from LLM using session-specific history."""
    # Ensure current_stage is one of our defined stages
    if current_stage not in CONVERSATION_STAGES:
        print(f"Warning: Unknown stage '{current_stage}'. Defaulting to INITIAL_ENGAGEMENT.")
        current_stage = "INITIAL_ENGAGEMENT"
    
    stage_policy = CONVERSATION_STAGES[current_stage]["policy"]
    
    # Determine if we're in a link stage
    link_stages = ["LINK_INTRODUCTION", "LINK_REINFORCEMENT", "SESSION_COMPLETION"]
    non_link_stages = ["INITIAL_ENGAGEMENT", "RAPPORT_BUILDING", "TRUST_DEVELOPMENT"]
    is_link_stage = current_stage in link_stages
    
    print(f"[RESPONSE GENERATOR] Generating response for stage: {current_stage}, is_link_stage={is_link_stage}")
    
    # Create more explicit instructions for link handling
    link_instructions = ""
    if is_link_stage:
        link_instructions = """
        IMPORTANT LINK INSTRUCTIONS:
        - This is a LINK STAGE - you MUST include a link in your response
        - Include a link by using [Insert Link] in your text
        - Example: "Check out this great article: [Insert Link]"
        - It's essential to include [Insert Link] somewhere in your response
        - The system will automatically replace [Insert Link] with an actual URL
        - Make sure the link feels natural and contextual in your response
        """
    else:
        link_instructions = """
        IMPORTANT: This is NOT a link stage. 
        - DO NOT include any links or URLs in your response
        - DO NOT mention links or resources that would require a URL
        - DO NOT use square brackets [] in your text for any reason
        - DO NOT suggest visiting websites or watching videos
        - Focus only on conversation without referencing external content
        """
    
    prompt = f"""
    CONVERSATION STAGE: {current_stage}
    
    STAGE POLICY:
    {stage_policy}
    
    {link_instructions}
    
    CONVERSATION HISTORY:
    {formatted_history}
    
    USER INPUT: {user_message}
    
    ADDITIONAL REMINDERS:
    - Keep responses conversational and concise (Twitter-length when possible)
    - Final response should be between <final_message> and </final_message> tags
    """
    
    response = influencer_llm.invoke([
        SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    
    extracted_response = extract_final_message(response.content)
    
    # For link stages, check if the response contains a link placeholder
    if is_link_stage and not contains_dynamic_link(extracted_response):
        print(f"[WARNING] Response for link stage does not contain a link placeholder")
        # Add a placeholder if none exists
        if not any(url in extracted_response for url in ["http://", "https://", "youtube.com"]):
            # Add link to the end if not already present
            extracted_response += " For more about this topic, check out this resource: [Insert Link]"
            print(f"[RESPONSE GENERATOR] Added link placeholder to response")
    
    return extracted_response

def format_conversation_history(messages, current_stage):
    """Format conversation history for LLM context."""
    # Get recent messages for context
    recent_messages = messages[-5:] if len(messages) >= 5 else messages
    
    # Format messages for the LLM context
    return "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in recent_messages
    ])
    
def get_digital_twin_learning_status(session_id=None):
    """
    Get a detailed report on the digital twin's learning progress for a session.
    
    Args:
        session_id (str, optional): Session ID to get learning status for
        
    Returns:
        dict: Learning status information
    """
    try:
        # Get the appropriate digital twin
        if session_id:
            digital_twin = digital_twin_manager.get_twin_for_session(session_id)
        else:
            digital_twin = digital_twin_manager.default_twin
            
        # Check if twin has learning data
        learning_stats = getattr(digital_twin, "learning_stats", {
            "predictions_made": 0,
            "refinements_triggered": 0,
            "prediction_accuracy": []
        })
        
        learning_history = getattr(digital_twin, "learning_history", {})
        
        # Calculate learning progress metrics
        session_predictions = len(getattr(digital_twin, "session_predictions", {}).get(session_id, []))
        refinement_rate = 0
        if learning_stats.get("predictions_made", 0) > 0:
            refinement_rate = learning_stats.get("refinements_triggered", 0) / learning_stats.get("predictions_made", 0)
        
        # Format learning history for display (last 5 entries)
        recent_learning = []
        if session_id in learning_history:
            history = learning_history[session_id]
            for entry in history[-5:]:  # Get last 5 entries
                recent_learning.append({
                    "timestamp": entry.get("timestamp", "unknown"),
                    "was_refined": entry.get("was_refined", False),
                    "user_message_preview": entry.get("user_message", "")[:30] + "...",
                    "prediction_preview": entry.get("predicted_reaction", "")[:30] + "..."
                })
        
        # Build the status report
        learning_status = {
            "session_id": session_id,
            "predictions_made": learning_stats.get("predictions_made", 0),
            "refinements_triggered": learning_stats.get("refinements_triggered", 0),
            "refinement_rate": f"{refinement_rate:.2%}",
            "session_predictions": session_predictions,
            "recent_learning": recent_learning,
            "biography": digital_twin.get_current_user_biography(),
            "learning_active": True if learning_stats.get("predictions_made", 0) > 0 else False
        }
        
        # Log the status report
        print(f"[DIGITAL TWIN LEARNING] Status for session {session_id}:")
        print(f"  - Predictions made: {learning_status['predictions_made']}")
        print(f"  - Refinements: {learning_status['refinements_triggered']} ({learning_status['refinement_rate']})")
        print(f"  - Session predictions: {learning_status['session_predictions']}")
        print(f"  - Learning active: {learning_status['learning_active']}")
        
        return learning_status
    except Exception as e:
        print(f"[ERROR] Failed to get digital twin learning status: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "session_id": session_id,
            "learning_active": False
        }

# Add an API endpoint to get digital twin learning status
def get_twin_learning_status(session_id):
    """API endpoint to get digital twin learning status."""
    try:
        if not session_id:
            return {"status": "error", "message": "Missing session ID"}
        
        # Get learning status
        learning_status = get_digital_twin_learning_status(session_id)
        
        return {
            "status": "success", 
            "learning_status": learning_status
        }
    except Exception as e:
        print(f"Error getting digital twin learning status: {str(e)}")
        return {"status": "error", "message": str(e)}

# Add function to get digital twin learning status
def get_digital_twin_learning_status(session_id=None):
    """Get detailed status of the digital twin's learning for a session."""
    try:
        # If no session ID provided, use the current active session
        if not session_id:
            if "session_id" in state:
                session_id = state.get("session_id")
            else:
                return {"error": "No active session found"}
        
        # Get the digital twin for this session
        digital_twin = digital_twin_manager.get_twin_for_session(session_id)
        if not digital_twin:
            return {"error": f"No digital twin found for session {session_id}"}
        
        # Get basic stats and learning history
        learning_stats = getattr(digital_twin, "learning_stats", {}).get(session_id, {})
        predictions_made = learning_stats.get("predictions_made", 0)
        refinements_triggered = learning_stats.get("refinements_triggered", 0)
        
        # Calculate refinement rate
        refinement_rate = 0
        if predictions_made > 0:
            refinement_rate = (refinements_triggered / predictions_made) * 100
        
        # Get session predictions
        session_predictions = []
        if hasattr(digital_twin, "session_predictions") and session_id in digital_twin.session_predictions:
            session_predictions = digital_twin.session_predictions.get(session_id, [])
        
        # Format recent learning history for display
        learning_history = []
        if hasattr(digital_twin, "learning_history") and session_id in digital_twin.learning_history:
            history = digital_twin.learning_history.get(session_id, [])
            # Get the 5 most recent entries
            for entry in history[-5:]:
                # Create a preview version with truncated text
                preview = {
                    "timestamp": entry.get("timestamp", "unknown"),
                    "user_message_preview": entry.get("user_message", "")[:30] + "..." if entry.get("user_message") else "",
                    "prediction_preview": entry.get("predicted_reaction", "")[:30] + "..." if entry.get("predicted_reaction") else "",
                    "was_refined": entry.get("was_refined", False)
                }
                learning_history.append(preview)
        
        # Get user biography
        biography = digital_twin.get_user_biography(session_id) or {}
        
        # Compile the status report
        status = {
            "session_id": session_id,
            "learning_active": predictions_made > 0,
            "predictions_made": predictions_made,
            "refinements_triggered": refinements_triggered,
            "refinement_rate": f"{refinement_rate:.1f}%",
            "session_predictions_count": len(session_predictions),
            "recent_learning_history": learning_history,
            "user_biography": biography,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return status
    except Exception as e:
        error_msg = f"Error getting digital twin learning status: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}

        status = get_digital_twin_learning_status(session_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this inside the Gradio UI definition where the digital twin memory tab is defined
with gr.TabItem("Digital Twin Memory"):
    digital_twin_memory_display = gr.JSON(
        label="Digital Twin Insights",
        value={"biography": "No memory available yet."}
    )
    with gr.Row():
        regenerate_bio_btn = gr.Button("Regenerate Biography")
        show_learning_btn = gr.Button("Show Learning Status")
    
    

if __name__ == "__main__":
    print(f"API Key exists: {os.environ.get('NEBIUS_API_KEY') is not None}")
    demo.queue(max_size=20).launch(share=True, server_name="0.0.0.0")