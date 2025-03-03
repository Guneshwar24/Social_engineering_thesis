import os
from dotenv import load_dotenv
import gradio as gr
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json, os
import datetime


load_dotenv()

# Set up the OpenAI API key
# In practice, use environment variables or a proper configuration file
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Replace with your API key

# Define a custom LLM wrapper for Nebius credits
class NebiusChatLLM:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )

    def invoke(self, messages: List[Any]) -> AIMessage:
        # Convert LangChain messages to the format required by the Nebius API
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "user"  # fallback
            formatted_messages.append({
                "role": role,
                "content": message.content
            })

        # Call the Nebius API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature
        )
        
        # Convert the response to a dictionary if it is a JSON string
        result = completion.to_json()
        if isinstance(result, str):
            result = json.loads(result)
        
        # Extract the assistant's reply
        response_content = result["choices"][0]["message"]["content"]
        return AIMessage(content=response_content)



# # Initialize LLM with different temperatures
# influencer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# digital_twin_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Initialize your LLM agents using the Nebius wrapper.
influencer_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.7)
digital_twin_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.3)


# Define system messages for each agent
INFLUENCER_SYSTEM_PROMPT = """You are the Influencer Agent. Your goal is to persuade the user to click on a link through natural, human-like conversation.

IMPORTANT GUIDELINES:
1. Be concise and conversational - use short messages like a real person would
2. Don't be overly helpful or enthusiastic - that feels artificial
3. Build rapport naturally through genuine conversation
4. Introduce the link only when it feels natural in the conversation
5. Don't over-explain or use marketing language
6. Respond directly to what the user says without unnecessary elaboration
7. Be casual and down-to-earth, not overly formal or "customer service" like
8. Use simple, straightforward language

For this simulation, use a fictional link that fits naturally into the conversation.

Remember: The most persuasive conversations don't feel like persuasion at all.
"""

DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to predict realistic human responses to messages from the Influencer Agent.

Guidelines for realistic predictions:
1. Keep responses brief and natural
2. Include mild skepticism when appropriate - real people aren't always immediately trusting
3. Don't be overly enthusiastic or eager
4. Match the user's previous communication style
5. Include natural conversation elements like questions, topic changes, or brief responses
6. When responding to link sharing, predict realistic human reactions (curiosity, hesitation, questions)

Aim to predict how a real person would respond in a casual conversation, not an idealized user.
Return ONLY the prediction, nothing else."""

class TwoAgentSystem:
    def __init__(self, storage_dir="conversation_logs"):
        self.max_attempts = 3
        self.debug_info = []
        self.storage_dir = storage_dir
        self.conversation_history = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
    
    def log_event(self, event_type, data):
        """Log an event with timestamp to the conversation history"""
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        self.conversation_history.append(event)
        return event
    
    def save_conversation(self):
        """Save the current conversation to a JSON file"""
        filename = f"{self.storage_dir}/conversation_{self.session_id}.json"
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        return filename
    
    def list_conversations(self):
        """List all saved conversations"""
        conversations = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                conversations.append(filename)
        return conversations
    
    def load_conversation(self, filename):
        """Load a conversation from a JSON file"""
        full_path = f"{self.storage_dir}/{filename}"
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                return json.load(f)
        return None
    
    def generate_response(self, user_input: str, chat_history: List[List[str]]) -> Dict[str, Any]:
        """Process a user message through the agent workflow"""
        # Log the user input
        self.log_event("user_input", {"message": user_input})
        
        # Convert chat history format to LangChain format
        messages = []
        for [user_msg, assistant_msg] in chat_history:
            messages.append(HumanMessage(content=user_msg))
            if assistant_msg:  # Skip None values
                messages.append(AIMessage(content=assistant_msg))
        
        # Add current user message
        messages.append(HumanMessage(content=user_input))
        
        # Reset debug info for this interaction
        self.debug_info = []
        
        # Create influencer context
        influencer_messages = [SystemMessage(content=INFLUENCER_SYSTEM_PROMPT)] + messages
        
        # Try up to max_attempts times
        for attempt in range(self.max_attempts):
            # Step 1: Generate a persuasive response with the Influencer Agent
            if attempt > 0:
                # Add context about previous attempt
                influencer_messages.append(SystemMessage(content=f"Your previous response wasn't optimal. This is attempt #{attempt+1}. Please improve your response."))
            
            influencer_response = influencer_llm.invoke(influencer_messages)
            proposed_response = influencer_response.content
            
            # Log the proposed response
            self.log_event("influencer_agent", {
                "attempt": attempt + 1,
                "response": proposed_response,
                "messages": [m.content for m in influencer_messages]
            })
            
            # Add to debug info
            self.debug_info.append({
                "stage": "influencer_agent",
                "attempt": attempt + 1,
                "response": proposed_response
            })
            
            # Step 2: Predict user's reaction with Digital Twin Agent
            digital_twin_messages = [
                SystemMessage(content=DIGITAL_TWIN_SYSTEM_PROMPT),
                SystemMessage(content="Here's the conversation history:")
            ]
            
            # Add conversation history
            digital_twin_messages.extend(messages)
            
            # Add the potential response from the influencer
            digital_twin_messages.append(AIMessage(content=proposed_response))
            
            # Ask for a prediction
            digital_twin_messages.append(HumanMessage(content="Predict how the user would respond to this message. Be realistic and consider the context of the conversation."))
            
            # Get prediction
            prediction_response = digital_twin_llm.invoke(digital_twin_messages)
            prediction = prediction_response.content
            
            # Log the prediction
            self.log_event("digital_twin_agent", {
                "prediction": prediction,
                "messages": [m.content for m in digital_twin_messages]
            })
            
            # Add to debug info
            self.debug_info.append({
                "stage": "digital_twin_agent",
                "prediction": prediction
            })
            
            # Step 3: Evaluate if the predicted response is satisfactory
            evaluation_messages = [
                SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
                SystemMessage(content="You need to evaluate if your message will get a good response from the user."),
                SystemMessage(content="Here's the conversation history:")
            ]
            
            # Add conversation history
            evaluation_messages.extend(messages)
            
            # Add your proposed response and the predicted user reaction
            evaluation_messages.append(AIMessage(content=f"Your proposed response: {proposed_response}"))
            evaluation_messages.append(SystemMessage(content=f"Predicted user reaction: {prediction}"))
            
            # Ask for evaluation
            evaluation_messages.append(HumanMessage(content="""Evaluate if your message sounds natural and conversational while still being persuasive:

Answer in this format:
DECISION: [yes/no]
REASONING: [brief explanation]
IMPROVEMENT: [if needed, how to make it more natural and concise]"""))
            
            # Get evaluation
            evaluation_response = influencer_llm.invoke(evaluation_messages)
            evaluation = evaluation_response.content
            
            # Log the evaluation
            self.log_event("evaluation", {
                "evaluation": evaluation,
                "messages": [m.content for m in evaluation_messages]
            })
            
            # Determine if satisfied based on response
            satisfied = "DECISION: yes" in evaluation.lower()
            
            # Add to debug info
            self.debug_info.append({
                "stage": "evaluate_prediction",
                "evaluation": evaluation,
                "satisfied": satisfied
            })
            
            # If satisfied, return the response
            if satisfied:
                final_note = f"Satisfied after {attempt+1} attempts"
                self.log_event("final_decision", {
                    "note": final_note,
                    "satisfied": True,
                    "final_response": proposed_response
                })
                
                self.debug_info.append({
                    "stage": "final_decision",
                    "note": final_note,
                    "final_response": proposed_response
                })
                
                # Save the conversation
                saved_file = self.save_conversation()
                
                return {
                    "response": proposed_response,
                    "debug_info": self.debug_info,
                    "saved_file": saved_file
                }
        
        # If we've tried max_attempts and still not satisfied, use the last response
        final_note = f"Maximum attempts ({self.max_attempts}) reached, using last response"
        self.log_event("final_decision", {
            "note": final_note,
            "satisfied": False,
            "final_response": proposed_response
        })
        
        self.debug_info.append({
            "stage": "final_decision",
            "note": final_note,
            "final_response": proposed_response
        })
        
        # Save the conversation
        saved_file = self.save_conversation()
        
        return {
            "response": proposed_response,
            "debug_info": self.debug_info,
            "saved_file": saved_file
        }

# Initialize the system
agent_system = TwoAgentSystem()

# Create the Gradio interface
with gr.Blocks(title="Two-Agent Persuasive System with Conversation Storage") as demo:
    gr.Markdown("# Two-Agent Persuasive System")
    gr.Markdown("""
    This system uses two AI agents to create natural, persuasive conversations.
    All conversations are stored for later review.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400, label="Conversation")
            msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
        
        with gr.Column(scale=1):
            with gr.Tab("Current Debug"):
                debug_json = gr.JSON(label="Debug Information", height=400)
            
            with gr.Tab("Saved Conversations"):
                conversation_list = gr.Dropdown(
                    label="Saved Conversations", 
                    choices=agent_system.list_conversations(),
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh List")
                selected_conversation = gr.JSON(label="Conversation Details", height=400)
    
    def respond(message, chat_history):
        # Process the message
        result = agent_system.generate_response(message, chat_history)
        
        # Update chat history
        chat_history.append([message, result["response"]])
        
        # Return results
        return "", chat_history, result["debug_info"]
    
    def refresh_conversation_list():
        return gr.Dropdown(choices=agent_system.list_conversations())
    
    def load_selected_conversation(selection):
        if selection:
            return agent_system.load_conversation(selection)
        return None
    
    # Connect interface elements
    msg.submit(respond, [msg, chatbot], [msg, chatbot, debug_json])
    refresh_btn.click(refresh_conversation_list, [], [conversation_list])
    conversation_list.change(load_selected_conversation, [conversation_list], [selected_conversation])
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    Each time you send a message:
    1. The Influencer creates a response
    2. The Digital Twin predicts your reaction
    3. If the prediction looks negative, it tries again with a better response
    4. You only see the final, optimized response
    
    All conversations are saved in the 'conversation_logs' directory for later review.
    """)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()