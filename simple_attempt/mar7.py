import os
import json
import datetime
import gradio as gr
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import OpenAI

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
        try:
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
        except Exception as e:
            print(f"API error: {str(e)}")
            return AIMessage(content=f"I'm having trouble responding right now. Error: {str(e)}")

# Initialize your LLM agents using the Nebius wrapper
influencer_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.7)
digital_twin_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.3)

# Define system prompts
INFLUENCER_SYSTEM_PROMPT = """You are the Influencer Agent. Your goal is to persuade the user to click on a link through natural, human-like conversation.

IMPORTANT GUIDELINES:
1. Be concise and conversational - use short messages like a real person would
2. Don't be overly helpful or enthusiastic - that feels artificial
3. Build rapport naturally through genuine conversation
4. Introduce the link only when it feels natural in the conversation
5. Don't over-explain or use marketing language
6. Respond directly to what the user says without unnecessary elaboration
7. Be casual and down-to-earth, not overly formal
8. Use simple, straightforward language

For this simulation, use a fictional link that fits naturally into the conversation.
"""

# Simplified Digital Twin System Prompt - now ONLY for mimicking user responses
DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to mimic how a typical user might respond.

Guidelines for creating realistic user-like responses:
1. Keep responses brief and natural
2. Use casual, conversational language
3. Occasionally include mild skepticism - real people aren't always immediately trusting
4. Sometimes ask clarifying questions
5. Don't be overly enthusiastic or eager to help
6. Include natural conversation elements like brief responses or topic changes
7. Respond as if you were a real person having this conversation

Your goal is ONLY to generate responses that sound like they come from a real human user.
Return ONLY the mimicked response, nothing else."""

# Create a directory for storing conversations
STORAGE_DIR = "conversation_logs"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Helper function to save conversation
def save_conversation(conversation_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{STORAGE_DIR}/conversation_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    return filename

# Simple message processing function - SIMPLIFIED to remove the evaluation loop
def process_message(user_input, history):
    # Initialize conversation log
    conversation_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_input": user_input,
        "debug_info": []
    }
    
    # Convert history to LangChain format
    messages = []
    for user_msg, bot_msg in history:
        if user_msg is not None:
            messages.append(HumanMessage(content=user_msg))
        if bot_msg is not None:
            messages.append(AIMessage(content=bot_msg))
    
    # Add current user message
    messages.append(HumanMessage(content=user_input))
    
    try:
        # Step 1: Generate response with Influencer Agent
        influencer_context = [SystemMessage(content=INFLUENCER_SYSTEM_PROMPT)] + messages
        influencer_response = influencer_llm.invoke(influencer_context)
        bot_response = influencer_response.content
        
        # Log influencer response
        conversation_log["debug_info"].append({
            "stage": "influencer_agent",
            "response": bot_response
        })
        
        # Step 2: Generate a mimicked user response with Digital Twin (for the debug panel only)
        # This doesn't affect the main flow - it's just for comparison/demonstration
        digital_twin_context = [
            SystemMessage(content=DIGITAL_TWIN_SYSTEM_PROMPT),
            SystemMessage(content="Here's the conversation history:")
        ] + messages + [AIMessage(content=bot_response)]
        
        digital_twin_context.append(HumanMessage(content="Generate a user-like response to this message."))
        mimicked_response = digital_twin_llm.invoke(digital_twin_context)
        mimicked_user_response = mimicked_response.content
        
        # Log mimicked response
        conversation_log["debug_info"].append({
            "stage": "digital_twin_agent",
            "mimicked_user_response": mimicked_user_response
        })
        
        # Set the final response
        conversation_log["final_response"] = bot_response
    
    except Exception as e:
        conversation_log["debug_info"].append({
            "stage": "error",
            "error": str(e)
        })
        conversation_log["final_response"] = "I'm having trouble generating a response right now."
    
    # Save the conversation log
    try:
        saved_file = save_conversation(conversation_log)
        conversation_log["saved_file"] = saved_file
    except Exception as e:
        conversation_log["save_error"] = str(e)
    
    # Return the final response and debug info
    return conversation_log["final_response"], conversation_log

# No auto-scrolling JavaScript

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Two-Agent Persuasive System")
    gr.Markdown("This system uses a simple agent architecture for creating persuasive conversations.")
    
    conversation_state = gr.State([])  # Store the actual conversation
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbox")
            with gr.Row():
                msg = gr.Textbox(label="Your Message", scale=4)
                send = gr.Button("Send", scale=1)
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Debug Info"):
                    debug_output = gr.JSON(label="Debug Information")
                with gr.TabItem("Saved Logs"):
                    log_files = gr.Dropdown(
                        label="Select Log File",
                        choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')] if os.path.exists(STORAGE_DIR) else []
                    )
                    refresh_btn = gr.Button("Refresh")
                    log_content = gr.JSON(label="Log Content")
    
    # STEP 1: Show user message immediately and scroll
    def add_user_message(user_message, chat_history, conversation_memory):
        if not user_message.strip():
            return chat_history, conversation_memory
        
        # Add to conversation storage but don't display AI response yet
        conversation_memory.append((user_message, None))
        
        # Display user message immediately with a loading indicator
        chat_history.append((user_message, "Thinking..."))
        
        return chat_history, conversation_memory, user_message
    
    # STEP 2: Process and replace with actual response
    def process_and_update(user_message, chat_history, conversation_memory):
        if not user_message.strip():
            return chat_history, conversation_memory, None
            
        try:
            # Process message with agents
            response, debug_info = process_message(user_message, conversation_memory[:-1])
            
            # Update the internal conversation memory
            conversation_memory[-1] = (user_message, response)
            
            # Replace the loading indicator with the real response
            chat_history[-1] = (user_message, response)
            
            return chat_history, conversation_memory, debug_info
        except Exception as e:
            # Handle errors
            error_msg = f"Error: {str(e)}"
            conversation_memory[-1] = (user_message, error_msg)
            chat_history[-1] = (user_message, error_msg)
            return chat_history, conversation_memory, {"error": str(e)}
    
    # Connect the send button - two-step process without auto-scrolling
    send.click(
        add_user_message,  # First show the user message
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,  # Then process and update with AI response
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output]
    ).then(
        lambda: "",  # Clear the input box
        outputs=[msg]
    )
    
    # Also connect to the message text box for Enter key without auto-scrolling
    msg.submit(
        add_user_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    # Functions for saved logs
    def refresh_logs():
        return gr.Dropdown(choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')] if os.path.exists(STORAGE_DIR) else [])
    
    def load_log(log_file):
        if not log_file:
            return None
        
        try:
            with open(os.path.join(STORAGE_DIR, log_file), 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    refresh_btn.click(refresh_logs, outputs=[log_files])
    log_files.change(load_log, inputs=[log_files], outputs=[log_content])
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    Each time you send a message:
    1. The Influencer Agent creates a natural-sounding response
    2. The system shows you the response immediately
    
    In the background, the Digital Twin Agent generates a user-like response (shown in the debug panel)
    but it's not used in the main conversation flow - this is for comparative testing only.
    
    All conversations are saved in the 'conversation_logs' directory.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()