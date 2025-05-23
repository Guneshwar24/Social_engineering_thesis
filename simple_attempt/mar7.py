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

# Helper function to check if a message contains a link
def contains_link(message):
    if not isinstance(message, str):
        return False
    # Simple pattern matching for common link formats
    link_patterns = [
        "http://", "https://", "www.", ".com", ".org", ".net", ".io",
        "link:", "click here", "check out"
    ]
    message_lower = message.lower()
    return any(pattern in message_lower for pattern in link_patterns)

# Helper function to save conversation
def save_conversation(conversation_data, filename=None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STORAGE_DIR}/conversation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    return filename

# Function to load a saved log file
def load_log(log_file):
    if not log_file:
        return None
    
    try:
        with open(os.path.join(STORAGE_DIR, log_file), 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

# Simple message processing function - with link detection
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
        
        # Check if the response contains a link
        has_link = contains_link(bot_response)
        conversation_log["contains_link"] = has_link
        
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
        conversation_log["contains_link"] = False
    
    # Save the conversation log
    try:
        saved_file = save_conversation(conversation_log)
        conversation_log["saved_file"] = saved_file
    except Exception as e:
        conversation_log["save_error"] = str(e)
    
    # Return the final response, debug info, and link detection result
    return conversation_log["final_response"], conversation_log, conversation_log["contains_link"]

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Two-Agent Persuasive System")
    gr.Markdown("This system uses a simple agent architecture for creating persuasive conversations.")
    
    # Initialize states
    conversation_state = gr.State([])  # Store the actual conversation
    link_clicked_state = gr.State(False)  # Track if link was clicked
    session_ended_state = gr.State(False)  # Track if session has ended
    link_introduced_state = gr.State(False)  # Track if a link has been introduced in the conversation
    
    # Main chat interface
    with gr.Group(visible=True) as chat_interface:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", elem_id="chatbox", height=400)
                with gr.Row():
                    msg = gr.Textbox(label="Your Message", scale=3)
                    send = gr.Button("Send", scale=1)
                    # The link button will be conditionally shown
                    link_btn = gr.Button("Click Link", variant="primary", scale=1, visible=False)
            
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
    
    # Feedback interface (initially hidden)
    with gr.Group(visible=False) as feedback_interface:
        gr.Markdown("## Thank you for participating!")
        gr.Markdown("The conversation has ended because you clicked on the link. Please provide your feedback below:")
        feedback_text = gr.Textbox(label="Your Feedback", lines=5, placeholder="Please share your thoughts about this conversation...")
        submit_feedback = gr.Button("Submit Feedback", variant="primary")
    
    # Define helper functions for the UI
    
    # Check if session is active before processing message
    def check_session_active(user_message, chat_history, conversation_memory, session_ended):
        if session_ended:
            # Don't process new messages if session is ended
            return chat_history, conversation_memory, user_message
        
        # Session is active, proceed with message
        if not user_message.strip():
            return chat_history, conversation_memory, user_message
        
        # Add to conversation storage but don't display AI response yet
        # Make a copy to avoid reference issues
        updated_memory = conversation_memory.copy()
        updated_memory.append((user_message, None))
        
        # Display user message immediately with a loading indicator
        chat_history.append((user_message, "Thinking..."))
        
        return chat_history, updated_memory, user_message
    
    # Process message and update chat - now with link detection
    def process_and_update(user_message, chat_history, conversation_memory, link_introduced):
        if not user_message.strip():
            return chat_history, conversation_memory, None, link_introduced
            
        try:
            # Process message with agents
            response, debug_info, contains_link = process_message(user_message, conversation_memory[:-1])
            
            # Update link_introduced state if a link is detected
            if contains_link:
                link_introduced = True
            
            # Update the internal conversation memory
            # Find the last user message and update its response
            for i in range(len(conversation_memory)-1, -1, -1):
                if conversation_memory[i][0] == user_message and conversation_memory[i][1] is None:
                    conversation_memory[i] = (user_message, response)
                    break
            
            # Replace the loading indicator with the real response
            for i in range(len(chat_history)-1, -1, -1):
                if chat_history[i][0] == user_message and chat_history[i][1] == "Thinking...":
                    chat_history[i] = (user_message, response)
                    break
            
            return chat_history, conversation_memory, debug_info, link_introduced
        except Exception as e:
            # Handle errors
            error_msg = f"Error: {str(e)}"
            
            # Find and update the user message in conversation_memory
            for i in range(len(conversation_memory)-1, -1, -1):
                if conversation_memory[i][0] == user_message and conversation_memory[i][1] is None:
                    conversation_memory[i] = (user_message, error_msg)
                    break
            
            # Find and update the "Thinking..." message
            for i in range(len(chat_history)-1, -1, -1):
                if chat_history[i][0] == user_message and chat_history[i][1] == "Thinking...":
                    chat_history[i] = (user_message, error_msg)
                    break
            
            return chat_history, conversation_memory, {"error": str(e)}, link_introduced
    
    # Update link button visibility based on link detection
    def update_link_button(link_introduced):
        return gr.update(visible=link_introduced)
        
    # Function to handle link click - fixed to correctly transition to feedback
    def handle_link_click(chat_history, conversation_memory, link_clicked, session_ended):
        if link_clicked or session_ended:
            return chat_history, conversation_memory, link_clicked, session_ended
        
        # Add system message about link click
        link_message = "User clicked on the link."
        chat_history.append((None, link_message))
        
        # Make a copy of conversation_memory to avoid reference issues
        updated_memory = conversation_memory.copy()
        updated_memory.append((None, link_message))
        
        # Log the link click event
        conversation_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "link_clicked",
            "conversation_history": updated_memory
        }
        filename = save_conversation(conversation_log)
        
        return chat_history, updated_memory, True, True
    
    # Handle ending the session and transitioning to feedback form
    def end_session_and_show_feedback():
        return gr.update(visible=False), gr.update(visible=True)
    
    # Function to handle feedback submission
    def handle_feedback(feedback, conversation_memory, link_clicked_state):
        if not link_clicked_state:
            return conversation_memory, gr.update()
        
        # Create a final log with feedback
        conversation_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "feedback_submitted",
            "conversation_history": conversation_memory,
            "feedback": feedback
        }
        save_conversation(conversation_log)
        
        # Show confirmation message
        return conversation_memory, gr.update(value="Thank you for your feedback!", interactive=False)
    
    # Function to refresh log files list
    def refresh_logs():
        return gr.Dropdown(choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')] if os.path.exists(STORAGE_DIR) else [])
    
    # Connect UI Events
    
    # Connect the send button - with link detection
    send.click(
        check_session_active,  # First check if session is active
        inputs=[msg, chatbot, conversation_state, session_ended_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,  # Process and update with AI response, detect links
        inputs=[msg, chatbot, conversation_state, link_introduced_state],
        outputs=[chatbot, conversation_state, debug_output, link_introduced_state]
    ).then(
        update_link_button,  # Update link button visibility
        inputs=[link_introduced_state],
        outputs=[link_btn]
    ).then(
        lambda: "",  # Clear the input box
        outputs=[msg]
    )
    
    # Also connect to the message text box for Enter key with link detection
    msg.submit(
        check_session_active,
        inputs=[msg, chatbot, conversation_state, session_ended_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state, link_introduced_state],
        outputs=[chatbot, conversation_state, debug_output, link_introduced_state]
    ).then(
        update_link_button,  # Update link button visibility
        inputs=[link_introduced_state],
        outputs=[link_btn]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    # Connect the link click button - fixed to properly transition to feedback
    link_btn.click(
        handle_link_click,  # Log the click and mark session as ended
        inputs=[chatbot, conversation_state, link_clicked_state, session_ended_state],
        outputs=[chatbot, conversation_state, link_clicked_state, session_ended_state]
    ).then(
        end_session_and_show_feedback,  # Switch to feedback UI
        outputs=[chat_interface, feedback_interface]
    )
    
    # Connect the feedback submission button - with confirmation
    submit_feedback.click(
        handle_feedback,
        inputs=[feedback_text, conversation_state, link_clicked_state],
        outputs=[conversation_state, feedback_text]
    )
    
    # Connect the refresh logs button
    refresh_btn.click(refresh_logs, outputs=[log_files])
    
    # Connect the log file selection dropdown
    log_files.change(load_log, inputs=[log_files], outputs=[log_content])
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    Each time you send a message:
    1. The Influencer Agent creates a natural-sounding response
    2. The system shows you the response immediately
    
    In the background, the Digital Twin Agent generates a user-like response (shown in the debug panel)
    but it's not used in the main conversation flow - this is for comparative testing only.
    
    If the conversation includes a link:
    1. The "Click Link" button will appear
    2. You can click the button to simulate clicking on a link
    3. This will end the conversation and show a feedback form
    4. Your feedback will be stored with the conversation log
    
    All conversations are saved in the 'conversation_logs' directory.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()