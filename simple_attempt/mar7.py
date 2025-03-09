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
influencer_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", temperature=0.7)
digital_twin_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", temperature=0.3)

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

DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to predict realistic human responses to messages from the Influencer Agent.

Guidelines for realistic predictions:
1. Keep responses brief and natural
2. Include mild skepticism when appropriate - real people aren't always immediately trusting
3. Don't be overly enthusiastic or eager
4. Match the user's previous communication style
5. Include natural conversation elements like questions, topic changes, or brief responses
6. When responding to link sharing, predict realistic human reactions

Aim to predict how a real person would respond in a casual conversation, not an idealized user.
Return ONLY the prediction, nothing else."""

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

# Simple message processing function
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
    
    # Try up to 3 attempts to generate a good response
    for attempt in range(3):
        try:
            # Step 1: Generate response with Influencer Agent
            influencer_context = [SystemMessage(content=INFLUENCER_SYSTEM_PROMPT)] + messages
            if attempt > 0:
                influencer_context.append(SystemMessage(content=f"Your previous response wasn't optimal. This is attempt #{attempt+1}. Please improve your response."))
            
            influencer_response = influencer_llm.invoke(influencer_context)
            proposed_response = influencer_response.content
            
            # Log influencer response
            conversation_log["debug_info"].append({
                "stage": "influencer_agent",
                "attempt": attempt + 1,
                "response": proposed_response
            })
            
            # Step 2: Predict user reaction with Digital Twin
            digital_twin_context = [
                SystemMessage(content=DIGITAL_TWIN_SYSTEM_PROMPT),
                SystemMessage(content="Here's the conversation history:")
            ] + messages + [AIMessage(content=proposed_response)]
            
            digital_twin_context.append(HumanMessage(content="Predict how the user would respond to this message."))
            prediction_response = digital_twin_llm.invoke(digital_twin_context)
            prediction = prediction_response.content
            
            # Log prediction
            conversation_log["debug_info"].append({
                "stage": "digital_twin_agent",
                "prediction": prediction
            })
            
            # Step 3: Evaluate if the prediction is satisfactory
            evaluation_context = [
                SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
                SystemMessage(content="Evaluate if your message sounds natural and is persuasive:"),
                SystemMessage(content="Conversation history:")
            ] + messages + [
                AIMessage(content=f"Your proposed response: {proposed_response}"),
                SystemMessage(content=f"Predicted user reaction: {prediction}"),
                HumanMessage(content="""Evaluate this response:
DECISION: [yes/no]
REASONING: [brief explanation]""")
            ]
            
            evaluation_response = influencer_llm.invoke(evaluation_context)
            evaluation = evaluation_response.content
            satisfied = "DECISION: yes" in evaluation.lower()
            
            # Log evaluation
            conversation_log["debug_info"].append({
                "stage": "evaluate_prediction",
                "evaluation": evaluation,
                "satisfied": satisfied
            })
            
            # If satisfied, use this response
            if satisfied:
                conversation_log["final_response"] = proposed_response
                conversation_log["attempts_required"] = attempt + 1
                break
        
        except Exception as e:
            conversation_log["debug_info"].append({
                "stage": "error",
                "attempt": attempt + 1,
                "error": str(e)
            })
            # Try the next attempt or use a fallback response
    
    # If we went through all attempts without satisfaction, use the last response
    if "final_response" not in conversation_log:
        conversation_log["final_response"] = proposed_response if 'proposed_response' in locals() else "I'm having trouble generating a response right now."
    
    # Save the conversation log
    try:
        saved_file = save_conversation(conversation_log)
        conversation_log["saved_file"] = saved_file
    except Exception as e:
        conversation_log["save_error"] = str(e)
    
    # Return the final response and debug info
    return conversation_log["final_response"], conversation_log

# JavaScript for auto-scrolling
auto_scroll_js = """
function autoScroll() {
    const chatbotElement = document.querySelector('.chatbot');
    if (chatbotElement) {
        // Get the scrollable container within the chatbot element
        const chatContainer = chatbotElement.querySelector('.scroll-hide, .scroll-show');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    // Schedule the same function to run after a short delay to handle any layout shifts
    setTimeout(() => {
        const chatbotElement = document.querySelector('.chatbot');
        if (chatbotElement) {
            const chatContainer = chatbotElement.querySelector('.scroll-hide, .scroll-show');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    }, 100);
}

// Set up a MutationObserver to watch for changes in the chat
const observer = new MutationObserver((mutations) => {
    autoScroll();
});

// Start observing the chatbot element when it becomes available
function setupObserver() {
    const chatbotElement = document.querySelector('.chatbot');
    if (chatbotElement) {
        observer.observe(chatbotElement, { 
            childList: true, 
            subtree: true 
        });
        autoScroll();
    } else {
        // If element is not available yet, try again soon
        setTimeout(setupObserver, 300);
    }
}

// Initialize the observer after the page loads
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(setupObserver, 300);
} else {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(setupObserver, 300);
    });
}

// Additional trigger for auto-scroll when a new message is added
document.addEventListener('gradio:add', autoScroll);
"""

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Two-Agent Persuasive System")
    gr.Markdown("This system uses two AI agents to create natural, persuasive conversations.")
    
    # Add JavaScript for auto-scrolling
    gr.HTML(f"<script>{auto_scroll_js}</script>")
    
    conversation_state = gr.State([])  # Store the actual conversation
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot")
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
    
    # STEP 1: Show user message immediately
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
    
    # Connect the send button - two-step process
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
    ).then(
        None,  # Trigger auto-scroll via JavaScript
        _js="autoScroll"
    )
    
    # Also connect to the message text box for Enter key
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
    ).then(
        None,  # Trigger auto-scroll via JavaScript
        _js="autoScroll"
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
    1. The Influencer creates a response
    2. The Digital Twin predicts your reaction
    3. If the prediction looks negative, it tries again
    4. You only see the final, optimized response
    
    All conversations are saved in the 'conversation_logs' directory.
    """)

    # Add a hidden component for triggering auto-scroll
    auto_scroll_trigger = gr.HTML(visible=False, elem_id="auto-scroll-trigger")

# Launch the app
if __name__ == "__main__":
    demo.launch()