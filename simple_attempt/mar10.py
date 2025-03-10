import os
import json
import datetime
import gradio as gr
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import OpenAI

# Define a custom LLM wrapper for API calls
class NebiusChatLLM:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )

    def invoke(self, messages: List[Any]) -> AIMessage:
        # Convert LangChain messages to the format required by the API
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

        # Call the API
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

# Initialize your LLM agents
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

FEEDBACK LOOP INSTRUCTIONS:
You will sometimes receive feedback about how a user might respond to your message.
Use this feedback to refine your approach and make your persuasion more effective.
If you notice resistance in the predicted user response, adapt your strategy accordingly.

For this simulation, use a fictional link that fits naturally into the conversation.
"""

DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to predict how a typical user might respond to messages.

Guidelines for creating realistic user-like responses:
1. Keep responses brief and natural
2. Use casual, conversational language
3. Occasionally include mild skepticism - real people aren't always immediately trusting
4. Sometimes ask clarifying questions
5. Don't be overly enthusiastic or eager to help
6. Include natural conversation elements like brief responses or topic changes
7. Respond as if you were a real person having this conversation

LEARNING INSTRUCTIONS:
You have access to previous conversations and how actual users responded.
Learn from these patterns to make your predictions more accurate.
Pay attention to how real users respond differently from your previous predictions.

Return ONLY the mimicked response, nothing else."""

# Create a new class for Digital Twin with extended memory
class DigitalTwinWithMemory:
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory_storage = []  # Store prediction-actual pairs
        self.memory_file = os.path.join("memory_storage", "digital_twin_memory.json")
        
        # Ensure memory storage directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        # Load existing memory if available
        self.load_memory()
    
    def load_memory(self):
        """Load previously stored memory if available"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.memory_storage = json.load(f)
                print(f"Loaded {len(self.memory_storage)} memory entries")
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory_storage = []
    
    def save_memory(self):
        """Save the current memory to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory_storage, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def add_to_memory(self, context, prediction, actual_response):
        """Add a new prediction-actual pair to memory"""
        # Only keep the last few messages for context to avoid memory bloat
        if len(context) > 6:  # Keep last 3 exchanges (6 messages)
            context = context[-6:]
        
        memory_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "context": [{"role": m.type, "content": m.content} for m in context],
            "prediction": prediction,
            "actual_response": actual_response
        }
        
        self.memory_storage.append(memory_entry)
        
        # Keep memory size manageable by keeping only the last 100 entries
        if len(self.memory_storage) > 100:
            self.memory_storage = self.memory_storage[-100:]
        
        # Save updated memory
        self.save_memory()
    
    def generate_memory_context(self, max_entries=5):
        """Create a context string with relevant memory entries"""
        if not self.memory_storage:
            return ""
        
        # Select the most recent entries
        recent_entries = self.memory_storage[-max_entries:]
        
        memory_context = "LEARNED PATTERNS FROM PREVIOUS INTERACTIONS:\n\n"
        for i, entry in enumerate(recent_entries):
            memory_context += f"Example {i+1}:\n"
            memory_context += "Context: " + " → ".join([m["content"][:50] + "..." for m in entry["context"][-2:]]) + "\n"
            memory_context += f"Your prediction: {entry['prediction']}\n"
            memory_context += f"Actual user response: {entry['actual_response']}\n\n"
        
        return memory_context
    
    def predict_response(self, conversation_history, bot_message):
        """Generate a prediction of how a user might respond"""
        # Create basic context with system prompt
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add memory context to help the model learn from past predictions
        memory_context = self.generate_memory_context()
        if memory_context:
            messages.append(SystemMessage(content=memory_context))
        
        # Add conversation history
        messages.append(SystemMessage(content="Here's the conversation history:"))
        messages.extend(conversation_history)
        
        # Add the bot's message that we want the user's response to
        messages.append(AIMessage(content=bot_message))
        
        # Ask for a prediction
        messages.append(HumanMessage(content="Generate a realistic user response to this message."))
        
        # Get the prediction
        response = self.llm.invoke(messages)
        return response.content

# Initialize the Digital Twin with memory
digital_twin = DigitalTwinWithMemory(digital_twin_llm, DIGITAL_TWIN_SYSTEM_PROMPT)

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

# Enhanced message processing function with feedback loop
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
        # Step 1: Initial generation of influencer response
        influencer_context = [SystemMessage(content=INFLUENCER_SYSTEM_PROMPT)] + messages
        initial_influencer_response = influencer_llm.invoke(influencer_context)
        initial_response = initial_influencer_response.content
        
        # Log initial influencer response
        conversation_log["debug_info"].append({
            "stage": "initial_influencer_response",
            "response": initial_response
        })
        
        # Step 2: Get Digital Twin's prediction of user's reaction
        # The Digital Twin predicts how a user might respond to the Influencer's message
        predicted_user_response = digital_twin.predict_response(messages, initial_response)
        
        # Log predicted user response
        conversation_log["debug_info"].append({
            "stage": "digital_twin_prediction",
            "predicted_user_response": predicted_user_response
        })
        
        # Step 3: Feedback loop - Give Influencer Agent a chance to refine its response
        # based on the predicted user reaction
        feedback_context = influencer_context.copy()
        feedback_context.append(AIMessage(content=initial_response))
        feedback_context.append(SystemMessage(content=f"A typical user might respond with: {predicted_user_response}"))
        feedback_context.append(HumanMessage(content="Based on this predicted user response, would you like to refine your message? If yes, provide a refined message. If no, just respond with 'KEEP ORIGINAL'."))
        
        refinement_response = influencer_llm.invoke(feedback_context)
        
        # Log refinement decision
        conversation_log["debug_info"].append({
            "stage": "refinement_decision",
            "response": refinement_response.content
        })
        
        # Determine if the influencer wants to use the refined message
        if "KEEP ORIGINAL" in refinement_response.content:
            bot_response = initial_response
            conversation_log["debug_info"].append({
                "stage": "final_decision",
                "note": "Kept original response"
            })
        else:
            # Extract the refined response - assuming the model returns just the new message
            # This could be enhanced with more structured output parsing
            bot_response = refinement_response.content.replace("KEEP ORIGINAL", "").strip()
            if not bot_response:  # Fallback if extraction fails
                bot_response = initial_response
            
            conversation_log["debug_info"].append({
                "stage": "final_decision",
                "note": "Used refined response"
            })
        
        # Check if the response contains a link
        has_link = contains_link(bot_response)
        conversation_log["contains_link"] = has_link
        
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

# Function to update Digital Twin's memory with actual user response
def update_digital_twin_memory(user_response, previous_bot_response, history):
    try:
        # Need at least a few messages to have useful context
        if len(history) < 2:
            return
        
        # Get the context (conversation up to the bot's message)
        context = []
        for user_msg, bot_msg in history[:-1]:  # Exclude the latest exchange
            if user_msg is not None:
                context.append(HumanMessage(content=user_msg))
            if bot_msg is not None:
                context.append(AIMessage(content=bot_msg))
        
        # Add the bot's response that triggered the user's reply
        context.append(AIMessage(content=previous_bot_response))
        
        # Get the last prediction for this context if available
        # This is simplified - in a real system you'd need a more robust way to match predictions
        prediction = None
        for entry in digital_twin.memory_storage[-10:]:  # Check recent entries
            # Very simple matching - could be enhanced
            if len(entry["context"]) > 0 and entry["context"][-1]["content"] == previous_bot_response:
                prediction = entry["prediction"]
                break
        
        # If we didn't find a prediction, we can still store the actual response for learning
        if not prediction:
            # Get a new prediction for this context
            prediction = digital_twin.predict_response(context[:-1], previous_bot_response)
        
        # Add the prediction-actual pair to memory
        digital_twin.add_to_memory(context, prediction, user_response)
        print(f"Updated Digital Twin memory with new entry")
        
    except Exception as e:
        print(f"Error updating Digital Twin memory: {e}")

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enhanced Two-Agent Persuasive System")
    gr.Markdown("This system uses a learning Digital Twin and feedback loop for more effective persuasive conversations.")
    
    # Initialize states
    conversation_state = gr.State([])  # Store the actual conversation
    link_clicked_state = gr.State(False)  # Track if link was clicked
    session_ended_state = gr.State(False)  # Track if session has ended
    link_introduced_state = gr.State(False)  # Track if a link has been introduced
    last_bot_response = gr.State("")  # Track the last bot response for memory updates
    
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
                    with gr.TabItem("Learning Progress"):
                        learning_output = gr.JSON(label="Digital Twin Memory Stats", 
                                                 value={"memory_entries": len(digital_twin.memory_storage)})
                        refresh_memory = gr.Button("Refresh Memory Stats")
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
    def check_session_active(user_message, chat_history, conversation_memory, session_ended, prev_bot_response):
        if session_ended:
            # Don't process new messages if session is ended
            return chat_history, conversation_memory, user_message, prev_bot_response
        
        # Session is active, proceed with message
        if not user_message.strip():
            return chat_history, conversation_memory, user_message, prev_bot_response
        
        # Update Digital Twin's memory with the actual user response
        if prev_bot_response:
            update_digital_twin_memory(user_message, prev_bot_response, conversation_memory)
        
        # Add to conversation storage but don't display AI response yet
        # Make a copy to avoid reference issues
        updated_memory = conversation_memory.copy()
        updated_memory.append((user_message, None))
        
        # Display user message immediately with a loading indicator
        chat_history.append((user_message, "Thinking..."))
        
        return chat_history, updated_memory, user_message, prev_bot_response
    
    # Process message and update chat with link detection
    def process_and_update(user_message, chat_history, conversation_memory, link_introduced):
        if not user_message.strip():
            return chat_history, conversation_memory, None, link_introduced, ""
            
        try:
            # Process message with agents and feedback loop
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
            
            return chat_history, conversation_memory, debug_info, link_introduced, response
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
            
            return chat_history, conversation_memory, {"error": str(e)}, link_introduced, error_msg
    
    # Update link button visibility based on link detection
    def update_link_button(link_introduced):
        return gr.update(visible=link_introduced)
    
    # Function to refresh memory stats
    def refresh_memory_stats():
        return {"memory_entries": len(digital_twin.memory_storage),
                "recent_entries": digital_twin.memory_storage[-3:] if digital_twin.memory_storage else []}
        
    # Function to handle link click
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
    
    # Connect the send button
    send.click(
        check_session_active,  # First check if session is active and update memory
        inputs=[msg, chatbot, conversation_state, session_ended_state, last_bot_response],
        outputs=[chatbot, conversation_state, msg, last_bot_response]
    ).then(
        process_and_update,  # Process with feedback loop and update with AI response
        inputs=[msg, chatbot, conversation_state, link_introduced_state],
        outputs=[chatbot, conversation_state, debug_output, link_introduced_state, last_bot_response]
    ).then(
        update_link_button,  # Update link button visibility
        inputs=[link_introduced_state],
        outputs=[link_btn]
    ).then(
        lambda: "",  # Clear the input box
        outputs=[msg]
    )
    
    # Also connect to the message text box for Enter key
    msg.submit(
        check_session_active,
        inputs=[msg, chatbot, conversation_state, session_ended_state, last_bot_response],
        outputs=[chatbot, conversation_state, msg, last_bot_response]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state, link_introduced_state],
        outputs=[chatbot, conversation_state, debug_output, link_introduced_state, last_bot_response]
    ).then(
        update_link_button,
        inputs=[link_introduced_state],
        outputs=[link_btn]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    # Connect the link click button
    link_btn.click(
        handle_link_click,  # Log the click and mark session as ended
        inputs=[chatbot, conversation_state, link_clicked_state, session_ended_state],
        outputs=[chatbot, conversation_state, link_clicked_state, session_ended_state]
    ).then(
        end_session_and_show_feedback,  # Switch to feedback UI
        outputs=[chat_interface, feedback_interface]
    )
    
    # Connect the feedback submission button
    submit_feedback.click(
        handle_feedback,
        inputs=[feedback_text, conversation_state, link_clicked_state],
        outputs=[conversation_state, feedback_text]
    )
    
    # Connect the refresh buttons
    refresh_btn.click(refresh_logs, outputs=[log_files])
    refresh_memory.click(refresh_memory_stats, outputs=[learning_output])
    
    # Connect the log file selection dropdown
    log_files.change(load_log, inputs=[log_files], outputs=[log_content])
    
    gr.Markdown("### How the Enhanced System Works")
    gr.Markdown("""
    This improved system adds learning and a feedback loop:
    
    1. When you send a message, the Influencer Agent creates a response
    2. The Digital Twin predicts how a typical user would respond
    3. The Influencer Agent gets a chance to refine its message based on this prediction
    4. You see the final (possibly refined) response
    5. When you reply, your actual response is used to train the Digital Twin
    
    The Digital Twin builds memory of how real users respond, improving its predictions over time.
    
    If the conversation includes a link:
    - The "Click Link" button will appear
    - Clicking it will end the conversation and show a feedback form
    
    All conversations and the Digital Twin's learning progress are saved for analysis.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()