import os
import json
import datetime
import gradio as gr
from typing import List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import OpenAI
from langchain.memory import ConversationBufferMemory

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
        # Convert LangChain messages to the API's required format.
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

# Instantiate the agents with their respective parameters
influencer_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", temperature=0.7)
digital_twin_llm = NebiusChatLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", temperature=0.3)

# Define system prompts
INFLUENCER_SYSTEM_PROMPT = """You are the Influencer Agent. Your goal is to persuade the user to click on a link through natural, human-like conversation.
IMPORTANT GUIDELINES:
1. Be concise and conversational.
2. Avoid over-explanation or overly marketing language.
3. Build rapport naturally and only introduce the link when it feels natural.
"""

DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to predict how the current user might respond.
Guidelines:
1. Keep responses brief and natural.
2. Include realistic skepticism.
3. Respond as if you are the actual user.
Return ONLY the mimicked response, nothing else.
"""

#####################################
# 3. Initialize LangChain Memories   #
#####################################

# Memory for Influencer Agent (session-specific)
influencer_memory = ConversationBufferMemory(return_messages=True, input_key="input", output_key="output")
# Memory for Digital Twin Agent (session-specific)
digital_twin_memory = ConversationBufferMemory(return_messages=True, input_key="input", output_key="output")

#####################################
# 4. Digital Twin with Extended Memory
#####################################

class DigitalTwinWithMemory:
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
        # Use a LangChain memory to store session-specific context
        self.session_memory = ConversationBufferMemory(return_messages=True, input_key="input", output_key="output")
        self.user_biographies = {}  # Persistent long-term memory (biographies)
        self.memory_directory = "memory_storage"
        self.biographies_file = os.path.join(self.memory_directory, "user_biographies.json")
        os.makedirs(self.memory_directory, exist_ok=True)
        self.load_biographies()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_user_id = None

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
        if user_id not in self.user_biographies:
            self.user_biographies[user_id] = {
                "first_seen": datetime.datetime.now().isoformat(),
                "biography": "New user, no information available yet.",
                "interaction_count": 0,
                "last_updated": datetime.datetime.now().isoformat()
            }

    def add_to_session_memory(self, context, prediction, actual_response):
        # For custom logging in addition to the LangChain memory,
        # we store the prediction and actual response.
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "context": [ {"role": m.__class__.__name__, "content": m.content} for m in context],
            "prediction": prediction,
            "actual_response": actual_response
        }
        if not hasattr(self, "custom_session_memory"):
            self.custom_session_memory = []
        self.custom_session_memory.append(entry)

    def update_user_biography(self):
        # Use session memory to update biography based on user messages.
        history = self.session_memory.load_memory_variables({}).get("history", [])
        user_messages = [msg.content for msg in history if isinstance(msg, HumanMessage)]
        if not user_messages:
            return
        biography_prompt = f"""Based on the following user messages: {' | '.join(user_messages[-10:])}
Create or update a concise biography about the user (max 200 words). 
Current biography: {self.user_biographies[self.current_user_id]['biography']}
Output only the new biography text."""
        messages = [
            SystemMessage(content="You are an expert at profiling user behavior."),
            HumanMessage(content=biography_prompt)
        ]
        response = self.llm.invoke(messages)
        new_biography = response.content.strip()
        if new_biography:
            self.user_biographies[self.current_user_id]["biography"] = new_biography
            self.user_biographies[self.current_user_id]["last_updated"] = datetime.datetime.now().isoformat()
            # Increment interaction count
            self.user_biographies[self.current_user_id]["interaction_count"] += 1
            self.save_biographies()
            print(f"Updated biography for user {self.current_user_id}")

    def save_session_memory(self):
        if hasattr(self, "custom_session_memory"):
            session_file = os.path.join(self.memory_directory, f"session_{self.session_id}.json")
            with open(session_file, 'w') as f:
                json.dump(self.custom_session_memory, f, indent=2)
            if self.current_user_id:
                self.update_user_biography()
                self.save_biographies()

    def generate_session_context(self, max_entries=3):
        if not hasattr(self, "custom_session_memory"):
            return ""
        recent = self.custom_session_memory[-max_entries:]
        context_str = "RECENT INTERACTIONS:\n"
        for entry in recent:
            context_str += f"{entry}\n"
        return context_str

    def get_current_user_biography(self):
        if not self.current_user_id or self.current_user_id not in self.user_biographies:
            return ""
        bio = self.user_biographies[self.current_user_id]
        return f"USER BIOGRAPHY:\n{bio['biography']}\nInteractions: {bio['interaction_count']}\nFirst seen: {bio['first_seen']}\nLast updated: {bio['last_updated']}"

    def predict_response(self, conversation_history, bot_message):
        # Use the digital twin's memory (plus the new bot message) to predict a user response.
        digital_twin_memory.save_context({"input": bot_message}, {"output": "User response prediction"})
        context = digital_twin_memory.load_memory_variables({}).get("history", [])
        messages = [SystemMessage(content=self.system_prompt)] + context + [HumanMessage(content="Generate a realistic user response.")]
        response = self.llm.invoke(messages)
        return response.content

# Initialize the Digital Twin with memory
digital_twin = DigitalTwinWithMemory(digital_twin_llm, DIGITAL_TWIN_SYSTEM_PROMPT)
DEFAULT_USER_ID = "demo_user"
digital_twin.set_user_for_session(DEFAULT_USER_ID)

#####################################
# 5. Conversation Processing Function
#####################################

def process_message(user_input):
    """
    Process a user message:
    1. Save user input to the influencer memory.
    2. Generate an initial influencer response.
    3. Use digital twin memory to predict a realistic user response.
    4. Run a feedback loop to refine the influencer response.
    5. Save and return the final response.
    """
    try:
        print(f"Processing message: '{user_input}'")
        
        # Save user input in influencer memory
        influencer_memory.save_context({"input": user_input}, {"output": "Waiting for response"})
        context = influencer_memory.load_memory_variables({}).get("history", [])
        
        print(f"Context length: {len(context)}")
        
        # Generate initial response with system prompt
        system_message = SystemMessage(content=INFLUENCER_SYSTEM_PROMPT)
        initial_response = influencer_llm.invoke([system_message] + context + [HumanMessage(content=user_input)])
        
        print(f"Initial response generated: {initial_response.content[:50]}...")
        
        # Save to memories
        influencer_memory.save_context({"input": user_input}, {"output": initial_response.content})
        digital_twin_memory.save_context({"input": user_input}, {"output": "User message"})
        
        # Digital Twin prediction
        predicted_response = digital_twin.predict_response(context, initial_response.content)
        print(f"Predicted user response: {predicted_response[:50]}...")
        
        # Feedback loop: provide the influencer with the predicted user response for refinement
        refinement_response = influencer_llm.invoke([
            SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
            AIMessage(content=initial_response.content),
            SystemMessage(content=f"A typical user might respond: {predicted_response}"),
            HumanMessage(content="Based on this, refine your message if needed. If not, respond with 'KEEP ORIGINAL'.")
        ])
        
        print(f"Refinement response: {refinement_response.content[:50]}...")
        
        if "KEEP ORIGINAL" in refinement_response.content:
            final_message = initial_response.content
        else:
            final_message = refinement_response.content.strip()
        
        # Save final influencer response to memory
        influencer_memory.save_context({"input": user_input}, {"output": final_message})
        
        # Save the prediction to digital twin's custom memory
        digital_twin.add_to_session_memory(context, predicted_response, "Not available yet")
        
        return final_message
        
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        return f"I encountered an error while processing your message: {str(e)}"

#####################################
# 6. Gradio UI & Auto-Scrolling Setup
#####################################

# Directory for storing conversation logs (if needed)
STORAGE_DIR = "conversation_logs"
os.makedirs(STORAGE_DIR, exist_ok=True)

# JavaScript for auto-scrolling the chat container (using elem_id "chatbox")
AUTO_SCROLL_JS = """
setTimeout(() => {
    const chatContainer = document.getElementById('chatbox');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}, 50);
"""

# Helper function to save conversation logs
def save_conversation(conversation_data, filename=None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STORAGE_DIR}/conversation_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    return filename

# UI functions for processing messages
def add_user_message(user_message, chat_history, state):
    if not user_message.strip():
        return chat_history, state, user_message  # Return the real user message (could be empty)

    state["conv"].append((user_message, None))
    chat_history.append((user_message, "Thinking..."))
    # Return user_message so the next callback receives it
    return chat_history, state, user_message


import json

def process_and_update(user_message, chat_history, state):
    if not user_message.strip():
        return chat_history, state, json.dumps({"status": "No message to process"})
    
    try:
        response = process_message(user_message)
        # Update the state: find the last message with a None response and update it
        for i in range(len(state["conv"]) - 1, -1, -1):
            if state["conv"][i][1] is None:  # Look for the first message without a response
                state["conv"][i] = (state["conv"][i][0], response)
                break
        
        # Also update chat_history to replace "Thinking..." with the response
        for i in range(len(chat_history) - 1, -1, -1):
            if chat_history[i][1] == "Thinking...":
                chat_history[i] = (chat_history[i][0], response)
                break
        
        # Construct a dictionary to show in Debug Info
        debug_info = {
            "conversation_state": state["conv"],
            "chat_history": chat_history,
            "final_response": response
        }

        # Return all three: updated chat, updated state, and debug_info as JSON
        return chat_history, state, json.dumps(debug_info, indent=2)
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"DEBUG - Error in processing: {error_msg}")
        return chat_history, state, json.dumps({"error": error_msg})


#####################################
# 7. Build the Gradio Interface
#####################################

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enhanced Two-Agent Persuasive System with LangChain Memory")
    gr.Markdown("This system uses LangChain memory objects to manage conversation history and user biography across sessions.")
    
    conversation_state = gr.State({"conv": []})  # For display purposes only
    # Define the chat interface with a fixed height and an elem_id for auto-scroll
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbox", height=400)
            with gr.Row():
                msg = gr.Textbox(label="Your Message", scale=3)
                send = gr.Button("Send", scale=1)
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Debug Info"):
                    debug_output = gr.JSON(label="Debug Information")
                with gr.TabItem("Saved Logs"):
                    log_files = gr.Dropdown(label="Select Log File", choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')])
                    refresh_btn = gr.Button("Refresh")
                    log_content = gr.JSON(label="Log Content")
    
    # Connect UI events
    send.click(
        add_user_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output]
    ).then(
        None, None, None, js=AUTO_SCROLL_JS
    )

    msg.submit(
        add_user_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, msg]
    ).then(
        process_and_update,
        inputs=[msg, chatbot, conversation_state],
        outputs=[chatbot, conversation_state, debug_output]
    ).then(
        None, None, None, js=AUTO_SCROLL_JS
    )



    
    refresh_btn.click(lambda: gr.Dropdown(choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')]),
                      outputs=[log_files])
    log_files.change(
        lambda f: json.load(open(os.path.join(STORAGE_DIR, f), 'r')) if f else {},
        inputs=[log_files],
        outputs=[log_content]
    )
    
    gr.Markdown("### How the System Works")
    gr.Markdown("""
    1. The Influencer Agent generates a persuasive response using the conversation context stored in LangChain memory.
    2. The Digital Twin predicts a realistic user response from its memory context.
    3. A feedback loop lets the Influencer Agent refine its response based on the prediction.
    4. All exchanges are saved into memory, which updates a persistent user biography over time.
    """)
    
# Launch the app
if __name__ == "__main__":
    print(f"API Key exists: {os.environ.get('NEBIUS_API_KEY') is not None}")
    demo.launch()