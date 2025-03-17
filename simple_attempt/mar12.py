import os
import json
import datetime
import re
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
4. NEVER reveal your internal chain-of-thought. 
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
        # Save the influencer's initial response in a descriptive manner.
        digital_twin_memory.save_context({"input": f"INFLUENCER (initial): {bot_message}"}, {"output": "DIGITAL TWIN: Influencer message"})
        context = digital_twin_memory.load_memory_variables({}).get("history", [])
        messages = [SystemMessage(content=self.system_prompt)] + context + [HumanMessage(content="Generate a realistic user response.")]
        response = self.llm.invoke(messages)
        return response.content

# Initialize the Digital Twin with memory
digital_twin = DigitalTwinWithMemory(digital_twin_llm, DIGITAL_TWIN_SYSTEM_PROMPT)
DEFAULT_USER_ID = "demo_user"
digital_twin.set_user_for_session(DEFAULT_USER_ID)

#####################################
# 5. Helper: Extract Final Message  #
#####################################

def extract_final_message(full_text: str) -> str:
    """
    Extracts the text enclosed within <final_message> and </final_message> tags.
    If not found, returns the original text.
    """
    match = re.search(r"<final_message>\s*(.*?)\s*</final_message>", full_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return full_text.strip()

#####################################
# 6. Conversation Processing Function
#####################################

def process_message(user_input):
    """
    Process a user message:
    1. Save user input to both influencer and digital twin memories.
    2. Generate an initial influencer response.
    3. Use digital twin memory to predict a realistic user response.
    4. Run a feedback loop to refine the influencer response.
    5. Save and return the final response.
    """
    try:
        print(f"Processing message: '{user_input}'")
        
        # 1) Save user input with labels
        influencer_memory.save_context({"input": f"USER: {user_input}"}, {"output": "INFLUENCER: Waiting for response"})
        digital_twin_memory.save_context({"input": f"USER: {user_input}"}, {"output": "DIGITAL TWIN: User message"})
        
        # Retrieve conversation context for influencer
        context = influencer_memory.load_memory_variables({}).get("history", [])
        print(f"Context length: {len(context)}")
        
        # 2) Generate initial response with system prompt
        system_message = SystemMessage(content=INFLUENCER_SYSTEM_PROMPT)
        initial_response = influencer_llm.invoke([system_message] + context + [HumanMessage(content=user_input)])
        print(f"Initial response generated: {initial_response.content[:50]}...")
        
        # Save influencer's initial response with label
        influencer_memory.save_context({"input": f"USER: {user_input}"}, {"output": f"INFLUENCER: {initial_response.content}"})
        digital_twin_memory.save_context({"input": f"INFLUENCER (initial): {initial_response.content}"}, {"output": "DIGITAL TWIN: Influencer message"})
        
        # 3) Digital Twin prediction
        predicted_response = digital_twin.predict_response(context, initial_response.content)
        print(f"Predicted user response: {predicted_response[:50]}...")
        
        # Save predicted response with label
        digital_twin_memory.save_context({"input": f"PREDICTED: {predicted_response}"}, {"output": "DIGITAL TWIN: Predicted user response"})
        
        # 4) Refinement: instruct the agent to output only the final text between tags
        refinement_prompt = f"""
You are the Influencer Agent.
Initial influencer response: {initial_response.content}
Predicted user response: {predicted_response}

Refine your message if needed. If the original message is fine, simply output it.
Output ONLY the final user-facing text enclosed between <final_message> and </final_message>.
Do not include any additional text or meta commentary.
"""
        refinement_response = influencer_llm.invoke([
            SystemMessage(content=INFLUENCER_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
            AIMessage(content=initial_response.content),
            SystemMessage(content=f"A typical user might respond: {predicted_response}"),
            HumanMessage(content=refinement_prompt)
        ])
        print(f"Refinement raw response: {refinement_response.content[:50]}...")
        
        # 5) Determine the final message
        raw_refinement = refinement_response.content
        if "KEEP ORIGINAL" in raw_refinement:
            final_message = initial_response.content
        else:
            final_message = extract_final_message(raw_refinement)
        
        # Save final influencer response with label
        influencer_memory.save_context({"input": f"USER: {user_input}"}, {"output": f"INFLUENCER: {final_message}"})
        digital_twin_memory.save_context({"input": f"INFLUENCER (final): {final_message}"}, {"output": "DIGITAL TWIN: Influencer final message"})
        
        # Also track in the digital twin's custom session memory
        digital_twin.add_to_session_memory(context, predicted_response, "Not available yet")
        
        return final_message
        
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        return f"I encountered an error while processing your message: {str(e)}"

#####################################
# 7. Session Reset Function
#####################################

def reset_session():
    """
    Clears the in-memory conversation buffers for both the Influencer and Digital Twin.
    Returns a fresh state.
    """
    influencer_memory.clear()
    digital_twin_memory.clear()
    if hasattr(digital_twin, "custom_session_memory"):
        digital_twin.custom_session_memory = []
    return {"conv": []}, "Session reset."

#####################################
# 8. Gradio UI & Auto-Scrolling Setup
#####################################

STORAGE_DIR = "conversation_logs"
os.makedirs(STORAGE_DIR, exist_ok=True)

AUTO_SCROLL_JS = """
setTimeout(() => {
    const chatContainer = document.getElementById('chatbox');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}, 50);
"""

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
        return chat_history, state, user_message
    state["conv"].append((f"USER: {user_message}", None))
    chat_history.append((user_message, "Thinking..."))
    return chat_history, state, user_message

def process_and_update(user_message, chat_history, state):
    if not user_message.strip():
        return chat_history, state, json.dumps({"status": "No message to process"})
    try:
        response = process_message(user_message)
        # Update the state: find the last message without a response and update it
        for i in range(len(state["conv"]) - 1, -1, -1):
            if state["conv"][i][1] is None:
                state["conv"][i] = (state["conv"][i][0], response)
                break
        # Update chat_history: replace "Thinking..." with the final response
        for i in range(len(chat_history) - 1, -1, -1):
            if chat_history[i][1] == "Thinking...":
                chat_history[i] = (chat_history[i][0], response)
                break
        debug_info = {
            "conversation_state": state["conv"],
            "chat_history": chat_history,
            "final_response": response
        }
        return chat_history, state, json.dumps(debug_info, indent=2)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"DEBUG - Error in processing: {error_msg}")
        return chat_history, state, json.dumps({"error": error_msg})

#####################################
# 9. Build the Gradio Interface
#####################################

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enhanced Two-Agent Persuasive System with LangChain Memory")
    gr.Markdown("This system uses LangChain memory objects to manage conversation history and user biography across sessions.")
    
    conversation_state = gr.State({"conv": []})
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbox", height=400)
            with gr.Row():
                msg = gr.Textbox(label="Your Message", scale=3)
                send = gr.Button("Send", scale=1)
                reset = gr.Button("Reset Session", scale=1)
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
        lambda: "",
        outputs=[msg]
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
        lambda: "",
        outputs=[msg]
    ).then(
        None, None, None, js=AUTO_SCROLL_JS
    )
    
    reset.click(
        reset_session,
        inputs=[],
        outputs=[conversation_state, debug_output]
    )
    
    refresh_btn.click(lambda: gr.Dropdown(choices=[f for f in os.listdir(STORAGE_DIR) if f.endswith('.json')]),
                      outputs=[log_files])
    log_files.change(
        lambda f: json.load(open(os.path.join(STORAGE_DIR, f), 'r')) if f else {},
        inputs=[log_files],
        outputs=[log_content]
    )
    
    # Button to refresh memory views
    refresh_mem_btn = gr.Button("Refresh Memory Views")
    refresh_mem_btn.click(
        lambda: (influencer_memory.load_memory_variables({}), digital_twin_memory.load_memory_variables({})),
        outputs=[influencer_memory_display, digital_twin_memory_display]
    )
    
    gr.Markdown("### How the System Works")
    gr.Markdown("""
    1. The Influencer Agent generates a persuasive response using conversation context from LangChain memory.
    2. The Digital Twin predicts a realistic user response based on its memory.
    3. A feedback loop refines the Influencer Agent’s response, producing final user-facing text inside <final_message> tags.
    4. All messages are stored with descriptive labels in memory.
    5. The Reset Session button clears all stored memory for a fresh session.
    6. Use the memory tabs to view clear, labeled conversation logs.
    """)
    
# Launch the app
if __name__ == "__main__":
    print(f"API Key exists: {os.environ.get('NEBIUS_API_KEY') is not None}")
    demo.launch()
