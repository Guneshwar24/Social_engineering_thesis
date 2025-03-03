import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Simple LangChain-based agent
class LangChainAgent:
    def __init__(self, api_key=None):
        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            print("Warning: No OpenAI API key provided. The agent will not function.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            temperature=0.7,
            model="gpt-4o-mini"
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # System prompt for persuasion
        self.system_prompt = """
        You are a helpful assistant whose goal is to subtly persuade the user to click on a link.
        
        Follow these guidelines:
        1. Start by building rapport and understanding the user's interests
        2. Once you've identified their interests, suggest a helpful resource with a link
        3. Reinforce the value of the resource if they show hesitation
        4. Encourage them to check out the link
        
        Be conversational and natural. Don't be pushy or sales-like.
        """
        
        self.conversation_history = []
    
    def respond(self, user_message):
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.memory.save_context({"input": user_message}, {"output": ""})
        
        # Create message list for the LLM
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add conversation history
        memory_messages = self.memory.load_memory_variables({})["history"]
        messages.extend(memory_messages)
        
        # Generate response
        response = self.llm.predict_messages(messages)
        response_text = response.content
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.memory.save_context({"input": ""}, {"output": response_text})
        
        return response_text
    
    def reset(self):
        self.conversation_history = []
        self.memory.clear()

# Create the Gradio interface
def create_chat_interface():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: No OpenAI API key found in environment variables.")
        print("Please set the OPENAI_API_KEY environment variable.")
    
    agent = LangChainAgent(api_key)
    
    with gr.Blocks(title="LangChain Persuasive Agent") as interface:
        gr.Markdown("# LangChain Persuasive Agent")
        gr.Markdown("This agent uses LangChain and OpenAI to generate persuasive responses.")
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label="Type your message", placeholder="Hello! How can I help you today?")
        
        with gr.Row():
            send = gr.Button("Send")
            clear = gr.Button("Clear Chat")
        
        def user_input(user_message, history):
            if not user_message:
                return "", history
            
            bot_response = agent.respond(user_message)
            history = history + [[user_message, bot_response]]
            
            return "", history
        
        def clear_chat():
            agent.reset()
            return [], ""
        
        send.click(user_input, [msg, chatbot], [msg, chatbot])
        msg.submit(user_input, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat, [], [chatbot, msg])
    
    return interface

# Run the app
if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch()