import os
import gradio as gr
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Set up the OpenAI API key
# In practice, use environment variables or a proper configuration file
os.environ["OPENAI_API_KEY"] = ""

# Initialize LLM with different temperatures
influencer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
digital_twin_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define system messages for each agent
INFLUENCER_SYSTEM_PROMPT = """You are the Influencer Agent. Your goal is to craft persuasive responses that will engage users and influence them positively.
Focus on creating messages that build trust and rapport while being ethical and respectful.
Use persuasive techniques like social proof, reciprocity, and authority when appropriate.
Always maintain a positive, helpful tone."""

DIGITAL_TWIN_SYSTEM_PROMPT = """You are the Digital Twin Agent. Your role is to accurately predict how a user would respond to the Influencer Agent's message.
Consider the context of the conversation and the psychology of human responses to persuasive communications.
Provide realistic, authentic responses as if you were the user.
Return ONLY the prediction, nothing else."""

class TwoAgentSystem:
    def __init__(self):
        self.max_attempts = 3
        self.debug_info = []
    
    def generate_response(self, user_input: str, chat_history: List[List[str]]) -> Dict[str, Any]:
        """Process a user message through the agent workflow"""
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
            evaluation_messages.append(HumanMessage(content="""Based on this predicted reaction, are you satisfied with your response? 
Answer in this format:
DECISION: [yes/no]
REASONING: [your explanation]"""))
            
            # Get evaluation
            evaluation_response = influencer_llm.invoke(evaluation_messages)
            evaluation = evaluation_response.content
            
            # Determine if satisfied based on response
            satisfied = "DECISION: yes" in evaluation.lower()
            
            # Log the evaluation
            self.debug_info.append({
                "stage": "evaluate_prediction",
                "evaluation": evaluation,
                "satisfied": satisfied
            })
            
            # If satisfied, return the response
            if satisfied:
                self.debug_info.append({
                    "stage": "final_decision",
                    "note": f"Satisfied after {attempt+1} attempts",
                    "final_response": proposed_response
                })
                return {
                    "response": proposed_response,
                    "debug_info": self.debug_info
                }
        
        # If we've tried max_attempts and still not satisfied, use the last response
        self.debug_info.append({
            "stage": "final_decision",
            "note": f"Maximum attempts ({self.max_attempts}) reached, using last response",
            "final_response": proposed_response
        })
        
        return {
            "response": proposed_response,
            "debug_info": self.debug_info
        }

# Initialize the system
agent_system = TwoAgentSystem()

# Create the Gradio interface
with gr.Blocks(title="Two-Agent Persuasive System") as demo:
    gr.Markdown("# Two-Agent Persuasive System")
    gr.Markdown("""
    This system uses two AI agents:
    1. **Influencer Agent**: Creates persuasive responses
    2. **Digital Twin Agent**: Predicts how you might respond
    
    The system evaluates predictions and regenerates responses until satisfied.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
        
        with gr.Column(scale=1):
            debug_json = gr.JSON(label="Behind the Scenes", visible=True)
    
    def respond(message, chat_history):
        # Process the message
        result = agent_system.generate_response(message, chat_history)
        
        # Update chat history
        chat_history.append([message, result["response"]])
        
        # Return results
        return "", chat_history, result["debug_info"]
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot, debug_json])
    
    gr.Markdown("### How it works")
    gr.Markdown("""
    Each time you send a message:
    1. The Influencer creates a response
    2. The Digital Twin predicts your reaction
    3. If the prediction looks negative, it tries again with a better response
    4. You only see the final, optimized response
    
    The debug panel shows you what's happening behind the scenes!
    """)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()