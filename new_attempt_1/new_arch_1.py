import os
import gradio as gr
import random

# A chat agent with basic persuasion stages
class PersuasionAgent:
    def __init__(self):
        # Define persuasion stages
        self.stages = [
            "RAPPORT",      # Building initial connection
            "UNDERSTAND",   # Learning about the user
            "SUGGEST",      # Suggesting resources (with links)
            "REINFORCE",    # Reinforcing the value of the suggestion
            "CALL_ACTION"   # Encouraging action
        ]
        
        # Templates for each stage
        self.templates = {
            "RAPPORT": [
                "Hi there! How are you doing today?",
                "It's nice to chat with you. What brings you here today?",
                "I've been helping people find resources lately. What topics interest you?"
            ],
            "UNDERSTAND": [
                "That's interesting! Could you tell me more about your interest in {topic}?",
                "I see you're interested in {topic}. What aspects of that do you enjoy most?",
                "What specifically about {topic} would you like to learn more about?"
            ],
            "SUGGEST": [
                "Given your interest in {topic}, I think you might find this resource helpful: https://example.com/resource",
                "I recently came across this article about {topic} that I think you'd enjoy: https://example.com/article",
                "Many people interested in {topic} have found this link valuable: https://example.com/valuable-resource"
            ],
            "REINFORCE": [
                "The resource I shared has helped many people learn more about {topic}.",
                "This link is particularly valuable because it covers {topic} in an accessible way.",
                "I've heard great feedback about this resource from others interested in {topic}."
            ],
            "CALL_ACTION": [
                "Why not take a quick look? You can always come back if you have questions.",
                "I'd love to hear what you think after checking it out!",
                "When you have a moment, click the link - I think you'll find it worth your time."
            ]
        }
        
        self.conversation_history = []
        self.current_stage = "RAPPORT"
        self.identified_topics = []
        self.link_introduced = False
    
    def respond(self, user_message):
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Extract potential topics of interest
        self._extract_topics(user_message)
        
        # Update the current stage based on conversation progress
        self._update_stage()
        
        # Generate response based on current stage
        response = self._generate_response()
        
        # Add agent response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Check if we've introduced a link
        if "https://" in response:
            self.link_introduced = True
        
        return response
    
    def _extract_topics(self, message):
        # Simple topic extraction - in a real system this would be more sophisticated
        common_topics = ["technology", "AI", "music", "movies", "books", "travel", "fitness", "health", "cooking"]
        for topic in common_topics:
            if topic.lower() in message.lower() and topic not in self.identified_topics:
                self.identified_topics.append(topic)
    
    def _update_stage(self):
        # Simple state machine to progress through persuasion stages
        if self.current_stage == "RAPPORT":
            # After first exchange, move to understanding
            if len(self.conversation_history) >= 2:
                self.current_stage = "UNDERSTAND"
                
        elif self.current_stage == "UNDERSTAND":
            # After gathering some info and identifying topics, suggest a resource
            if len(self.identified_topics) > 0 and len(self.conversation_history) >= 4:
                self.current_stage = "SUGGEST"
                
        elif self.current_stage == "SUGGEST":
            # After suggesting a resource, reinforce its value
            if self.link_introduced:
                self.current_stage = "REINFORCE"
                
        elif self.current_stage == "REINFORCE":
            # Finally, encourage action
            self.current_stage = "CALL_ACTION"
    
    def _generate_response(self):
        # Get templates for the current stage
        current_templates = self.templates.get(self.current_stage, self.templates["RAPPORT"])
        
        # Choose a random template
        template = random.choice(current_templates)
        
        # Fill in the topic if needed
        if "{topic}" in template:
            topic = random.choice(self.identified_topics) if self.identified_topics else "general topics"
            response = template.replace("{topic}", topic)
        else:
            response = template
        
        return response
    
    def reset(self):
        self.conversation_history = []
        self.current_stage = "RAPPORT"
        self.identified_topics = []
        self.link_introduced = False

# Create the Gradio interface
def create_chat_interface():
    agent = PersuasionAgent()
    
    with gr.Blocks(title="Persuasive Chat Agent") as interface:
        gr.Markdown("# Persuasive Chat Agent")
        gr.Markdown("This chat agent uses basic persuasion techniques to guide the conversation.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(label="Type your message", placeholder="Hello! How can I help you today?")
                
                with gr.Row():
                    send = gr.Button("Send")
                    clear = gr.Button("Clear Chat")
            
            # Simple debug panel
            with gr.Column(scale=1):
                gr.Markdown("### Current State")
                stage_info = gr.Textbox(label="Current Stage", value=agent.current_stage)
                topics = gr.Textbox(label="Identified Topics", value="None")
                link_status = gr.Textbox(label="Link Introduced", value="No")
        
        def user_input(user_message, history):
            if not user_message:
                return "", history, agent.current_stage, ", ".join(agent.identified_topics) or "None", "Yes" if agent.link_introduced else "No"
            
            bot_response = agent.respond(user_message)
            history = history + [[user_message, bot_response]]
            
            return "", history, agent.current_stage, ", ".join(agent.identified_topics) or "None", "Yes" if agent.link_introduced else "No"
        
        def clear_chat():
            agent.reset()
            return [], "", agent.current_stage, "None", "No"
        
        send.click(user_input, [msg, chatbot], [msg, chatbot, stage_info, topics, link_status])
        msg.submit(user_input, [msg, chatbot], [msg, chatbot, stage_info, topics, link_status])
        clear.click(clear_chat, [], [chatbot, msg, stage_info, topics, link_status])
    
    return interface

# Run the app
if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch()