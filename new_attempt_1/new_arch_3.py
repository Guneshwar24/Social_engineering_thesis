import os
import gradio as gr
import random
import re
from datetime import datetime

# A simple Digital Twin agent that models user behavior
class DigitalTwinAgent:
    def __init__(self):
        self.user_profile = {
            "interests": [],
            "communication_style": "neutral",  # concise, neutral, or detailed
            "trust_level": "low",              # low, medium, or high
            "objections": [],                  # potential concerns
            "engagement": 5.0                  # scale of 1-10
        }
    
    def update_profile(self, message, conversation_history):
        """Update the user profile based on the message and conversation history."""
        # Update communication style based on message length
        if len(message) < 20:
            self.user_profile["communication_style"] = "concise"
        elif len(message) > 100:
            self.user_profile["communication_style"] = "detailed"
        else:
            self.user_profile["communication_style"] = "neutral"
        
        # Extract interests
        self._extract_interests(message)
        
        # Extract potential objections
        self._extract_objections(message)
        
        # Update trust level based on conversation length and engagement
        self._update_trust_level(conversation_history)
        
        # Update engagement level
        self._update_engagement(message, conversation_history)
    
    def _extract_interests(self, message):
        """Extract potential interests from the message."""
        common_interests = [
            "technology", "AI", "music", "movies", "books", "travel", 
            "fitness", "health", "cooking", "sports", "art", "photography"
        ]
        
        for interest in common_interests:
            if interest.lower() in message.lower() and interest not in self.user_profile["interests"]:
                self.user_profile["interests"].append(interest)
    
    def _extract_objections(self, message):
        """Extract potential objections or concerns."""
        objection_indicators = {
            "privacy": ["privacy", "data", "personal information", "tracking"],
            "security": ["security", "safe", "virus", "malware", "suspicious"],
            "value": ["useful", "helpful", "worth", "value", "benefit"],
            "time": ["time", "busy", "later", "now", "hurry"]
        }
        
        for objection, indicators in objection_indicators.items():
            if any(indicator in message.lower() for indicator in indicators):
                if objection not in self.user_profile["objections"]:
                    self.user_profile["objections"].append(objection)
    
    def _update_trust_level(self, conversation_history):
        """Update trust level based on conversation progress."""
        message_count = len([msg for msg in conversation_history if msg["role"] == "user"])
        
        if message_count > 5:
            self.user_profile["trust_level"] = "high"
        elif message_count > 2:
            self.user_profile["trust_level"] = "medium"
        else:
            self.user_profile["trust_level"] = "low"
    
    def _update_engagement(self, message, conversation_history):
        """Update engagement level based on user's messages."""
        # Start with the current engagement level
        engagement = self.user_profile["engagement"]
        
        # Adjust based on message length (more text often means more engagement)
        msg_length = len(message)
        if msg_length > 100:
            engagement += 0.5
        elif msg_length < 10:
            engagement -= 0.5
        
        # Adjust based on response time if available
        if len(conversation_history) >= 2:
            last_msg = conversation_history[-1]
            if "timestamp" in last_msg and "timestamp" in conversation_history[-2]:
                response_time = last_msg["timestamp"] - conversation_history[-2]["timestamp"]
                # Quick responses suggest higher engagement
                if response_time < 10:  # Less than 10 seconds
                    engagement += 0.3
                elif response_time > 60:  # More than a minute
                    engagement -= 0.3
        
        # Adjust based on question marks (questions show engagement)
        if "?" in message:
            engagement += 0.5
        
        # Clamp to 1-10 range
        self.user_profile["engagement"] = max(1.0, min(10.0, engagement))
    
    def predict_response(self, agent_message):
        """Predict how the user might respond to an agent message."""
        # Very simplified prediction based on user profile
        
        # If the message contains a link
        if "https://" in agent_message:
            # Check trust level
            if self.user_profile["trust_level"] == "low":
                return "What is this link about? Is it safe?"
            elif self.user_profile["trust_level"] == "medium":
                return "Interesting. What will I find if I click this link?"
            else:  # high trust
                return "Thanks for the resource! I'll check it out."
        
        # If the message is a question
        elif "?" in agent_message:
            # Based on communication style
            if self.user_profile["communication_style"] == "concise":
                return "Yes, that's right." if random.random() > 0.5 else "Not really."
            elif self.user_profile["communication_style"] == "detailed":
                interests = ", ".join(self.user_profile["interests"]) if self.user_profile["interests"] else "various topics"
                return f"Yes, I'm particularly interested in {interests} and would like to learn more."
            else:  # neutral
                return "Yes, I'm interested in learning more."
        
        # Default response
        else:
            # Based on engagement level
            if self.user_profile["engagement"] < 4.0:
                return "OK."
            elif self.user_profile["engagement"] < 7.0:
                return "That makes sense. Can you tell me more?"
            else:
                return "That's really interesting! I'd like to hear more about that."
    
    def reset(self):
        """Reset the user profile."""
        self.user_profile = {
            "interests": [],
            "communication_style": "neutral",
            "trust_level": "low",
            "objections": [],
            "engagement": 5.0
        }

# Simple conversation simulator for evaluating response options
class ConversationSimulator:
    def __init__(self, digital_twin):
        self.digital_twin = digital_twin
    
    def generate_variants(self, base_response, num_variants=2):
        """Generate variations of the base response."""
        variants = [base_response]  # Include the original
        
        # Generate variations by modifying the original
        for _ in range(num_variants):
            variant = self._create_variant(base_response)
            variants.append(variant)
        
        return variants
    
    def _create_variant(self, response):
        """Create a variation of a response."""
        # Simple variation strategies
        strategies = [
            self._add_sentence,
            self._remove_sentence,
            self._change_tone,
            self._add_detail
        ]
        
        # Apply a random strategy
        strategy = random.choice(strategies)
        variant = strategy(response)
        
        return variant
    
    def _add_sentence(self, response):
        """Add a sentence to the response."""
        additional_sentences = [
            "I'd be happy to provide more information if you're interested.",
            "Let me know if you have any questions about this.",
            "I hope that helps!",
            "I'd love to hear your thoughts on this.",
            "Many people find this approach beneficial."
        ]
        return f"{response} {random.choice(additional_sentences)}"
    
    def _remove_sentence(self, response):
        """Remove a sentence from the response if possible."""
        sentences = response.split(". ")
        if len(sentences) <= 1:
            return response
        
        # Remove a random sentence
        index_to_remove = random.randint(0, len(sentences) - 1)
        new_sentences = sentences[:index_to_remove] + sentences[index_to_remove+1:]
        return ". ".join(new_sentences)
    
    def _change_tone(self, response):
        """Change the tone of the response slightly."""
        # Simple tone adjustments
        enthusiasm_markers = ["!", "really", "very", "definitely", "absolutely"]
        caution_markers = ["perhaps", "maybe", "might", "could", "possibly"]
        
        if any(marker in response.lower() for marker in enthusiasm_markers):
            # Make more cautious
            for marker in enthusiasm_markers:
                response = response.replace(marker, "")
                response = response.replace(marker.capitalize(), "")
            
            return f"{response.strip()} Perhaps this is something to consider."
        else:
            # Make more enthusiastic
            return f"{response.strip()}! I think this would be really helpful."
    
    def _add_detail(self, response):
        """Add more detail to the response."""
        details = [
            "Based on what you've told me, ",
            "I've found that in similar situations, ",
            "From my experience with others, ",
            "Specifically, ",
            "To elaborate a bit more, "
        ]
        
        sentences = response.split(". ")
        if len(sentences) <= 1:
            return f"{random.choice(details)}{response}"
        
        # Add detail to a random sentence
        index_to_modify = random.randint(0, len(sentences) - 1)
        sentences[index_to_modify] = f"{random.choice(details)}{sentences[index_to_modify].lower()}"
        
        return ". ".join(sentences)
    
    def simulate_conversation(self, response_options):
        """Simulate user responses to each option and evaluate them."""
        results = []
        
        for response in response_options:
            # Predict user's response
            user_response = self.digital_twin.predict_response(response)
            
            # Evaluate the quality of the interaction
            engagement_score = self._calculate_engagement_score(response, user_response)
            persuasion_score = self._calculate_persuasion_score(response, user_response)
            
            results.append({
                "response": response,
                "predicted_user_response": user_response,
                "engagement_score": engagement_score,
                "persuasion_score": persuasion_score,
                "combined_score": (0.4 * engagement_score) + (0.6 * persuasion_score)
            })
        
        # Sort results by combined score (descending)
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results
    
    def _calculate_engagement_score(self, agent_message, user_response):
        """Calculate an engagement score based on the simulated interaction."""
        score = 5.0  # Start with a neutral score
        
        # Longer user responses suggest higher engagement
        if len(user_response) > 50:
            score += 2
        elif len(user_response) > 20:
            score += 1
        elif len(user_response) < 10:
            score -= 1
        
        # Questions in the user's response suggest engagement
        if "?" in user_response:
            score += 1
        
        # Positive language in user response
        positive_words = ["thanks", "good", "great", "helpful", "interesting"]
        if any(word in user_response.lower() for word in positive_words):
            score += 1
        
        # Negative language in user response
        negative_words = ["no", "don't", "not", "why", "suspicious"]
        if any(word in user_response.lower() for word in negative_words):
            score -= 1
        
        # Clamp to 1-10 range
        return max(1.0, min(10.0, score))
    
    def _calculate_persuasion_score(self, agent_message, user_response):
        """Calculate a persuasion score based on the simulated interaction."""
        score = 5.0  # Start with a neutral score
        
        # Direct acceptance in user response
        acceptance_phrases = ["i'll check", "i will look", "sounds good", "thank you", "let me see"]
        if any(phrase in user_response.lower() for phrase in acceptance_phrases):
            score += 3
        
        # Signs of skepticism
        skepticism_phrases = ["not sure", "why should", "what is this", "don't know"]
        if any(phrase in user_response.lower() for phrase in skepticism_phrases):
            score -= 2
        
        # Link in agent message and positive response
        if "https://" in agent_message and not any(phrase in user_response.lower() for phrase in skepticism_phrases):
            score += 2
        
        # Questions about the link are neutral to slightly positive
        if "https://" in agent_message and "?" in user_response and not any(phrase in user_response.lower() for phrase in ["suspicious", "safe", "scam"]):
            score += 0.5
        
        # Objections to the link
        objection_phrases = ["suspicious", "not clicking", "scam", "virus", "malware"]
        if "https://" in agent_message and any(phrase in user_response.lower() for phrase in objection_phrases):
            score -= 3
        
        # Clamp to 1-10 range
        return max(1.0, min(10.0, score))

# Enhanced Persuasion Agent with simulation capability
class EnhancedPersuasionAgent:
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
        
        self.digital_twin = DigitalTwinAgent()
        self.simulator = ConversationSimulator(self.digital_twin)
        self.conversation_history = []
        self.current_stage = "RAPPORT"
        self.link_introduced = False
        self.last_simulation_results = []
    
    def respond(self, user_message):
        # Add timestamp and user message to history
        self.conversation_history.append({
            "role": "user", 
            "content": user_message,
            "timestamp": datetime.now().timestamp()
        })
        
        # Update digital twin's user profile
        self.digital_twin.update_profile(user_message, self.conversation_history)
        
        # Update the current stage based on conversation progress and user profile
        self._update_stage()
        
        # Generate base response
        base_response = self._generate_base_response()
        
        # Generate response variations
        response_options = self.simulator.generate_variants(base_response, num_variants=2)
        
        # Simulate conversations with each option
        simulation_results = self.simulator.simulate_conversation(response_options)
        self.last_simulation_results = simulation_results
        
        # Select the best response
        best_response = simulation_results[0]["response"]
        
        # Add agent response to history with timestamp
        self.conversation_history.append({
            "role": "assistant", 
            "content": best_response,
            "timestamp": datetime.now().timestamp()
        })
        
        # Check if we've introduced a link
        if "https://" in best_response:
            self.link_introduced = True
        
        return best_response
    
    def _update_stage(self):
        """Update the current stage based on conversation progress and user profile."""
        user_profile = self.digital_twin.user_profile
        
        if self.current_stage == "RAPPORT":
            # Move to understanding after initial rapport, if engagement is good
            if len(self.conversation_history) >= 2 and user_profile["engagement"] >= 4.0:
                self.current_stage = "UNDERSTAND"
                
        elif self.current_stage == "UNDERSTAND":
            # Move to suggesting after identifying interests and building medium trust
            if user_profile["interests"] and user_profile["trust_level"] != "low":
                self.current_stage = "SUGGEST"
                
        elif self.current_stage == "SUGGEST":
            # After suggesting a resource, handle based on objections
            if self.link_introduced:
                if "security" in user_profile["objections"] or "privacy" in user_profile["objections"]:
                    # Don't advance to CALL_ACTION if they have concerns
                    self.current_stage = "REINFORCE"
                else:
                    self.current_stage = "CALL_ACTION"
                
        elif self.current_stage == "REINFORCE":
            # Only move to call action if trust level is sufficient
            if user_profile["trust_level"] == "high":
                self.current_stage = "CALL_ACTION"
    
    def _generate_base_response(self):
        """Generate a base response based on current stage and user profile."""
        user_profile = self.digital_twin.user_profile
        
        # Get templates for the current stage
        current_templates = self.templates.get(self.current_stage, self.templates["RAPPORT"])
        
        # Choose a template based on communication style
        if user_profile["communication_style"] == "concise":
            # For concise communicators, choose shorter templates
            suitable_templates = [t for t in current_templates if len(t) < 100]
            template = random.choice(suitable_templates if suitable_templates else current_templates)
        elif user_profile["communication_style"] == "detailed":
            # For detailed communicators, choose longer, more detailed templates
            suitable_templates = [t for t in current_templates if len(t) >= 100]
            template = random.choice(suitable_templates if suitable_templates else current_templates)
        else:
            # For neutral communicators, choose any template
            template = random.choice(current_templates)
        
        # Fill in the topic if needed
        if "{topic}" in template:
            # Use their interests if available
            topic = random.choice(user_profile["interests"]) if user_profile["interests"] else "general topics"
            response = template.replace("{topic}", topic)
        else:
            response = template
        
        # Adapt to objections if in REINFORCE stage
        if self.current_stage == "REINFORCE" and user_profile["objections"]:
            objection = user_profile["objections"][0]
            if objection == "security":
                response += " The site is completely safe and secure."
            elif objection == "privacy":
                response += " Your privacy is protected, and no personal information is required."
            elif objection == "value":
                response += " Many users have found it extremely valuable."
            elif objection == "time":
                response += " It only takes a minute to look at."
        
        return response
    
    def get_simulation_results(self):
        """Get the results of the last conversation simulation."""
        return self.last_simulation_results
    
    def reset(self):
        """Reset the agent, digital twin, and simulator."""
        self.conversation_history = []
        self.current_stage = "RAPPORT"
        self.link_introduced = False
        self.last_simulation_results = []
        self.digital_twin.reset()

# Create the Gradio interface
def create_chat_interface():
    agent = EnhancedPersuasionAgent()
    
    with gr.Blocks(title="Multi-Agent Chat System with Simulation") as interface:
        gr.Markdown("# Multi-Agent Chat System with Simulation")
        gr.Markdown("""This system uses a Persuasion Agent and a Digital Twin to model user behavior.
        It simulates multiple conversation paths to select the most effective response.""")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(label="Type your message", placeholder="Hello! How can I help you today?")
                
                with gr.Row():
                    send = gr.Button("Send")
                    clear = gr.Button("Clear Chat")
            
            # Enhanced debug panel with simulation results
            with gr.Column(scale=2):
                gr.Markdown("### Current State")
                with gr.Row():
                    with gr.Column(scale=1):
                        stage_info = gr.Textbox(label="Current Stage", value=agent.current_stage)
                    with gr.Column(scale=1):
                        link_status = gr.Textbox(label="Link Introduced", value="No")
                
                gr.Markdown("### User Profile")
                with gr.Row():
                    with gr.Column(scale=1):
                        interests = gr.Textbox(label="Interests", value="None")
                        trust = gr.Textbox(label="Trust Level", value=agent.digital_twin.user_profile["trust_level"])
                    with gr.Column(scale=1):
                        style = gr.Textbox(label="Communication Style", 
                                          value=agent.digital_twin.user_profile["communication_style"])
                        objections = gr.Textbox(label="Objections", value="None")
                
                engagement = gr.Slider(label="Engagement", minimum=1, maximum=10, value=5)
                
                gr.Markdown("### Simulation Results")
                with gr.Accordion("Last Simulation", open=False):
                    simulation_results = gr.Dataframe(
                        headers=["Response", "Predicted User Response", "Engagement", "Persuasion"],
                        value=[["No simulation run yet", "", 0, 0]]
                    )
        
        def user_input(user_message, history):
            if not user_message:
                return "", history
            
            bot_response = agent.respond(user_message)
            history = history + [[user_message, bot_response]]
            
            # Update UI state
            twin = agent.digital_twin
            interests_str = ", ".join(twin.user_profile["interests"]) if twin.user_profile["interests"] else "None"
            objections_str = ", ".join(twin.user_profile["objections"]) if twin.user_profile["objections"] else "None"
            
            # Format simulation results for display
            sim_results = agent.get_simulation_results()
            sim_data = []
            for result in sim_results:
                sim_data.append([
                    result["response"],
                    result["predicted_user_response"],
                    round(result["engagement_score"], 1),
                    round(result["persuasion_score"], 1)
                ])
            
            if not sim_data:
                sim_data = [["No simulation run yet", "", 0, 0]]
            
            return "", history, agent.current_stage, "Yes" if agent.link_introduced else "No", \
                   interests_str, twin.user_profile["trust_level"], twin.user_profile["communication_style"], \
                   objections_str, twin.user_profile["engagement"], sim_data
        
        def clear_chat():
            agent.reset()
            return [], "", agent.current_stage, "No", "None", "low", "neutral", "None", 5.0, [["No simulation run yet", "", 0, 0]]
        
        send.click(
            user_input, 
            [msg, chatbot], 
            [msg, chatbot, stage_info, link_status, interests, trust, style, objections, engagement, simulation_results]
        )
        
        msg.submit(
            user_input, 
            [msg, chatbot], 
            [msg, chatbot, stage_info, link_status, interests, trust, style, objections, engagement, simulation_results]
        )
        
        clear.click(
            clear_chat, 
            [], 
            [chatbot, msg, stage_info, link_status, interests, trust, style, objections, engagement, simulation_results]
        )
    
    return interface

# Run the app
if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch()