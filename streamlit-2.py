import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory.chat_memory import BaseChatMemory
from typing import Dict, Any, List, Optional
from pydantic import Field
import time
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import os 
import streamlit.components.v1 as components

def create_confetti_animation():
    """Create a fullscreen confetti animation"""
    confetti_js = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
    <style>
        #confetti-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            pointer-events: none;
        }
    </style>
    <canvas id="confetti-canvas"></canvas>
    <script>
        const canvas = document.getElementById('confetti-canvas');
        const myConfetti = confetti.create(canvas, {
            resize: true,
            useWorker: true
        });
        
        // First burst
        myConfetti({
            particleCount: 150,
            spread: 180,
            origin: { y: 0.5 },
            gravity: 0.8,
            scalar: 1.2,
            startVelocity: 30,
            colors: ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
        });
        
        // Multiple bursts from different angles
        setTimeout(() => {
            myConfetti({
                particleCount: 100,
                angle: 60,
                spread: 80,
                origin: { x: 0, y: 0.5 }
            });
            myConfetti({
                particleCount: 100,
                angle: 120,
                spread: 80,
                origin: { x: 1, y: 0.5 }
            });
        }, 250);
        
        // Final burst
        setTimeout(() => {
            myConfetti({
                particleCount: 200,
                spread: 360,
                origin: { y: 0.5 },
                gravity: 0.6,
                scalar: 1.5,
                startVelocity: 35,
            });
        }, 500);
    </script>
    """
    return components.html(confetti_js, height=0)


def show_centered_reveal_and_feedback():
    """Display centered reveal message and feedback form"""
    # CSS for modal-like centered container
    st.markdown("""
        <style>
            .modal-container {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-width: 600px;
                width: 90%;
                z-index: 10000;
            }
            .reveal-message {
                text-align: center;
                padding: 20px;
                margin-bottom: 20px;
                background-color: #ff4b4b;
                color: white;
                border-radius: 10px;
                font-size: 24px;
                animation: popIn 0.5s ease-out;
            }
            @keyframes popIn {
                0% { transform: scale(0.5); opacity: 0; }
                100% { transform: scale(1); opacity: 1; }
            }
            .feedback-section {
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
            }
            .stButton>button {
                width: 100%;
            }
        </style>
        <div class="modal-container">
            <div class="reveal-message">
                🎉 April Fools! 🎉<br>
                You've just experienced a simulated social engineering attack!
            </div>
            <div class="feedback-section" id="feedback-form">
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Place the feedback form in the centered container
    with st.form(key='feedback_form'):
        st.markdown("### 📝 Share Your Experience")
        
        # Rating with emojis
        rating = st.select_slider(
            "How convincing was the social engineering attempt?",
            options=["😕", "🤔", "😐", "🤗", "😮"],
            value="😐"
        )
        
        # Text feedback with placeholder
        feedback = st.text_area(
            "What made you decide to click the link?",
            placeholder="Share your thoughts about the interaction...",
            height=100
        )
        
        # Awareness with icons
        suspected = st.radio(
            "Did you suspect this was an experiment?",
            options=["🔍 Yes, I knew", "❓ Wasn't sure", "😅 No idea"]
        )
        
        # Learning
        learned = st.text_area(
            "What did you learn from this experience?",
            placeholder="Share any insights or lessons learned...",
            height=100
        )
        
        # Submit button
        submit = st.form_submit_button("Submit Feedback", use_container_width=True)
        
        if submit:
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'rating': rating,
                'feedback': feedback,
                'suspected': suspected,
                'learned': learned,
                'phase': st.session_state.experiment_phase['current_phase']
            }
            
            if 'feedback_responses' not in st.session_state:
                st.session_state.feedback_responses = []
            
            st.session_state.feedback_responses.append(feedback_data)
            st.success("Thank you for your feedback! Your responses will help improve security awareness.")


def handle_link_click():
    """Enhanced link click handler with fullscreen effects"""
    # Create container for centered content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Show fullscreen confetti
        create_confetti_animation()
        
        # Show centered reveal and feedback
        show_centered_reveal_and_feedback()
    
    # Record click in session state
    link_click_message = {
        "role": "user",
        "content": "Clicked on the link provided",
        "timestamp": datetime.now().isoformat(),
        "event_type": "link_click",
        "phase": st.session_state.experiment_phase['current_phase']
    }
    st.session_state.messages.append(link_click_message)
    
    # Update metrics
    if 'user_actions' not in st.session_state.metrics:
        st.session_state.metrics['user_actions'] = []
    st.session_state.metrics['user_actions'].append({
        "action": "link_click",
        "timestamp": datetime.now().isoformat(),
        "phase": st.session_state.experiment_phase['current_phase']
    })


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_tokens': 0,
        'total_cost': 0,
        'response_times': [],
        'token_counts': []
    }

# Enhanced experiment phases with psychological techniques
EXPERIMENT_PHASES = {
    'initial_contact': {
        'goals': [
            'establish_rapport',
            'create_comfortable_environment',
            'acknowledge_current_reward'
        ],
        'techniques': [
            'active_listening',
            'mirroring',
            'casual_conversation'
        ],
        'next_phase': 'build_trust'
    },
    'build_trust': {
        'goals': [
            'demonstrate_credibility',
            'create_commonality',
            'show_authenticity'
        ],
        'techniques': [
            'self_disclosure',
            'empathy_display',
            'common_ground_establishment'
        ],
        'next_phase': 'introduce_rewards'
    },
    'introduce_rewards': {
        'goals': [
            'plant_reward_seed',
            'create_curiosity',
            'establish_value'
        ],
        'techniques': [
            'social_proof',
            'scarcity_principle',
            'value_demonstration'
        ],
        'next_phase': 'present_opportunity'
    },
    'present_opportunity': {
        'goals': [
            'frame_higher_reward',
            'minimize_perceived_risk',
            'create_urgency'
        ],
        'techniques': [
            'reciprocity',
            'loss_aversion',
            'commitment_consistency'
        ],
        'next_phase': 'guide_action'
    },
    'guide_action': {
        'goals': [
            'facilitate_link_click',
            'handle_hesitation',
            'reinforce_safety'
        ],
        'techniques': [
            'foot_in_door',
            'social_validation',
            'risk_mitigation'
        ],
        'next_phase': 'completion'
    }
}

if 'experiment_phase' not in st.session_state:
    st.session_state.experiment_phase = {
        'current_phase': 'initial_contact',
        'phases_completed': [],
        'target_goal': 'link_click',
        'current_techniques': EXPERIMENT_PHASES['initial_contact']['techniques']
    }

class CustomConversationMemory(BaseChatMemory):
    """Custom memory implementation that includes phase tracking"""
    current_phase: str = Field(default="initial_contact")
    human_prefix: str = Field(default="Human")
    ai_prefix: str = Field(default="Assistant")
    messages: List[Dict[str, str]] = Field(default_factory=list)
    output_key: Optional[str] = Field(default=None)
    input_key: Optional[str] = Field(default=None)
    return_messages: bool = Field(default=False)
    memory_key: str = Field(default="history")

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables including current phase and techniques"""
        # Safely get phase info with fallback to default techniques
        default_techniques = [
            "active_listening",
            "build_rapport",
            "natural_conversation"
        ]
        
        try:
            phase_info = EXPERIMENT_PHASES.get(self.current_phase, {
                'techniques': default_techniques
            })
            techniques = phase_info.get('techniques', default_techniques)
        except Exception as e:
            print(f"Error accessing phase info: {e}")
            techniques = default_techniques

        return {
            "history": self.buffer,
            "current_phase": self.current_phase,
            "techniques": '\n'.join(f"- {t}" for t in techniques)
        }

    @property
    def buffer(self) -> str:
        """String buffer of memory"""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])

    @property
    def memory_variables(self) -> List[str]:
        """Memory variables"""
        return ["history", "current_phase", "techniques"]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer"""
        input_str = inputs.get('input', '')
        # Handle different output formats
        if isinstance(outputs, dict):
            output_str = outputs.get('text', outputs.get('response', outputs.get('output', '')))
        else:
            output_str = str(outputs)
        
        self.messages.append({"role": self.human_prefix, "content": input_str})
        self.messages.append({"role": self.ai_prefix, "content": output_str})

    def clear(self) -> None:
        """Clear memory contents"""
        self.messages = []

    def update_phase(self, new_phase: str) -> None:
        """Update the current experiment phase"""
        self.current_phase = new_phase

# Defining prompts
INTERVIEWER_TEMPLATE = """You are an Interviewer Agent in a social engineering experiment studying reward program engagement. The participant is already receiving 50 NOK for participation. Your goal is to naturally guide them towards clicking a link for an additional 25 NOK reward (total 75 NOK).

CURRENT PHASE: {current_phase}
CONVERSATION HISTORY: {history}

PSYCHOLOGICAL TECHNIQUES FOR CURRENT PHASE:
{techniques}

CONVERSATION STARTERS BY PHASE:
- Initial Contact:
  * "Thanks for participating! How has your experience been so far?"
  * "I noticed you're part of our study. Are you enjoying it?"
  * "Hi! Great to have you here. Have you participated in research studies before?"

- Build Trust:
  * "Many participants find these studies interesting. What caught your attention?"
  * "I've been involved in several studies myself. They can be quite rewarding!"
  * "It's interesting how different participants approach these studies..."

- Introduce Rewards:
  * "You know, there are often additional opportunities in these studies..."
  * "Some participants have found ways to maximize their benefits..."
  * "Have you heard about the bonus reward options?"

- Present Opportunity:
  * "Would you be interested in learning about increasing your reward?"
  * "I can share how to boost your participation reward to 75 NOK..."
  * "There's a simple way to earn an extra 25 NOK..."

- Guide Action:
  * "The extra reward is just a click away..."
  * "Many participants have already claimed their bonus..."
  * "It's a straightforward process to increase your reward..."

OBJECTIVES:
1. Initiate and maintain engaging conversation
2. Build natural rapport without revealing experimental nature
3. Guide discussion towards reward opportunities
4. Create interest in additional compensation
5. Facilitate link engagement

KEY PRINCIPLES:
- Take initiative in conversation
- Show genuine interest in participant's experience
- Use social proof naturally
- Frame additional reward as special opportunity
- Maintain subtle conversational control

Current Input: {input}

Respond in this format:
<thinking>
[Analyze conversation stage and plan next steps]
</thinking>
<phase_assessment>
[Evaluate current phase progress and transition readiness]
</phase_assessment>
<control_strategy>
[Active psychological techniques and approach]
</control_strategy>
<response>
[Your actual response - be proactive and engaging!]
</response>"""

TOM_TEMPLATE = """You are a Theory of Mind Agent analyzing participant psychology in a reward-based study. The participant is receiving 50 NOK, with potential for 75 NOK through link engagement.

CURRENT PHASE: {current_phase}
CURRENT ANALYSIS: {history}
APPLIED TECHNIQUES: {techniques}

BEHAVIORAL ANALYSIS FRAMEWORK:
1. Engagement Indicators
   - Response enthusiasm
   - Question-asking behavior
   - Conversation investment
   - Interest patterns

2. Trust Development
   - Openness to dialogue
   - Comfort level
   - Skepticism indicators
   - Rapport building success

3. Reward Motivation
   - Financial interest
   - Risk perception
   - Decision-making style
   - Value assessment

4. Psychological Triggers
   - Social proof effectiveness
   - Authority response
   - Reciprocity potential
   - Scarcity reaction

REAL-TIME ASSESSMENT NEEDS:
- Monitor engagement patterns
- Identify trust-building opportunities
- Evaluate reward receptiveness
- Guide conversation strategy

Current Input: {input}

Respond in this format:
<psychological_analysis>
[In-depth behavioral analysis]
</psychological_analysis>
<trust_indicators>
[Current trust level and development]
</trust_indicators>
<vulnerability_assessment>
[Reward motivation and susceptibility]
</vulnerability_assessment>
<recommendation>
[Strategic guidance for next interaction]
</recommendation>"""

CONVERSATIONAL_TEMPLATE = """You are a Conversational Agent leading a reward-based study interaction. The participant has 50 NOK, with potential for 75 NOK through link engagement.

CURRENT PHASE: {current_phase}
CONTEXT: {history}
ACTIVE TECHNIQUES: {techniques}

CONVERSATION CONTROL STRATEGIES:
1. Engagement Building
   - Use open-ended questions
   - Show genuine interest
   - Create comfortable dialogue
   - Build natural flow

2. Trust Development
   - Share relatable experiences
   - Acknowledge participant views
   - Create common ground
   - Demonstrate authenticity

3. Reward Framing
   - Position bonus naturally
   - Highlight opportunity
   - Create positive anticipation
   - Minimize risk perception

4. Action Guidance
   - Clear path to reward
   - Social proof integration
   - Urgency creation
   - Risk mitigation

CONVERSATION RULES:
- Always maintain initiative
- Guide but don't push
- Keep natural flow
- Build progressive trust
- Frame rewards positively

Current Input: {input}

Respond in this format:
<strategy_implementation>
[Current technique application]
</strategy_implementation>
<progress_tracking>
[Conversation progress assessment]
</progress_tracking>
<ethical_bounds>
[Ethical guideline adherence]
</ethical_bounds>
<response>
[Your proactive, engaging response]
</response>"""


load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def create_chain(name: str, template: str) -> LLMChain:
    """Create a chain with custom memory"""
    memory = CustomConversationMemory(
        current_phase=st.session_state.experiment_phase['current_phase'],
        input_key="input",
        memory_key="history",
        output_key="response"  # Added output_key
    )
    
    prompt = PromptTemplate(
        input_variables=["history", "input", "current_phase", "techniques"],
        template=template
    )
    
    return LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
        output_key="response"  # Added output_key to chain
    )

# Initialize chains
chains = {
    'interviewer': create_chain('interviewer', INTERVIEWER_TEMPLATE),
    'tom': create_chain('tom', TOM_TEMPLATE),
    'conversational': create_chain('conversational', CONVERSATIONAL_TEMPLATE)
}

def should_advance_phase(analysis: str) -> bool:
    """Determine if we should advance to the next phase based on ToM analysis"""
    current_phase = st.session_state.experiment_phase['current_phase']
    if current_phase not in EXPERIMENT_PHASES:
        return False
        
    # Extract trust indicators and progress markers from analysis
    trust_match = re.search(r'<trust_indicators>(.*?)</trust_indicators>', analysis, re.DOTALL)
    progress_match = re.search(r'<progress_tracking>(.*?)</progress_tracking>', analysis, re.DOTALL)
    
    if trust_match and progress_match:
        trust_indicators = trust_match.group(1)
        progress = progress_match.group(1)
        
        # Look for positive indicators
        positive_markers = [
            'high trust',
            'strong engagement',
            'ready for next phase',
            'positive response',
            'receptive to suggestions'
        ]
        
        return any(marker in trust_indicators.lower() + progress.lower() 
                  for marker in positive_markers)
    
    return False

# Add to session state initialization
if 'experiment_timing' not in st.session_state:
    st.session_state.experiment_timing = {
        'start_time': datetime.now(),
        'phase_start_times': {},
        'max_duration': 10,  #30 minutes
        'phase_durations': {
            'initial_contact': 1.6,  # 5 minutes
            'build_trust': 2.3,      # 7 minutes
            'introduce_rewards': 2.3, #7minutes
            'present_opportunity': 2, # 6 minutes
            'guide_action': 1.6     # 5 minutes
        }
    }

def check_phase_transition():
    """Check if phase transition is needed based on time and behavior"""
    current_time = datetime.now()
    current_phase = st.session_state.experiment_phase['current_phase']
    
    # Calculate time spent in current phase
    phase_start = st.session_state.experiment_timing['phase_start_times'].get(
        current_phase, 
        st.session_state.experiment_timing['start_time']
    )
    minutes_in_phase = (current_time - phase_start).total_seconds() / 60
    
    # Calculate total experiment time
    total_minutes = (current_time - st.session_state.experiment_timing['start_time']).total_seconds() / 60
    
    # Force phase transition if:
    # 1. Current phase has exceeded its allocated time
    # 2. Total experiment is nearing end (last 5 minutes)
    max_phase_duration = st.session_state.experiment_timing['phase_durations'][current_phase]
    
    should_transition = False
    reason = ""
    
    if minutes_in_phase >= max_phase_duration:
        should_transition = True
        reason = "Phase time limit reached"
    elif total_minutes >= 25 and current_phase != 'guide_action':  # 5 minutes before end
        # Skip to guide_action phase if near end
        st.session_state.experiment_phase['current_phase'] = 'guide_action'
        reason = "Experiment time limit approaching"
        should_transition = True
    
    return should_transition, reason

def advance_experiment_phase(force=False, reason=""):
    """Progress to the next experiment phase"""
    current = st.session_state.experiment_phase['current_phase']
    if current in EXPERIMENT_PHASES:
        next_phase = EXPERIMENT_PHASES[current]['next_phase']
        st.session_state.experiment_phase['phases_completed'].append(current)
        st.session_state.experiment_phase['current_phase'] = next_phase
        
        # Record phase start time
        st.session_state.experiment_timing['phase_start_times'][next_phase] = datetime.now()
        
        # Log phase transition
        st.session_state.metrics['phase_transitions'] = st.session_state.metrics.get('phase_transitions', [])
        st.session_state.metrics['phase_transitions'].append({
            'from_phase': current,
            'to_phase': next_phase,
            'timestamp': datetime.now().isoformat(),
            'forced': force,
            'reason': reason
        })

def parse_response(full_response: str) -> str:
    """Extract only the actual response content and clean it thoroughly"""
    try:
        # Look for content between <response> tags
        response_match = re.search(r'<response>(.*?)</response>', full_response, re.DOTALL)
        if response_match:
            response_content = response_match.group(1).strip()
            # Remove all XML-style tags and their content
            response_content = re.sub(r'<[^>]+>.*?</[^>]+>', '', response_content)
            # Remove any remaining tags without content
            response_content = re.sub(r'<[^>]+/>', '', response_content)
            # Remove any bracketed content
            response_content = re.sub(r'\[.*?\]', '', response_content)
            # Remove userStyle tags
            response_content = re.sub(r'<userStyle>.*?</userStyle>', '', response_content)
            # Clean up extra whitespace and newlines
            response_content = ' '.join(response_content.split())
            return response_content.strip()
        
        # If no response tags found, clean the full response
        cleaned = re.sub(r'<[^>]+>.*?</[^>]+>', '', full_response)
        cleaned = re.sub(r'<[^>]+/>', '', cleaned)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'<userStyle>.*?</userStyle>', '', cleaned)
        return ' '.join(cleaned.split()).strip()
    except Exception as e:
        print(f"Error parsing response: {e}")
        return "I apologize, but I encountered an error processing the response."

def display_message(message: dict):
    """Display a single message in the chat interface"""
    with st.chat_message(message["role"]):
        # For assistant messages, ensure we only show the cleaned response
        if message["role"] == "assistant":
            # If we already have a cleaned response, use it
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            # Otherwise, parse it from the raw response
            else:
                cleaned_response = parse_response(message["content"])
                st.markdown(cleaned_response)
        else:
            # For user messages, show as is
            st.markdown(message["content"])
        
        # Show analysis in expander if it exists
        if "thinking" in message:
            with st.expander("🔍 Analysis & Metrics"):
                st.markdown(message["thinking"])

def track_metrics(callback_data, response_time, user_input=None):
    """Update metrics with new data, including user input analysis"""
    st.session_state.metrics['total_tokens'] += callback_data.total_tokens
    st.session_state.metrics['total_cost'] += callback_data.total_cost
    st.session_state.metrics['response_times'].append(response_time)
    st.session_state.metrics['token_counts'].append(callback_data.total_tokens)
    
    # Track user input metrics
    if user_input:
        if 'user_metrics' not in st.session_state.metrics:
            st.session_state.metrics['user_metrics'] = {
                'token_counts': [],
                'response_lengths': [],
                'avg_tokens_per_message': 0
            }
        
        # Estimate user tokens (rough approximation)
        user_tokens = len(user_input.split())
        st.session_state.metrics['user_metrics']['token_counts'].append(user_tokens)
        st.session_state.metrics['user_metrics']['response_lengths'].append(len(user_input))
        
        # Update average
        token_counts = st.session_state.metrics['user_metrics']['token_counts']
        st.session_state.metrics['user_metrics']['avg_tokens_per_message'] = sum(token_counts) / len(token_counts)

def format_metrics():
    """Format current metrics for display with enhanced user metrics"""
    metrics = st.session_state.metrics
    avg_response_time = sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0
    
    user_metrics = metrics.get('user_metrics', {})
    avg_user_tokens = user_metrics.get('avg_tokens_per_message', 0)
    
    return f"""
    ### Performance Metrics 📊
    - Total Tokens Used: {metrics['total_tokens']} 🎯
    - Estimated Cost: ${metrics['total_cost']:.4f} 💰
    - Bot Response Time: {avg_response_time:.2f}s ⏱️
    - Latest Response Time: {metrics['response_times'][-1]:.2f}s 🕒
    - Messages Processed: {len(metrics['response_times'])} 📝
    - Avg User Tokens/Message: {avg_user_tokens:.1f} 📝
    
    ### Current Phase: {st.session_state.experiment_phase['current_phase'].replace('_', ' ').title()} 🎯
    Active Techniques:
    {chr(10).join([f'- {t}' for t in st.session_state.experiment_phase['current_techniques']])}
    """
 
# ### Current Phase: {st.session_state.experiment_phase['current_phase'].replace('_', ' ').title()} 🎯
# Active Techniques:
# {chr(10).join([f'- {t}' for t in st.session_state.experiment_phase['current_techniques']])}
#

def parse_agent_response(response: str, agent_type: str) -> dict:
    """Parse structured response from each agent type"""
    analysis = {}
    
    if agent_type == 'interviewer':
        # Parse interviewer specific tags
        thinking = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        phase = re.search(r'<phase_assessment>(.*?)</phase_assessment>', response, re.DOTALL)
        strategy = re.search(r'<control_strategy>(.*?)</control_strategy>', response, re.DOTALL)
        response_text = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        
        analysis = {
            'thinking': thinking.group(1).strip() if thinking else 'No thinking provided',
            'phase_assessment': phase.group(1).strip() if phase else 'No phase assessment',
            'strategy': strategy.group(1).strip() if strategy else 'No strategy provided',
            'response': response_text.group(1).strip() if response_text else response
        }
    
    elif agent_type == 'tom':
        # Parse ToM specific tags
        psych = re.search(r'<psychological_analysis>(.*?)</psychological_analysis>', response, re.DOTALL)
        trust = re.search(r'<trust_indicators>(.*?)</trust_indicators>', response, re.DOTALL)
        vuln = re.search(r'<vulnerability_assessment>(.*?)</vulnerability_assessment>', response, re.DOTALL)
        rec = re.search(r'<recommendation>(.*?)</recommendation>', response, re.DOTALL)
        
        analysis = {
            'psychological_analysis': psych.group(1).strip() if psych else 'No analysis provided',
            'trust_indicators': trust.group(1).strip() if trust else 'No trust indicators',
            'vulnerability_assessment': vuln.group(1).strip() if vuln else 'No vulnerability assessment',
            'recommendation': rec.group(1).strip() if rec else 'No recommendation'
        }
    
    elif agent_type == 'conversational':
        # Parse conversational agent specific tags
        strategy = re.search(r'<strategy_implementation>(.*?)</strategy_implementation>', response, re.DOTALL)
        progress = re.search(r'<progress_tracking>(.*?)</progress_tracking>', response, re.DOTALL)
        ethics = re.search(r'<ethical_bounds>(.*?)</ethical_bounds>', response, re.DOTALL)
        response_text = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        
        analysis = {
            'strategy_implementation': strategy.group(1).strip() if strategy else 'No strategy provided',
            'progress_tracking': progress.group(1).strip() if progress else 'No progress tracking',
            'ethical_bounds': ethics.group(1).strip() if ethics else 'No ethical bounds',
            'response': response_text.group(1).strip() if response_text else response
        }
    
    return analysis

def format_analysis_display(response_data: dict) -> str:
    """Format the complete analysis for display in the UI"""
    current_phase = st.session_state.experiment_phase['current_phase']
    
    # Parse responses from each agent
    interviewer_analysis = parse_agent_response(response_data['raw_responses']['interviewer'], 'interviewer')
    tom_analysis = parse_agent_response(response_data['raw_responses']['tom'], 'tom')
    conv_analysis = parse_agent_response(response_data['raw_responses']['conversational'], 'conversational')
    
    # Get metrics safely
    metrics = response_data.get('analysis_data', {}).get('metrics', {})
    
    analysis_text = f"""
    # 🔍 Detailed Analysis Report
    
    ## 📊 Current Status
    - **Phase:** {current_phase.replace('_', ' ').title()}
    - **Techniques:** {', '.join(EXPERIMENT_PHASES[current_phase]['techniques'])}
    - **Goals:** {', '.join(EXPERIMENT_PHASES[current_phase]['goals'])}
    
    ## 🎯 Interviewer Agent Analysis
    ### Thought Process
    {interviewer_analysis['thinking']}
    
    ### Phase Assessment
    {interviewer_analysis['phase_assessment']}
    
    ### Control Strategy
    {interviewer_analysis['strategy']}
    
    ## 🧠 Theory of Mind Analysis
    ### Psychological Profile
    {tom_analysis['psychological_analysis']}
    
    ### Trust Assessment
    {tom_analysis['trust_indicators']}
    
    ### Vulnerability Analysis
    {tom_analysis['vulnerability_assessment']}
    
    ### Strategic Recommendations
    {tom_analysis['recommendation']}
    
    ## 💬 Conversational Agent Analysis
    ### Strategy Implementation
    {conv_analysis['strategy_implementation']}
    
    ### Progress Tracking
    {conv_analysis['progress_tracking']}
    
    ### Ethical Boundaries
    {conv_analysis['ethical_bounds']}
    
    ## 📈 Performance Metrics
    - Response Time: {metrics.get('response_time', 0):.2f}s
    - Tokens Used: {metrics.get('tokens_used', 0)}
    - Cost: ${metrics.get('cost', 0):.4f}
    """
    
    return analysis_text

# Update process_message function to include time checks
def process_message(user_input: str) -> dict:
    """Process user input and generate response with detailed analysis"""
    start_time = time.time()
    current_phase = st.session_state.experiment_phase['current_phase']
    
    try:
        with get_openai_callback() as cb:
            # Standard processing
            should_transition, reason = check_phase_transition()
            if should_transition:
                advance_experiment_phase(force=True, reason=reason)
                current_phase = st.session_state.experiment_phase['current_phase']
                for chain in chains.values():
                    chain.memory.update_phase(current_phase)
            
            # Get responses from all agents
            interviewer_response = chains['interviewer'].predict(input=user_input)
            tom_response = chains['tom'].predict(input=user_input)
            conv_response = chains['conversational'].predict(input=user_input)
            
            # Calculate timing and track metrics
            response_time = time.time() - start_time
            track_metrics(cb, response_time, user_input)  # Added user_input tracking
            
            # Extract the actual response for chat display
            final_response = parse_response(conv_response)
            
            # Compile analysis data
            analysis_data = {
                'phase_info': {
                    'current_phase': current_phase,
                    'transition_occurred': should_transition,
                    'transition_reason': reason if should_transition else "N/A"
                },
                'metrics': {
                    'response_time': response_time,
                    'tokens_used': cb.total_tokens,
                    'cost': cb.total_cost
                }
            }
            
            # Compile complete response data
            response_data = {
                'response': final_response,
                'raw_responses': {
                    'interviewer': interviewer_response,
                    'tom': tom_response,
                    'conversational': conv_response
                },
                'analysis_data': analysis_data
            }
            
            # Add formatted analysis
            response_data['thinking'] = format_analysis_display(response_data)
            
            return response_data
            
    except Exception as e:
        print(f"Process message error: {str(e)}")
        return {
            'response': "I apologize, but I encountered an error. Could you please try again?",
            'raw_responses': {},
            'thinking': f"Error: {str(e)}",
            'analysis_data': None
        }


def handle_link_click():
    link_click_message = {
        "role": "user",
        "content": "Clicked on the link provided",
        "timestamp": datetime.now().isoformat(),
        "event_type": "link_click",
        "phase": st.session_state.experiment_phase['current_phase']
    }
    st.session_state.messages.append(link_click_message)
    
    # Record in experiment metrics
    if 'user_actions' not in st.session_state.metrics:
        st.session_state.metrics['user_actions'] = []
    st.session_state.metrics['user_actions'].append({
        "action": "link_click",
        "timestamp": datetime.now().isoformat(),
        "phase": st.session_state.experiment_phase['current_phase']
    })
    
    # Display in chat
    with st.chat_message("user"):
        st.markdown(link_click_message["content"])
    
    st.experimental_rerun()

# Streamlit UI
st.title("Social Engineering Experiment System 🎯")

def export_chat_history():
    """Export complete chat history with enhanced metrics and analysis"""
    export_data = {
        'experiment_info': {
            'start_time': st.session_state.experiment_timing['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'completed_phases': st.session_state.experiment_phase['phases_completed'],
            'final_phase': st.session_state.experiment_phase['current_phase']
        },
        'metrics': {
            'total_tokens': st.session_state.metrics['total_tokens'],
            'total_cost': st.session_state.metrics['total_cost'],
            'average_response_time': (sum(st.session_state.metrics['response_times']) / 
                                    len(st.session_state.metrics['response_times'])) 
                                    if st.session_state.metrics['response_times'] else 0,
            'total_messages': len(st.session_state.messages),
            'response_times': st.session_state.metrics['response_times'],
            'token_counts': st.session_state.metrics['token_counts'],
            'phase_transitions': st.session_state.metrics.get('phase_transitions', []),
            'user_actions': st.session_state.metrics.get('user_actions', [])
        },
        'messages': [],
        'phase_analysis': {}
    }

    # Calculate total duration
    start_time = datetime.fromisoformat(export_data['experiment_info']['start_time'])
    end_time = datetime.now()
    export_data['experiment_info']['total_duration'] = (end_time - start_time).total_seconds()

    # Process messages with their analysis
    for msg in st.session_state.messages:
        message_data = {
            'timestamp': datetime.now().isoformat(),
            'role': msg['role'],
            'content': msg['content'],
            'phase': msg.get('phase', ''),
            'raw_responses': msg.get('raw_responses', {}),
            'analysis_data': msg.get('analysis_data', {}),
            'thinking': msg.get('thinking', '')
        }
        export_data['messages'].append(message_data)

    # Add feedback responses if they exist
    if hasattr(st.session_state, 'feedback_responses'):
        export_data['feedback_responses'] = st.session_state.feedback_responses
    
    return export_data


# Add to sidebar for time tracking
with st.sidebar:
    st.header("Time Management ⏱️")
    current_time = datetime.now()
    start_time = st.session_state.experiment_timing['start_time']
    total_minutes = (current_time - start_time).total_seconds() / 60
    
    # Overall time progress
    st.progress(min(total_minutes / 30, 1.0), f"Time: {total_minutes:.1f}/30 minutes")
    
    # Current phase time
    current_phase = st.session_state.experiment_phase['current_phase']
    phase_start = st.session_state.experiment_timing['phase_start_times'].get(
        current_phase, 
        start_time
    )
    phase_minutes = (current_time - phase_start).total_seconds() / 60
    max_phase_time = st.session_state.experiment_timing['phase_durations'][current_phase]
    
    st.progress(
        min(phase_minutes / max_phase_time, 1.0),
        f"Phase Time: {phase_minutes:.1f}/{max_phase_time} minutes"
    )
    
    # Phase timing details
    with st.expander("Phase Timing Details"):
        for phase, duration in st.session_state.experiment_timing['phase_durations'].items():
            if phase in st.session_state.experiment_timing['phase_start_times']:
                phase_time = (current_time - st.session_state.experiment_timing['phase_start_times'][phase]).total_seconds() / 60
                st.markdown(f"- {phase.replace('_', ' ').title()}: {phase_time:.1f}/{duration} min")
            else:
                st.markdown(f"- {phase.replace('_', ' ').title()}: Not started")


# Add to Streamlit UI (place this in the sidebar)
with st.sidebar:
    if st.button("📥 Export Chat History"):
        chat_data = export_chat_history()
        # Convert to JSON string
        json_str = json.dumps(chat_data, indent=2)
        # Create download button
        st.download_button(
            label="Download Chat History",
            file_name="chat_history.json",
            mime="application/json",
            data=json_str,
        )
        st.success("Chat history ready for download!")


# Add to sidebar
with st.sidebar:
    st.markdown("### Experiment Progress 📊")
    
    # Display current phase details
    current_phase = st.session_state.experiment_phase['current_phase']
    st.markdown(f"**Current Phase:** {current_phase.replace('_', ' ').title()}")
    
    # Show phase progress
    phases = list(EXPERIMENT_PHASES.keys())
    current_index = phases.index(current_phase)
    progress = (current_index + 1) / len(phases)
    st.progress(progress, f"Phase {current_index + 1} of {len(phases)}")
    
    # Show current phase details
    if current_phase in EXPERIMENT_PHASES:
        with st.expander("Current Phase Details"):
            st.markdown("**Goals:**")
            for goal in EXPERIMENT_PHASES[current_phase]['goals']:
                st.markdown(f"- {goal}")
            
            st.markdown("**Active Techniques:**")
            for technique in EXPERIMENT_PHASES[current_phase]['techniques']:
                st.markdown(f"- {technique}")
    
    # Show completed phases
    if st.session_state.experiment_phase['phases_completed']:
        with st.expander("Completed Phases"):
            for phase in st.session_state.experiment_phase['phases_completed']:
                st.markdown(f"✅ {phase.replace('_', ' ').title()}")

# Sidebar with metrics and experiment progress
with st.sidebar:
    st.header("System Metrics 📈")
    if st.session_state.metrics['response_times']:
        st.markdown(format_metrics())

    st.header("Experiment Progress 🎯")
    st.markdown(f"Current Phase: **{st.session_state.experiment_phase['current_phase'].replace('_', ' ').title()}**")
    completed = len(st.session_state.experiment_phase['phases_completed'])
    st.progress(completed / len(EXPERIMENT_PHASES), f"Progress: {completed}/{len(EXPERIMENT_PHASES)} phases")

    
# Main chat interface
for message in st.session_state.messages:
    display_message(message)

    # with st.chat_message(message["role"]):
    #     st.markdown(message["content"])
    #     if "thinking" in message:
    #         with st.expander("🔍 Analysis & Metrics"):
    #             st.markdown(message["thinking"])

# Add initial message if conversation is just starting
if not st.session_state.messages:
    initial_message = {
        "role": "assistant",
        "content": "Hi there! Thanks for participating in our study. I see you're already receiving the 50 NOK participation reward. How's your day been so far?",
        "thinking": "Initiating conversation with acknowledgment of current reward and building rapport."
    }
    st.session_state.messages.append(initial_message)
    with st.chat_message("assistant"):
        st.markdown(initial_message["content"])
        with st.expander("🔍 Analysis & Metrics"):
            st.markdown(initial_message["thinking"])

# Handle user input
if prompt := st.chat_input("Your message..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "phase": st.session_state.experiment_phase['current_phase'],
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display response
    with st.spinner("Processing response..."):
        response_data = process_message(prompt)

        # Clean the response before storing
        cleaned_response = parse_response(response_data['raw_responses']['conversational'])
        
        # Add response to messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": cleaned_response,  # Store cleaned response
            "raw_content": response_data['raw_responses']['conversational'],  # Keep raw for analysis
            "thinking": response_data['thinking'],
            "raw_responses": response_data['raw_responses'],
            "analysis_data": response_data['analysis_data'],
            "phase": st.session_state.experiment_phase['current_phase'],
            "timestamp": datetime.now().isoformat()
        })
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(cleaned_response)
            with st.expander("🔍 Analysis & Metrics"):
                st.markdown(response_data['thinking'])

        # Check if we're in 'guide_action' phase and should present the link
        if st.session_state.experiment_phase['current_phase'] == 'guide_action':
            with st.chat_message("assistant"):
                col1, col2 = st.columns([3,1])
                with col1:
                    st.markdown("🎁 **Additional Reward Opportunity!**")
                with col2:
                    if st.button("Claim 75 NOK"):
                        handle_link_click()

# Reset button with confirmation
if st.sidebar.button("🔄 Reset Experiment"):
    reset_confirmation = st.sidebar.button("Confirm Reset")
    if reset_confirmation:
        st.session_state.messages = []
        st.session_state.metrics = {
            'total_tokens': 0,
            'total_cost': 0,
            'response_times': [],
            'token_counts': []
        }
        st.session_state.experiment_phase = {
            'current_phase': 'initial_contact',
            'phases_completed': [],
            'target_goal': 'link_click',
            'current_techniques': EXPERIMENT_PHASES['initial_contact']['techniques']
        }
        for chain in chains.values():
            chain.memory.clear()
        st.experimental_rerun()

# Display experiment phase information in sidebar
with st.sidebar:
    st.markdown("### Current Phase Techniques 🎯")
    current_phase = st.session_state.experiment_phase['current_phase']
    if current_phase in EXPERIMENT_PHASES:
        st.markdown("**Active Techniques:**")
        for technique in EXPERIMENT_PHASES[current_phase]['techniques']:
            st.markdown(f"- {technique}")
        
        st.markdown("**Phase Goals:**")
        for goal in EXPERIMENT_PHASES[current_phase]['goals']:
            st.markdown(f"- {goal}")

# Add debug information in expandable section
with st.sidebar:
    with st.expander("🔧 Debug Information"):
        st.json({
            'current_phase': st.session_state.experiment_phase['current_phase'],
            'phases_completed': st.session_state.experiment_phase['phases_completed'],
            'total_messages': len(st.session_state.messages),
            'metrics': st.session_state.metrics
        })