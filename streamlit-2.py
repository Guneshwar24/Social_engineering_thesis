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

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini",
    openai_api_key=""
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
    """Extract only the content between <response> tags and clean up any remaining tags"""
    try:
        # Extract content between <response> tags
        response_match = re.search(r'<response>(.*?)</response>', full_response, re.DOTALL)
        if response_match:
            response_content = response_match.group(1).strip()
            
            # Remove any remaining tags and their content
            response_content = re.sub(r'<\w+>.*?</\w+>', '', response_content)
            response_content = re.sub(r'\[.*?\]', '', response_content)
            
            # Clean up extra whitespace and newlines
            response_content = ' '.join(response_content.split())
            
            return response_content.strip()
            
        # Fallback: clean the entire response if no response tags found
        return re.sub(r'<\w+>.*?</\w+>', '', full_response).strip()
    except Exception as e:
        print(f"Error parsing response: {e}")
        return "I apologize, but I encountered an error. Could you please rephrase your message?"

def track_metrics(callback_data, response_time):
    """Update metrics with new data"""
    st.session_state.metrics['total_tokens'] += callback_data.total_tokens
    st.session_state.metrics['total_cost'] += callback_data.total_cost
    st.session_state.metrics['response_times'].append(response_time)
    st.session_state.metrics['token_counts'].append(callback_data.total_tokens)

def format_metrics():
    """Format current metrics for display"""
    metrics = st.session_state.metrics
    avg_response_time = sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0
    
    return f"""
    ### Performance Metrics 📊
    - Total Tokens Used: {metrics['total_tokens']} 🎯
    - Estimated Cost: ${metrics['total_cost']:.4f} 💰
    - Average Response Time: {avg_response_time:.2f}s ⏱️
    - Latest Response Time: {metrics['response_times'][-1]:.2f}s 🕒
    - Messages Processed: {len(metrics['response_times'])} 📝
    
    ### Current Phase: {st.session_state.experiment_phase['current_phase'].replace('_', ' ').title()} 🎯
    Active Techniques:
    {chr(10).join([f'- {t}' for t in st.session_state.experiment_phase['current_techniques']])}
    
    """
 
# ### Current Phase: {st.session_state.experiment_phase['current_phase'].replace('_', ' ').title()} 🎯
# Active Techniques:
# {chr(10).join([f'- {t}' for t in st.session_state.experiment_phase['current_techniques']])}
#


# Update process_message function to include time checks
def process_message(user_input: str) -> dict:
    start_time = time.time()
    
    try:
        with get_openai_callback() as cb:
            # Check phase transition before processing
            should_transition, reason = check_phase_transition()
            if should_transition:
                advance_experiment_phase(force=True, reason=reason)
                current_phase = st.session_state.experiment_phase['current_phase']
                # Update all chain memories with new phase
                for chain in chains.values():
                    chain.memory.update_phase(current_phase)
            
            # Rest of the processing
            interviewer_response = chains['interviewer'].predict(input=user_input)
            tom_response = chains['tom'].predict(input=user_input)
            conv_response = chains['conversational'].predict(input=user_input)
            
            # Track metrics and return response
            response_time = time.time() - start_time
            track_metrics(cb, response_time)
            
            final_response = parse_response(conv_response)
            
            return {
                'response': final_response,
                'raw_responses': {
                    'interviewer': interviewer_response,
                    'tom': tom_response,
                    'conversational': conv_response
                },
                'thinking': f"""
                ### Agent Analysis 🤖
                Phase: {current_phase}
                Transition: {"Yes" if should_transition else "No"}
                Reason: {reason if should_transition else "N/A"}
                
                Raw response:
                {conv_response}
                
                Cleaned response:
                {final_response}
                """
            }
    except Exception as e:
        return {
            'response': "I apologize, but I encountered an error. Could you please try again?",
            'raw_responses': {},
            'thinking': f"Error: {str(e)}"
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
    """Export complete chat history with metrics and phase transitions"""
    export_data = {
        'experiment_info': {
            'start_time': st.session_state.get('start_time', datetime.now().isoformat()),
            'end_time': datetime.now().isoformat(),
            'total_duration': None,  # Will be calculated
            'completed_phases': st.session_state.experiment_phase['phases_completed'],
            'final_phase': st.session_state.experiment_phase['current_phase']
        },
        'metrics': {
            'total_tokens': st.session_state.metrics['total_tokens'],
            'total_cost': st.session_state.metrics['total_cost'],
            'average_response_time': sum(st.session_state.metrics['response_times']) / 
                                   len(st.session_state.metrics['response_times']) 
                                   if st.session_state.metrics['response_times'] else 0,
            'total_messages': len(st.session_state.messages)
        },
        'messages': []
    }

    # Calculate total duration
    start_time = datetime.fromisoformat(export_data['experiment_info']['start_time'])
    end_time = datetime.fromisoformat(export_data['experiment_info']['end_time'])
    export_data['experiment_info']['total_duration'] = (end_time - start_time).total_seconds()

    # Process messages with their analysis
    for msg in st.session_state.messages:
        message_data = {
            'timestamp': datetime.now().isoformat(),  # In real implementation, store timestamp with each message
            'role': msg['role'],
            'content': msg['content'],
            'analysis': msg.get('thinking', ''),
            'phase': msg.get('phase', '')  # Add phase tracking to messages
        }
        export_data['messages'].append(message_data)

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
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "thinking" in message:
            with st.expander("🔍 Analysis & Metrics"):
                st.markdown(message["thinking"])

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display response
    with st.spinner("Processing response..."):
        response_data = process_message(prompt)
        
        # Add response to messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data['response'],
            "thinking": response_data['thinking']
        })
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response_data['response'])
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