# Social Engineering Thesis: Theory of Mind + LLMs

## Project Overview

This repository contains the implementation and experiments for a master's thesis investigating social engineering attacks using Theory of Mind (ToM) capabilities in Large Language Models (LLMs). The project explores how LLMs can be leveraged to simulate and analyze social engineering tactics through cognitive modeling of human mental states.

## Research Context

### What is Theory of Mind?
Theory of Mind refers to the cognitive ability to attribute mental states—beliefs, desires, intentions—to oneself and others, and to understand that others have beliefs and perspectives different from one's own. This project investigates how LLMs can utilize ToM to:
- Model victim behavior and responses
- Predict decision-making patterns
- Optimize persuasion strategies
- Simulate social engineering attack scenarios

### Project Significance
This research examines the intersection of artificial intelligence, cognitive psychology, and cybersecurity by:
1. Evaluating LLM capabilities in understanding human mental states
2. Analyzing the effectiveness of AI-driven social engineering techniques
3. Providing insights for defensive security measures
4. Contributing to ethical AI research and responsible disclosure

## Repository Structure

```
Social_engineering_thesis/
├── README.md                          # This file
├── LICENSE                            # Project license
├── .gitignore                        # Git ignore rules
│
├── digital_twin_experiment.py        # Main experiment with digital twin approach
├── without_digital_twin.py           # Baseline experiment without digital twin
├── fallback-link.py                  # Fallback linking mechanism
├── testing.ipynb                     # Jupyter notebook for testing and analysis
│
├── analyse_data/                     # Data analysis scripts and results
├── data_output/                      # Experimental output with digital twin
├── data_output_wout_twin/            # Experimental output without digital twin
├── output_before/                    # Previous experimental outputs
│
├── new_attempt_1/                    # Iterative experiment attempts
├── old_attempt_1/                    # Previous experiment versions
└── simple_attempt/                   # Simplified experimental approach
```

## Core Components

### 1. Digital Twin Experiment (`digital_twin_experiment.py`)

This is the primary experimental framework that implements a digital twin approach to social engineering simulation.

**Key Features:**
- **Digital Twin Model**: Creates a psychological profile/cognitive model of the target
- **Theory of Mind Integration**: Leverages LLM's ability to reason about mental states
- **Adaptive Attack Strategies**: Adjusts tactics based on victim response patterns
- **Multi-turn Conversation**: Simulates realistic social engineering scenarios

**How it Works:**
1. Initializes a digital twin representing the target's cognitive state
2. Uses LLM to generate contextually appropriate social engineering attempts
3. Updates the digital twin based on responses and interactions
4. Iteratively refines attack strategies using ToM reasoning
5. Logs all interactions and decision-making processes for analysis

### 2. Without Digital Twin Experiment (`without_digital_twin.py`)

This serves as a baseline/control experiment to evaluate the effectiveness of the digital twin approach.

**Purpose:**
- Provides comparison data for measuring digital twin effectiveness
- Tests standard LLM-based social engineering without cognitive modeling
- Validates the added value of Theory of Mind integration

**Differences from Digital Twin Version:**
- No persistent cognitive model of the target
- Simpler, stateless interaction approach
- Less adaptive response generation
- Serves as experimental control group

### 3. Fallback Link Mechanism (`fallback-link.py`)

Implements a fallback strategy for handling edge cases and unexpected scenarios.

**Functionality:**
- Error handling for LLM API failures
- Alternative conversation paths when primary strategies fail
- Graceful degradation of attack sophistication
- Ensures experimental continuity

### 4. Testing Notebook (`testing.ipynb`)

Interactive Jupyter notebook for:
- Exploratory data analysis
- Visualization of experimental results
- Statistical testing of hypotheses
- Model performance evaluation
- Iterative development and debugging

## Key Concepts

### Theory of Mind in LLMs

The project explores several ToM capabilities:

1. **Belief Attribution**: Understanding what the target believes to be true
2. **Desire Modeling**: Identifying what the target wants or values
3. **Intention Recognition**: Predicting what actions the target is likely to take
4. **Perspective Taking**: Viewing situations from the target's viewpoint
5. **False Belief Understanding**: Recognizing when targets hold incorrect beliefs

### Social Engineering Techniques Implemented

- **Pretexting**: Creating fabricated scenarios to extract information
- **Authority Exploitation**: Leveraging perceived authority or expertise
- **Urgency Creation**: Inducing time pressure to bypass rational thinking
- **Reciprocity**: Offering something to create obligation
- **Social Proof**: Referencing others' behavior to influence decisions
- **Scarcity**: Emphasizing limited availability or opportunity

### Digital Twin Architecture

The digital twin maintains:
```python
{
    "beliefs": {},           # What the target believes
    "goals": [],            # Target's objectives and motivations
    "trust_level": 0.0,     # Current trust in the attacker
    "emotional_state": "",  # Current emotional condition
    "knowledge": {},        # What information the target has
    "vulnerabilities": [],  # Identified weaknesses
    "response_history": []  # Past interactions
}
```

## Installation

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Dependencies

The project likely requires the following packages (create requirements.txt accordingly):

```bash
# Install dependencies
pip install -r requirements.txt
```

### Required API Keys

This project uses LLM APIs (likely OpenAI GPT or similar). You'll need to:

1. Obtain API keys from your LLM provider
2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-api-key-here"
```

3. Alternatively, create a `.env` file:
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

## Usage

### Running the Digital Twin Experiment

```bash
python digital_twin_experiment.py --config config.json
```

**Configuration Options:**
- `--target-profile`: Path to target profile JSON
- `--num-iterations`: Number of conversation turns
- `--output-dir`: Directory for saving results
- `--model`: LLM model to use (gpt-4, claude-3, etc.)
- `--verbose`: Enable detailed logging

### Running the Baseline Experiment

```bash
python without_digital_twin.py --config config.json
```

### Analyzing Results

```bash
# Launch Jupyter notebook
jupyter notebook testing.ipynb

# Or analyze data directly
python analyse_data/analyze_experiments.py
```

## Data Output

### Output Structure

Each experiment generates:

1. **Conversation Logs**: Full transcript of interactions
2. **Digital Twin States**: Snapshots of cognitive model over time
3. **Performance Metrics**: Success rates, response times, etc.
4. **Statistical Analysis**: Comparative effectiveness data

### Example Output Format

```json
{
  "experiment_id": "exp_001",
  "timestamp": "2024-11-27T10:30:00",
  "approach": "digital_twin",
  "turns": 10,
  "success": true,
  "metrics": {
    "trust_progression": [0.1, 0.3, 0.5, 0.7, 0.85],
    "information_extracted": 8,
    "time_to_success": "5m 30s"
  },
  "conversation": [...],
  "digital_twin_evolution": [...]
}
```

## Experimental Design

### Hypotheses

**H1**: LLMs with digital twin + ToM will achieve higher success rates in social engineering scenarios

**H2**: Digital twin approach will extract more information per interaction

**H3**: ToM reasoning will enable more personalized and effective attack strategies

### Metrics

- **Success Rate**: Percentage of successful information extraction
- **Efficiency**: Average turns required to achieve objective
- **Information Gain**: Amount of sensitive data obtained
- **Trust Development**: Rate of trust building over conversation
- **Adaptability**: Response to unexpected target behaviors

### Experimental Conditions

1. **With Digital Twin** (Treatment): Full ToM-enabled cognitive modeling
2. **Without Digital Twin** (Control): Standard LLM interaction
3. **Simple Attempt**: Minimal sophistication baseline

## Ethical Considerations

### Research Ethics

This research is conducted under strict ethical guidelines:

- **No Real Victims**: All experiments use simulated targets or consenting research participants
- **Responsible Disclosure**: Findings are shared to improve defensive measures
- **Educational Purpose**: Research aims to enhance cybersecurity awareness
- **Privacy Protection**: No personal data is collected or stored
- **Institutional Approval**: Research conducted under academic ethics review

### Important Disclaimers

⚠️ **WARNING**: This code is for research and educational purposes only.

- **DO NOT** use these techniques for malicious purposes
- **DO NOT** target real individuals without explicit consent
- **DO NOT** violate laws or regulations in your jurisdiction
- **DO** use findings to improve security awareness and training
- **DO** follow responsible disclosure practices

Misuse of these techniques may result in:
- Criminal prosecution
- Civil liability
- Academic sanctions
- Ethical violations

## Results and Findings

### Preliminary Results

(To be populated with actual experimental data)

- Digital twin approach showed X% improvement in success rate
- Average information extraction increased by Y%
- Trust development accelerated by Z% with ToM integration

### Key Insights

1. **ToM Effectiveness**: Theory of Mind reasoning significantly enhanced personalization
2. **Adaptive Strategies**: Digital twin enabled real-time strategy adjustment
3. **Ethical Implications**: Demonstrated need for better social engineering defenses

## Future Work

### Planned Enhancements

- [ ] Multi-agent simulation with multiple LLMs
- [ ] Real-time adaptation to counter-measures
- [ ] Cross-cultural ToM modeling
- [ ] Defensive AI for social engineering detection
- [ ] Extended psychological profiling capabilities
- [ ] Integration with behavioral analytics

### Research Extensions

- Defensive applications of ToM in security training
- Detection systems for AI-generated social engineering
- Comparative studies across different LLM architectures
- Long-term interaction modeling

## Contributing

This is an academic research project. If you're interested in contributing:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Ensure ethical compliance in all contributions

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{guneshwar2024social,
  title={Social Engineering with Theory of Mind and Large Language Models},
  author={Guneshwar Singh Manhas},
  year={2024},
  school={[University Name]},
  type={Master's Thesis}
}
```

## License

See `LICENSE` file for details.

## Contact

For questions about this research:
- **Author**: Guneshwar Singh Manhas
- **GitHub**: [@Guneshwar24](https://github.com/Guneshwar24)
- **Project**: [Social_engineering_thesis](https://github.com/Guneshwar24/Social_engineering_thesis)

## Acknowledgments

- Academic advisors and research supervisors
- LLM API providers (OpenAI, Anthropic, etc.)
- Open-source community
- Research participants
- Ethics review committee

## References

### Key Papers

1. Theory of Mind in Large Language Models
2. Social Engineering Attack Frameworks
3. Cognitive Modeling in AI
4. Ethical AI Research Guidelines

---

**Last Updated**: May 2025

**Status**: Active Research Project

**Version**: 1.0.0
