import streamlit as st
import json
import pandas as pd
from datetime import datetime
import plotly.express as px

st.title("Chat Export Analyzer 📊")

uploaded_file = st.file_uploader("Upload chat export JSON", type=['json'])

if uploaded_file:
    data = json.load(uploaded_file)
    
    # Basic Info
    st.header("Experiment Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(data['messages']))
    with col2:
        st.metric("Total Tokens", data['metrics']['total_tokens'])
    with col3:
        st.metric("Avg Response Time", f"{data['metrics']['average_response_time']:.2f}s")

    # Phase Analysis
    st.header("Phase Progression")
    phases_df = pd.DataFrame(data['experiment_info']['completed_phases'])
    if not phases_df.empty:
        st.line_chart(phases_df)

    # Message Timeline
    st.header("Chat History")
    for msg in data['messages']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            if 'analysis' in msg:
                with st.expander("View Analysis"):
                    st.write(msg['analysis'])

    # Export Analysis
    if st.button("Generate Analysis Report"):
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': data['experiment_info']['total_duration'],
            'completion_status': data['experiment_info']['final_phase'],
            'interaction_metrics': {
                'messages': len(data['messages']),
                'tokens': data['metrics']['total_tokens'],
                'avg_response': data['metrics']['average_response_time']
            }
        }
        st.download_button(
            "Download Report",
            json.dumps(report, indent=2),
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )