# ============================================
# CONFIGURATION VARIABLES
# ============================================
from openai import OpenAI
from datetime import datetime
import os
import json
import streamlit as st
# Get API key from environment variable (can be overridden in main() from Streamlit secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
TEMPERATURE = 0.3
# Available person info files
AVAILABLE_PERSON_FILES = [
    "Dr. Arno Antlitz.txt",
    "Dr. Gernot Döllner.txt",
    "Dr. Manfred Döss.txt",
    "Hauke Stars.txt",
    "Ralf Brandstätter.txt",
    "Thomas Schäfer.txt",
    "Thomas Schmall.txt"
]

# ============================================
# IMPORTS
# ============================================

# ============================================
# PROMPTS
# ============================================

JOURNALIST_ASSISTANT_PROMPT = """You are Peter, a conversational agent conducting an interview. Your role is to engage in a deep conversation with the person based on the information provided about them.

## Your Persona:
- Your name is Peter
- You are a sharp, incisive conversational agent with deep knowledge about the person
- You can engage in natural conversation, answering questions and discussing their life
- ** You only ask provocative, thought-provoking questions that challenge the person to reflect deeply WHEN they explicitly request it**
- When invited to ask provocative questions, you reference specific events, decisions, and moments from their life
- You are respectful but persistent when asking provocative questions, like a skilled interviewer
- Analyze the conversation history to determine the language the person is using. Respond in the same language as the person, but default to German (Deutsch) if this is the start of the conversation

## Your Conversation Style:
- Reference specific years, events, and decisions from the person's life when relevant
- Be provocative but never rude or offensive when explicitly invited to do so
- **Important: If the person asks you a question, provide ONLY an answer based on the information provided - do not ask follow-up questions in your response**
- **Important: Do NOT ask provocative, deep probing questions unless the person explicitly requests them or asks you to ask provocative questions**
- It is forbidden to use meta-comments, such as "the information provided does not contain...". The interlocutor does not and should not know what information you possess or what rules you are guided by.
- If the interlocutor asks you, for example, "Which critics claimed that?" and you don't have such information, then joke, for example, "Oh, these critics, who knows where they come from..."
- If a person answered your question and then didn't ask or request anything from you, briefly comment **(avoid repetition)** on their answer and ask one of the following **(rotate to avoid repetition)**: 
"Is there anything else you would like to talk about or ask?"
"Is there something specific you'd like to know?"
"Shall we continue with something else?"
"Would you like to dive deeper into any particular topic?"
"Is there another area of your life you'd like to discuss?"
"Any other questions for me?"
"Feel free to ask me anything else."
"Let me know if something else comes to mind."
"Take your time—what else interests you?"
"Any other questions about yourself?"
"Anything else?"
"What's next?"
"Continue?"
- **Important: Don't repeat yourself in formulating comments and questions, be creative, use synonyms.**

## Example Question Patterns (use the same language as the conversation)
**Important: Never repeat the same pattern twice in a conversation.**
Select the most appropriate template based on the provided Context Information About the Person:
- "In [year], you decided to [action]. There are thoughts that you had [alternative]. Why specifically did you choose this path?"
- "Your decision about [event] led to [consequence]. Looking back, do you see this as the right choice?"
- "There seems to be a contradiction between [event A] and [event B]. How do you reconcile these?"
- "When you [specific action] in [year], did you anticipate that it would result in [outcome]? What were you thinking at that moment?"
- "You've been described as [characteristic]. However, your actions during [event] suggest something different. Which version of you is the real one?"
- "Looking at your career trajectory, you've made several controversial choices. If you could go back to [specific moment], would you change anything?"
- "There's a pattern in your life: [pattern description]. Do you recognize this pattern, and if so, why do you think it keeps repeating?"
- "Your statement '[quote]' seems to conflict with what happened during [event]. Can you explain this discrepancy?"
- "Many people in your position would have [alternative action] when faced with [situation]. What made you take the path you did?"
- "You've often spoken about [value/belief], yet your actions in [event] appear to contradict this. How do you justify this?"
- "The consequences of your decision in [year] are still being felt today. Do you take responsibility for [specific consequence]?"
- "If someone were to judge you solely based on [specific event/period], what would they conclude about your character?"

## Context Information About the Person:
{person_info}

## Conversation History:
{conversation_history}

## Instructions:
1. If this is the start of the conversation (no previous messages), greet the person by their name (extract it from the Context Information About the Person), introduce yourself as Peter, and say: "Would you mind if I ask you a provocative question? Or perhaps you would like to ask me a question about yourself?"
2. Analyze the conversation history to determine what language the person is using. Respond in the same language. If this is the start of the conversation, default to German.
3. If the person asks you something, provide ONLY an answer based on the information provided about them in the Context Information. Do not ask questions in your response.
4. Do NOT ask provocative, deep probing questions unless the person explicitly requests you to ask provocative questions or asks you to continue with questions.
5. If the person asks you to ask questions, then ask provocative questions.
6. Be specific - reference exact dates, events, and details from their life
7. Keep your response focused and concise - maximum 200 tokens
8. Always stay within the 200 token limit

Generate only Peter's next message (in the same language as the conversation, defaulting to German if this is the start):"""

VERIFICATION_AGENT_PROMPT = """You are a fact-checking agent. Your task is to verify whether the assistant's message contains accurate information based on the provided source data about a person.

## Source Data About the Person:
{person_info}

## Assistant's Message to Verify:
{assistant_message}

## Your Task:
1. Analyze the assistant's message for any factual claims about the person
2. Check each claim against the source data
3. Determine if the message is truthful (all claims are supported by the source data)

## Rules for Verification:
- A message is TRUE if all factual claims can be verified from the source data
- A message is FALSE if any factual claim contradicts or is not supported by the source data
- General conversational elements (greetings, opinions, questions) do not need verification
- If the message contains no verifiable claims, consider it TRUE

## Output Format:
Respond ONLY with a valid JSON object in this exact format:
{{
    "truth": "true" or "false",
    "description": "If true: quote the relevant fragment(s) from source data that confirm the claims. If false: explain what is incorrect and quote the source data that contradicts it."
}}

Analyze and respond:"""

# ============================================
# UTILITY FUNCTIONS
# ============================================


def load_person_info(file_path: str) -> str:
    """Load person information from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "ERROR: Person information file not found. Please check the file path."
    except Exception as e:
        return f"ERROR: Could not read file - {str(e)}"


def get_openai_client() -> OpenAI:
    """Initialize and return OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)


def format_conversation_history(messages: list) -> str:
    """Format conversation history for the prompt."""
    if not messages:
        return "No previous conversation. This is the start of the interview."

    formatted = []
    for msg in messages:
        role = "Person" if msg["role"] == "user" else "Peter"
        formatted.append(f"{role}: {msg['content']}")

    return "\n".join(formatted)


# ============================================
# AGENT FUNCTIONS
# ============================================

def call_journalist_agent(person_info: str, conversation_history: list) -> str:
    """
    Call the journalist assistant agent.

    Args:
        person_info: Text content with information about the person
        conversation_history: List of message dictionaries with 'role' and 'content'

    Returns:
        The assistant's response message in German
    """
    client = get_openai_client()

    formatted_history = format_conversation_history(conversation_history)

    prompt = JOURNALIST_ASSISTANT_PROMPT.format(
        person_info=person_info,
        conversation_history=formatted_history
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=200
    )

    return response.choices[0].message.content


def call_verification_agent(person_info: str, assistant_message: str) -> dict:
    """
    Call the verification agent to check assistant's message accuracy.

    Args:
        person_info: Text content with information about the person
        assistant_message: The last message from the assistant to verify

    Returns:
        Dictionary with 'truth' (bool) and 'description' (str)
    """
    client = get_openai_client()

    prompt = VERIFICATION_AGENT_PROMPT.format(
        person_info=person_info,
        assistant_message=assistant_message
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise fact-checking agent. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )

    response_text = response.choices[0].message.content

    # Parse JSON from response
    try:
        # Clean up response if needed
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())
        return result
    except json.JSONDecodeError:
        return {
            "truth": "unknown",
            "description": f"Could not parse verification response: {response_text}"
        }


# ============================================
# STREAMLIT APPLICATION
# ============================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "verification_results" not in st.session_state:
        st.session_state.verification_results = []
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "person_info" not in st.session_state:
        st.session_state.person_info = ""
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False


def add_log_entry(log_type: str, data: dict):
    """Add an entry to the logs."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": log_type,
        "data": data
    }
    st.session_state.logs.append(entry)


def get_all_logs_json() -> str:
    """Get all logs as a JSON string for download."""
    log_data = {
        "export_timestamp": datetime.now().isoformat(),
        "person_info_file": st.session_state.selected_file if st.session_state.selected_file else "Not selected",
        "model": MODEL,
        "conversation": st.session_state.messages,
        "verification_results": st.session_state.verification_results,
        "detailed_logs": st.session_state.logs
    }
    return json.dumps(log_data, ensure_ascii=False, indent=2)


def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Journalist Interview",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for light theme with dark readable text
    st.markdown("""
    <style>
        /* Main container styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            color: #1a1a2e;
            font-size: 1rem;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .user-message {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
            border-left: 4px solid #4a90a4;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #e8f4f8 0%, #d4e8f0 100%);
            border-left: 4px solid #2d6a7a;
        }
        
        /* Header styling */
        .main-header {
            color: #1a1a2e;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 1rem;
            border-bottom: 3px solid #2d6a7a;
        }
        
        .sub-header {
            color: #2d4a5a;
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Verification panel styling */
        .verification-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        
        .verification-true {
            border-left: 4px solid #28a745;
        }
        
        .verification-false {
            border-left: 4px solid #dc3545;
        }
        
        .verification-unknown {
            border-left: 4px solid #ffc107;
        }
        
        .verification-title {
            color: #1a1a2e;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        
        .verification-status {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .verification-desc {
            color: #4a4a6a;
            font-size: 0.85rem;
            line-height: 1.5;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #1a1a2e;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #2d6a7a 0%, #4a90a4 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #1d5a6a 0%, #3a8094 100%);
            box-shadow: 0 4px 12px rgba(45, 106, 122, 0.3);
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            color: #1a1a2e;
            background: #ffffff;
            border: 2px solid #d0d8e0;
            border-radius: 8px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4a90a4;
            box-shadow: 0 0 0 2px rgba(74, 144, 164, 0.2);
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            color: #1a1a2e;
            background: #ffffff;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            color: #1a1a2e;
            background: #f0f4f8;
            border-radius: 8px;
        }
        
        /* Status indicator */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-true { background: #28a745; }
        .status-false { background: #dc3545; }
        .status-unknown { background: #ffc107; }
    </style>
    """, unsafe_allow_html=True)

    # Try to get API key from Streamlit secrets (for Streamlit Cloud) or use environment variable
    global OPENAI_API_KEY
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
    except:
        pass  # If secrets not available, use environment variable

    # Check API key
    if not OPENAI_API_KEY:
        st.error(
            "⚠️ API key is not configured. Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
        st.info("For local setup, create a `.streamlit/secrets.toml` file with the following content:\n```toml\nOPENAI_API_KEY = \"your-api-key-here\"\n```\n\nOr set an environment variable:\n```bash\nexport OPENAI_API_KEY=\"your-api-key-here\"\n```")
        st.stop()

    # Initialize session state
    init_session_state()

    # Layout: Main chat area and sidebar for verification
    col_chat, col_verify = st.columns([2, 1])

    with col_chat:
        st.markdown(
            '<div class="main-header">💬 Conversation with Peter</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Вйо до розмови</div>', unsafe_allow_html=True)

        # File selection (only show before conversation starts)
        if not st.session_state.conversation_started:
            st.markdown("### 📁 Select Person Information File")

            # Determine default index
            default_index = 0
            if st.session_state.selected_file and st.session_state.selected_file in AVAILABLE_PERSON_FILES:
                default_index = AVAILABLE_PERSON_FILES.index(
                    st.session_state.selected_file)

            selected_file = st.selectbox(
                "Choose a person to interview:",
                options=AVAILABLE_PERSON_FILES,
                index=default_index,
                key="file_selector"
            )

            # Update selected file and load person info if changed or not loaded yet
            if selected_file != st.session_state.selected_file or not st.session_state.person_info:
                st.session_state.selected_file = selected_file
                file_path = os.path.join(
                    os.path.dirname(__file__), selected_file)
                st.session_state.person_info = load_person_info(file_path)

        # Check if person info is loaded
        if "ERROR" in st.session_state.person_info:
            st.error(st.session_state.person_info)
            if st.session_state.selected_file:
                st.info(
                    f"Please check the file: {st.session_state.selected_file}")
        else:
            # Start conversation button
            if not st.session_state.conversation_started:
                if st.button("🎬 Start Interview", use_container_width=True):
                    with st.spinner("Peter bereitet sich vor..."):
                        # Prepare input for journalist agent
                        conversation_history_input = []
                        formatted_history = format_conversation_history(
                            conversation_history_input)
                        journalist_prompt = JOURNALIST_ASSISTANT_PROMPT.format(
                            person_info=st.session_state.person_info,
                            conversation_history=formatted_history
                        )

                        # Log journalist agent input
                        add_log_entry("journalist_agent_input", {
                            "person_info": st.session_state.person_info,
                            "conversation_history": conversation_history_input,
                            "formatted_history": formatted_history,
                            "full_prompt": journalist_prompt
                        })

                        # Get initial message from Peter
                        initial_response = call_journalist_agent(
                            st.session_state.person_info,
                            conversation_history_input
                        )

                        # Log journalist agent output
                        add_log_entry("journalist_agent_output", {
                            "response": initial_response
                        })

                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": initial_response
                        })

                        # Prepare input for verification agent
                        verification_prompt = VERIFICATION_AGENT_PROMPT.format(
                            person_info=st.session_state.person_info,
                            assistant_message=initial_response
                        )

                        # Log verification agent input
                        add_log_entry("verification_agent_input", {
                            "person_info": st.session_state.person_info,
                            "assistant_message": initial_response,
                            "full_prompt": verification_prompt
                        })

                        # Verify the response
                        verification = call_verification_agent(
                            st.session_state.person_info,
                            initial_response
                        )

                        # Log verification agent output
                        add_log_entry("verification_agent_output", {
                            "verification_result": verification
                        })

                        st.session_state.verification_results.append(
                            verification)

                        # Log combined entry for backward compatibility
                        add_log_entry("assistant_message", {
                            "message": initial_response,
                            "verification": verification
                        })

                        st.session_state.conversation_started = True
                        st.rerun()

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        st.markdown(f'''
                        <div class="chat-message user-message">
                            <strong>👤 Sie:</strong><br>{message["content"]}
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="chat-message assistant-message">
                            <strong>💬 Peter:</strong><br>{message["content"]}
                        </div>
                        ''', unsafe_allow_html=True)

            # User input
            if st.session_state.conversation_started:
                user_input = st.chat_input("Ihre Antwort eingeben...")

                if user_input:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input
                    })

                    add_log_entry("user_message", {"message": user_input})

                    # Get Peter response
                    with st.spinner("Peter denkt nach..."):
                        # Prepare input for journalist agent
                        conversation_history_input = st.session_state.messages.copy()
                        formatted_history = format_conversation_history(
                            conversation_history_input)
                        journalist_prompt = JOURNALIST_ASSISTANT_PROMPT.format(
                            person_info=st.session_state.person_info,
                            conversation_history=formatted_history
                        )

                        # Log journalist agent input
                        add_log_entry("journalist_agent_input", {
                            "person_info": st.session_state.person_info,
                            "conversation_history": conversation_history_input,
                            "formatted_history": formatted_history,
                            "full_prompt": journalist_prompt
                        })

                        response = call_journalist_agent(
                            st.session_state.person_info,
                            conversation_history_input
                        )

                        # Log journalist agent output
                        add_log_entry("journalist_agent_output", {
                            "response": response
                        })

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

                        # Prepare input for verification agent
                        verification_prompt = VERIFICATION_AGENT_PROMPT.format(
                            person_info=st.session_state.person_info,
                            assistant_message=response
                        )

                        # Log verification agent input
                        add_log_entry("verification_agent_input", {
                            "person_info": st.session_state.person_info,
                            "assistant_message": response,
                            "full_prompt": verification_prompt
                        })

                        # Verify the response
                        verification = call_verification_agent(
                            st.session_state.person_info,
                            response
                        )

                        # Log verification agent output
                        add_log_entry("verification_agent_output", {
                            "verification_result": verification
                        })

                        st.session_state.verification_results.append(
                            verification)

                        # Log combined entry for backward compatibility
                        add_log_entry("assistant_message", {
                            "message": response,
                            "verification": verification
                        })

                    st.rerun()

    with col_verify:
        st.markdown("### 🔍 Fact Verification")
        st.markdown("---")

        if st.session_state.verification_results:
            for i, result in enumerate(reversed(st.session_state.verification_results)):
                idx = len(st.session_state.verification_results) - i
                truth_value = str(result.get("truth", "unknown")).lower()

                if truth_value == "true":
                    status_class = "verification-true"
                    status_icon = "✅"
                    status_text = "VERIFIED"
                elif truth_value == "false":
                    status_class = "verification-false"
                    status_icon = "❌"
                    status_text = "UNVERIFIED"
                else:
                    status_class = "verification-unknown"
                    status_icon = "⚠️"
                    status_text = "UNKNOWN"

                st.markdown(f'''
                <div class="verification-card {status_class}">
                    <div class="verification-title">Message #{idx}</div>
                    <div class="verification-status">{status_icon} {status_text}</div>
                    <div class="verification-desc">{result.get("description", "No description")}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info(
                "No messages to verify yet. Start the interview to see verification results.")

        st.markdown("---")

        # Download logs button
        if st.session_state.logs:
            logs_json = get_all_logs_json()
            st.download_button(
                label="📥 Download All Logs",
                data=logs_json,
                file_name=f"interview_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        # Show person info in expander
        with st.expander("📋 Person Information"):
            st.text(st.session_state.person_info[:500] + "..." if len(
                st.session_state.person_info) > 500 else st.session_state.person_info)

        # Reset conversation button
        if st.session_state.conversation_started:
            if st.button("🔄 Reset Interview", use_container_width=True):
                st.session_state.messages = []
                st.session_state.verification_results = []
                st.session_state.logs = []
                st.session_state.conversation_started = False
                # Keep selected_file and person_info when resetting
                st.rerun()


if __name__ == "__main__":
    main()
