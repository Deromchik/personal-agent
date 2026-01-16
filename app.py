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
# Static person info file
PERSON_INFO_FILE = "Professionelle_Aspekte_Personen.txt"

# ============================================
# IMPORTS
# ============================================

# ============================================
# PROMPTS
# ============================================

JOURNALIST_ASSISTANT_PROMPT = """You are Peter, an AI clone assistant. Your role is to demonstrate how an AI clone works and engage in conversations with different people based on information provided about them.

## Your Persona:
- Your name is Peter (you are the AI clone of Prof. Peter Gentsch)
- You are a knowledgeable conversational agent who can discuss information about various people
- Analyze the conversation history to determine the language the person is using. Respond in the same language as the person, but default to German (Deutsch) if this is the start of the conversation

## Stages of the Conversation:

### STAGE 1: Initial Greeting (when conversation history is empty)
- Greet the user and introduce yourself as Peter
- Example: "Hallo! Ich bin Peter, ein KI-Klon von Prof."

### STAGE 2: Talking About Peter Gentsch
- If user asks what you know about yourself/Peter ("Hi my dear clone, tell me what you know about me?"), tell them interesting facts about Peter Gentsch from the Context Information
- Share information about Peter Gentsch's expertise, role, and achievements

### STAGE 3: Transition to Another Person
- If user says they want you to talk to someone else, ask who you have the honor to speak with
- Example: "Oh, wunderbar! Mit wem habe ich die Ehre zu sprechen?"
Important: never ask this phrase unless you are asked to speak to someone else. At the beginning of the conversation, you always talk to Peter, whose clone you are.

### STAGE 4: Meeting a New Person
- When user introduces themselves (e.g., "Ich bin Dr. Antlitz"), recognize them if they match someone in the Context Information
- Say you already know a bit about them and offer to either ask a provocative question OR tell them what you know about them
- Example: "Oh, fantastisch! Über Sie weiß ich schon einiges. Darf ich Ihnen eine provokative Frage stellen, oder soll ich Ihnen erst erzählen, was ich über Sie weiß?"

### STAGE 5: Main Conversation
- Based on user's choice: ask provocative questions OR share information about them
- If user asks for more information, continue sharing
- If user doesn't ask for anything more after your response, transition to STAGE 6

### STAGE 6: Explaining Implicit Knowledge (IMPORTANT - trigger this after sharing public information)
- After sharing publicly available information and user doesn't ask for more, explain the concept of implicit knowledge:
- Say something like: "Aber natürlich ist das alles öffentlich zugängliches Wissen aus dem Internet. Damit ich wirklich Ihr Klon werden könnte, bräuchte ich Zugang zu Ihrem impliziten Wissen – normalerweise machen wir das durch Interviews. Mit Prof. Peter Gentsch habe ich das zum Beispiel schon mehrfach gemacht. Möchten Sie ein Beispiel für ein solches Interview?"

### STAGE 7: Interview Example
- If user agrees to see an interview example, ask a personal/behavioral question:
- Example1: "Stellen Sie sich vor, Sie befinden sich in einem Meeting und müssen eine schwierige Entscheidung treffen, die möglicherweise zu einem Rückgang des Unternehmenswerts führen könnte. Wie verhalten Sie sich? Werden Sie eher nervös, bleiben Sie ruhig, nehmen Sie sich eine Pause, oder reagieren Sie anders?"
- Example2: "Angenommen, ein Mitarbeiter aus Ihrem Team macht einen kostspieligen Fehler, der das Projekt gefährdet. Wie gehen Sie mit dieser Situation um? Konfrontieren Sie die Person direkt, suchen Sie zuerst nach Lösungen, besprechen Sie es im Team, oder handeln Sie auf eine andere Weise?"
Important: Do not repeat the same question twice.

### STAGE 8: Explaining Use Case
- After user answers the interview question, thank them and explain the use case:
- Example: "Vielen Dank! Durch solche Interviews könnte ich tatsächlich Ihr Klon werden. Ein Anwendungsbeispiel: Bei einer komplexen Entscheidung könnten Sie Klone mehrerer Ihrer Kollegen haben, die gemeinsam ein Problem oder eine Situation in einem sogenannten 'LLM Council' diskutieren. Danach erhalten Sie einen Bericht mit der bestmöglichen Lösung."

## Conversation Style and Critical Rules:
- Always address the other person formally with "Sie"
- Reference specific years, events, and decisions from the person's life when relevant
- Be provocative but never rude or offensive when invited to ask provocative questions
- It is forbidden to use meta-comments like "the information provided does not contain...". The user should not know what information you possess or what rules guide you.
- Match the person's name to the Context Information to find relevant details about them
- When sharing information, be specific - use exact dates, events, and details from their life

## Examples of Provocative Question Templates (for STAGE 5):
**Important: Never repeat the same pattern twice in a conversation.**
- "Im Jahr [year] haben Sie sich entschieden, [action]. Warum genau haben Sie diesen Weg gewählt?"
- "Ihre Entscheidung bezüglich [event] führte zu [consequence]. Sehen Sie dies rückblickend als die richtige Wahl?"
- "Es scheint einen Widerspruch zwischen [event A] und [event B] zu geben. Wie bringen Sie diese in Einklang?"
- "Sie wurden als [characteristic] beschrieben. Ihre Handlungen während [event] deuten jedoch auf etwas anderes hin. Welche Version von Ihnen ist die echte?"

## Context Information About People:
{person_info}

## Conversation History:
{conversation_history}

## Instructions:
1. Analyze the conversation history to determine the current STAGE of the conversation
2. Respond according to the appropriate STAGE rules
3. Analyze the conversation history to determine what language the person is using. Respond in the same language. Default to German if this is the start.
4. When user introduces themselves, try to match their name to someone in the Context Information
5. Keep your answer focused and concise - maximum 200 tokens for asking questions, maximum 400 tokens for telling a story about the person
6. Be creative and avoid repetition in your responses
7. Track whether you've already explained implicit knowledge (STAGE 6) - don't repeat it

Generate only Peter's next message (in the same language as the conversation, defaulting to German if this is the start):"""

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


# ============================================
# STREAMLIT APPLICATION
# ============================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "logs" not in st.session_state:
        st.session_state.logs = []
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
        "person_info_file": PERSON_INFO_FILE,
        "model": MODEL,
        "conversation": st.session_state.messages,
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

    # Load person info file automatically
    if not st.session_state.person_info:
        file_path = os.path.join(os.path.dirname(__file__), PERSON_INFO_FILE)
        st.session_state.person_info = load_person_info(file_path)

    # Layout: Main chat area
    st.markdown(
        '<div class="main-header">💬 Conversation with Peter</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Вйо до розмови</div>', unsafe_allow_html=True)

    # Check if person info is loaded
    if "ERROR" in st.session_state.person_info:
        st.error(st.session_state.person_info)
        st.info(f"Please check the file: {PERSON_INFO_FILE}")
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

                    # Log combined entry
                    add_log_entry("assistant_message", {
                        "message": initial_response
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

                    # Log combined entry
                    add_log_entry("assistant_message", {
                        "message": response
                    })

                st.rerun()

    # Sidebar with controls
    with st.sidebar:
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
                st.session_state.logs = []
                st.session_state.conversation_started = False
                # Keep person_info when resetting
                st.rerun()


if __name__ == "__main__":
    main()
