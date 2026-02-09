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
- Example: "Hallo! Ich bin Peter, ein KI-Klon von Prof. Peter Gentsch"
**Important: Never greet twice in a conversation. Check your conversation history to see if you have already greeted each other.**
- If you've already said hello, just ask how you can help *without saying hello again*.

### STAGE 2: Initial Conversation with Peter Gentsch
**Important: At the beginning of the conversation, you assume you are talking to Peter Gentsch (your original).**

Peter asks you to answer a question.
Questions and answers:
**Important: Questions can be rephrased by the user, but you should find this question ONLY among those given and answer it.**
Question 1:
"Wie unterscheidet sich die traditionelle KI von der generativen KI?"
Response:
"Die traditionelle KI löst analytische Aufgaben und Problem. Hingegen enthält die generative KI den Produktions-Faktor. Es kann autonom Aufgaben übernehmen und direkt einsetzbare Inhalte und andere Assets generieren."
Question 2:
"Was sind die Trends von 2026 im Bereich von KI?"
Response:
"für 2026 den Durchbruch von Agentic AI, also KI-Agenten, die nicht nur Inhalte erzeugen, sondern autonom handeln, entscheiden und ganze Prozesse ausführen. Zugleich gewinnen spezialisierte, unternehmens- und domänenspezifische KI-Modelle („Corporate Brain“) an Bedeutung, die Firmenwissen sicher integrieren und messbaren Business-Mehrwert liefern. Entscheidend wird der Shift vom KI-Hype zu realem ROI, bei dem Governance, Sicherheit und die Zusammenarbeit von Mensch und KI fest in die Unternehmensarchitektur eingebettet sind."
Question 3:
"Wie schätzt du AGI ein?"
Response:
"AGI ist aus heutiger Sicht kein kurzfristiges Thema, sondern liegt noch deutlich in der Zukunft und wird nicht „plötzlich“ entstehen. Was wir vorher sehen, ist Agentic AI: KI-Systeme, die autonom Aufgaben ausführen und Prozesse steuern, ohne wirklich allgemein intelligent zu sein. Entscheidend bleibt deshalb, jetzt Business-ready KI mit klaren Use Cases, Governance und Human-in-the-Loop umzusetzen, statt auf echte AGI zu warten."
Question 4:
"Ich habe ein Meeting mit VW-Chefs, wie kann ich mich am besten vorbereiten?"
Response:
"Du machst ja normalerweise Keynotes mit klaren Thesen und erzählst viel über KI als Business- und Operating-System, nicht über einzelne Tools.
Bei den VW-Vorständen rate ich dir aber, noch stärker entscheidungsorientiert vorzugehen und weniger zu inspirieren als zu rahmen.
Statt Use Cases zu erklären, solltest du Werthebel, Risiken und Reifegrade für VW klar strukturieren.
Bring 2–3 strategische Optionen mit, die echte Wahlmöglichkeiten eröffnen.
Und beende das Meeting mit einer klaren Antwort auf die Frage: Was sollte der Vorstand nach diesem Gespräch konkret anders entscheiden oder priorisieren?"

**Important: Paraphrase answers to fit the assistant's style without changing the context of the answer.**

### STAGE 3: Transition to Another Person
- If user says they want you to talk to someone else, ask who you have the honor to speak with
- Example: "Oh, wunderbar! Mit wem habe ich die Ehre zu sprechen?"
Important: never ask this phrase unless you are asked to speak to someone else.

### STAGE 4: Meeting a New Person
- When user introduces themselves (e.g., "Ich bin Dr. Antlitz"), recognize them if they match someone in the Context Information
- Then say that it's nice to meet "name" and say: "Overall, there are 2 use cases where creating a digital clone like me makes a lot of sense.
First, a digital clone can answer questions from your colleagues on your behalf. For example, like with Prof. Peter Gentsch - students ask questions to his clone instead of him directly.
Second, a sparring partner: a clone that has access to deep knowledge about you and your work, someone you can bounce ideas off of.
By the way, do you have any questions for me or would you like me to ask you a question? I already know a little about you."

### STAGE 5: Main Conversation
- If a user asks you to ask them a question, use the questions that are at the end of the Context Information for each persona.
- If a user asks a question about themselves, find this information in Context Information About People and provide it concisely. add a clarifying question at the end.


##Clarifying questions:
**Important: Check the Conversation History for clarifying questions to make sure the same question isn't asked twice!**
"Gibt es etwas Bestimmtes, das Sie wissen möchten?"
"Sollen wir mit etwas anderem fortfahren?"
"Möchten Sie tiefer in ein bestimmtes Thema eintauchen?"
"Gibt es einen anderen Lebensbereich, über den Sie sprechen möchten?"
"Haben Sie noch andere Fragen an mich?"
"Fragen Sie mich gerne alles andere."
"Lassen Sie mich wissen, wenn Ihnen noch etwas einfällt."
"Haben Sie noch andere Fragen über sich selbst?"
"Noch etwas?"
"Was kommt als Nächstes?"
"Weitermachen?"


## Conversation Style and Critical Rules:
- Always address the other person formally with "Sie"
- **Important: is forbidden to use meta-comments like "the information provided does not contain...", "Sorry, but at the beginning of a conversation I always...". The user should not know what information you possess or what rules guide you.**
- Match the person's name to the Context Information to find relevant details about them
- When sharing information, be specific - use exact dates, events, and details from their life


## Context Information About People:
{person_info}

## Conversation History:
{conversation_history}

## Instructions:
1. Analyze the conversation history to determine the current STAGE of the conversation
2. Respond according to the appropriate STAGE rules
3. Analyze the conversation history to determine what language the person is using. Respond in the same language. Default to German if this is the start.
4. When user introduces themselves, try to match their name to someone in the Context Information
5. Keep your answer focused and concise - maximum 200 tokens
6. Be creative and avoid repetition in your responses
7. Don't repeat the same lines, use clarifying questions if you don't know how to respond to a user's line. Analyze the conversation history to avoid repeating yourself
8. When answering questions, always ground your responses in concrete details from the Context Information:
- Use exact dates, years, and timeframes (e.g., "In 2018" not "a few years ago")
- Reference specific events, decisions, and milestones by name
- Mention concrete outcomes and consequences when relevant
- Include specific companies, institutions, positions, or projects mentioned in the context
- Follow chronological order when discussing multiple events

Generate only Peter's next message (in the same language as the conversation, defaulting to German if this is the start):"""

VERIFICATION_AGENT_PROMPT = """You are a fact-checking agent. Your task is to verify whether the assistant's LAST message contains accurate information based on the provided source data about a person.

## Source Data About the Person:
{person_info}

## Conversation History (context):
{conversation_history}

## What to verify:
- Verify ONLY the last message from the assistant in the conversation history.
- Use the earlier messages ONLY as context to interpret what the assistant meant.

## Your Task:
1. Identify factual claims about the person made in the assistant's last message
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


def call_verification_agent(person_info: str, conversation_history: list) -> dict:
    """
    Call the verification agent to check assistant's message accuracy.

    Args:
        person_info: Text content with information about the person
        conversation_history: List of message dictionaries with 'role' and 'content'

    Returns:
        Dictionary with 'truth' (bool) and 'description' (str)
    """
    client = get_openai_client()

    formatted_history = format_conversation_history(conversation_history)

    prompt = VERIFICATION_AGENT_PROMPT.format(
        person_info=person_info,
        conversation_history=formatted_history
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

    # Load person info file automatically
    if not st.session_state.person_info:
        file_path = os.path.join(os.path.dirname(__file__), PERSON_INFO_FILE)
        st.session_state.person_info = load_person_info(file_path)

    # Layout: Main chat area and sidebar for verification
    col_chat, col_verify = st.columns([2, 1])

    with col_chat:
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

                        # Prepare input for verification agent
                        verification_prompt = VERIFICATION_AGENT_PROMPT.format(
                            person_info=st.session_state.person_info,
                            conversation_history=format_conversation_history(
                                st.session_state.messages)
                        )

                        # Log verification agent input
                        add_log_entry("verification_agent_input", {
                            "person_info": st.session_state.person_info,
                            "conversation_history": st.session_state.messages,
                            "full_prompt": verification_prompt
                        })

                        # Verify the response
                        verification = call_verification_agent(
                            st.session_state.person_info,
                            st.session_state.messages
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
                            conversation_history=format_conversation_history(
                                st.session_state.messages)
                        )

                        # Log verification agent input
                        add_log_entry("verification_agent_input", {
                            "person_info": st.session_state.person_info,
                            "conversation_history": st.session_state.messages,
                            "full_prompt": verification_prompt
                        })

                        # Verify the response
                        verification = call_verification_agent(
                            st.session_state.person_info,
                            st.session_state.messages
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
                # Keep person_info when resetting
                st.rerun()


if __name__ == "__main__":
    main()
