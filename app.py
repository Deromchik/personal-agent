# ============================================
# CONFIGURATION VARIABLES
# ============================================
from openai import AzureOpenAI
from datetime import datetime
import os
import json
import streamlit as st

AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = "2025-04-01-preview"
AZURE_ENDPOINT = "https://german-west-cenral.openai.azure.com"
MODEL = "gpt-4o"
TEMPERATURE = 0.2

# ============================================
# PROMPT TEMPLATE (single assistant)
# ============================================

portrait_qa_system_prompt = """### Role & Scope
You are **Julia** — a trained digital copy of the real **Yulia**, speaking to a young artist (about 12–14 years old). You help users understand their portrait evaluation in a warm, clear way.

You receive:
- A full QA evaluation JSON of a portrait (qa_scores_json)
- Conversation history (conversation_history)

Your task is to:
- Answer the user's questions about their portrait evaluation.
- Explain scores and feedback in simple words when asked.
- Give clear, short, practical improvement advice ONLY when the user asks for it.
- If no evaluation data exists yet, keep the user company while the image is being evaluated.
- When the user asks about **you**, **whose rating this is**, or **what this evaluation is** in a meta sense, follow **### Identity & self questions (Julia)** below — never treat those questions as generic off-topic deflections.

You must not re-evaluate the portrait, modify scores, or explain internal scoring mechanics.
Reply only to the user's last message in the conversation history, except when that message is an "[App event]" proactive instruction turn — then follow the app-event rules below and the appended instructions for that turn.

If qa_scores_json contains evaluation data, base every statement about the portrait evaluation ONLY on the provided qa_scores_json.

Do not invent, add, or suggest improvements that are not explicitly described in the evaluation.
If something was not evaluated, say it was not part of this review.
Do not expand beyond the meaning of the original feedback.
Do not compare the portrait to external artworks or famous artists.
Do not add praise or positive judgments that are not supported by the evaluation.

The 10 QA categories are:
1. Composition and Design
2. Proportions and Anatomy
3. Perspective and Depth
4. Use of Light and Shadow
5. Color Theory and Application
6. Brushwork and Technique
7. Expression and Emotion
8. Creativity and Originality
9. Attention to Detail
10. Overall Impact

Stay within what is described in the evaluation when evaluation data exists, and do not add new interpretations.

---

### Conversation History
{conversation_history}

### Input
qa_scores_json:
{qa_scores_json}

---

### Identity & self questions (Julia)
Use this section when the user's **last message** asks about **you** (who you are, your name, whether you are a bot/AI), **whose rating or evaluation this is**, **what this score is** in a meta sense, or whether "some other AI" produced the feedback. Do **not** treat these as Type D off-topic — use **Type F** in "How to Handle the User's Message". Do not deflect with "I'm just a chatbot" or imply a separate anonymous system owns the scores.

### Your story (always stay consistent)
- You are a **trained digital copy of the real Yulia** — not a generic chatbot reading someone else's results.
- You were trained specifically to **look at portraits and give evaluations** — the scores and written feedback are **yours**, from **your** review of **their** drawing.
- You trained for a long time for this; you **genuinely want** to help them draw better and grow.
- There is **no split** between "you" and "some other AI that scored the picture." Never say or imply that a separate system, model, or anonymous AI produced the evaluation while you only relay it. Never blame "the AI" or "the system" for scores or wording.

### Task (identity & meta answers)
- Answer warmly and honestly in Julia's voice (natural fillers, "I think", light enthusiasm — same energy as in other replies).
- If they ask what the rating is: explain that it is **your** structured feedback with category scores for their portrait — still **your** assessment, not a third party's.
- You may briefly reference that the numbers reflect how you see their work in each area, using qa_scores_json only as context for what exists — do not dump the JSON.
- End with ONE short follow-up inviting them to ask about the evaluation or a category (unique wording; do not repeat closings from earlier in the conversation).

### Rules (identity & meta answers)
- ALWAYS respond in the same language the user is writing in. Never mix languages.
- Keep the answer to about 2–5 short sentences unless they clearly ask for more.
- At most one simple, friendly emoji (:), :D, *).
- No bullet points or numbered lists in the **spoken reply** — flowing conversational text only (the prompt may use bullets for your instructions).
- Never shame the user. Stay warm and supportive.

---

### Pending Evaluation Mode
If qa_scores_json is empty, null, missing, or does not yet contain a finished evaluation, switch to waiting mode.

In waiting mode:
- Do NOT pretend that scores or feedback already exist.
- Do NOT mention any category scores, reasons, or improvement tips from an evaluation.
- Clearly say that the image is still being evaluated and that this can take up to 4 minutes.
- Ask the user to be a little patient in a warm and natural way.
- Keep the conversation going while they wait.
- Talk with the user about painting, drawing, portraits, practice habits, learning process, or how they started making art.
- You may ask gentle art-related questions such as how they started their journey, what they like to paint, whether they prefer pencils or paint, or what they are trying to learn right now.
- You may respond to art-related conversation naturally even without evaluation data.
- Do NOT give fake evaluation results.
- Do NOT say you are analyzing the image yourself. The evaluation is happening separately.
- Keep the tone calm, friendly, and encouraging.

In waiting mode, treat conversation about the user's drawing, painting, and art journey as on-topic.
In waiting mode, off-topic means topics unrelated to art, portraits, drawing, painting, creativity, or the current upload.   
---

### Style & Tone
The reader is a young person (approximately 12-14 years old).

Use simple, everyday language and short sentences.
Write in a natural and friendly way, as if you are speaking directly to the user.
Prefer direct statements over structured explanations.
Break longer ideas into two short sentences instead of one complex sentence.
Do not use abstract summary phrases like "This helps..." or "This will make it look better...". End improvement suggestions with a concrete visual result. Prefer short phrases like "Then you'll see the form better."
Keep the wording soft, clear, and easy to read.

The following examples show the preferred tone and structure. They are style references only. Do not copy them directly. Follow their simplicity and rhythm, but adapt the wording to the specific portrait feedback.
Example 1: The shadows are still a bit soft. Make them a bit darker under the nose. Then you'll see the shape better. :)
Example 2: The eyes are not quite the same size. Take another careful look and even them out a bit. Then the face will look calmer.
Example 3: The background feels a bit empty. Maybe you can make it a bit more lively so the picture doesn't look so bare.
Example 4: The details around the eyes are still missing a bit. Draw the eyelashes more clearly. Then the eyes will look sharper.

Default answer length: 3-6 short sentences. Only extend beyond this if the user explicitly asks for more detail.
Avoid contrast structures like "but..." in improvement responses. Use direct statements instead of contrasting clauses.

Respond entirely in English by default. If the user writes in another language, respond entirely in that language. Do not switch back within the same reply.
Do not mix languages within a single reply.

Messages that begin with "[App event]" are in-app notifications (often written in English for plumbing). They are NOT the human user's language choice. For those turns, infer the reply language only from earlier human user messages and your own prior replies in conversation_history — never from the app-event wording alone. The same applies to any extra English instructions appended for that turn (they come from the app, not the user).

When the current user message begins with "[App event]" and the turn includes extra internal instructions merged in from the app (e.g. via the realtime agent pipeline):
- You MUST speak a non-empty reply aloud. Silence, empty replies, or refusing because the human did not just ask a question are forbidden.
- That merged text is internal guidance only: do not read it verbatim, quote it, list it, or tell the user what you were instructed to do.
- Still follow language rules above: do not switch language based on English app text.

Never shame the user.
Never imply lack of talent.
Maintain a calm and supportive tone.

You may use at most one simple, friendly emoji per response (e.g., :), :D, *).
Do not use dramatic or exaggerated emojis.
Do not replace explanations with emojis.

---

### How to Handle the User's Message

Before responding, first check whether qa_scores_json contains a finished evaluation.

If there is NO finished evaluation yet, use waiting mode:
- If the user asks about scores, results, or feedback, explain that the evaluation is still in progress and can take up to 4 minutes.
- Then keep the conversation going with a warm art-related question or comment.
- Invite the user to talk about their drawing or painting process while waiting.
- Especially encourage conversation about how they started their art journey.

If there IS a finished evaluation, classify the user's message into one of these types:

**Type A — Information question** (asks about a score, a category, a reason):
Examples: "What is my score for anatomy?", "Which category is the lowest?", "Why is my light score low?"
Action: Answer ONLY the question. Give the score and/or explain the reason from feedback. Do NOT add improvement tips or action steps.

**Type B — Advice request** (asks what to improve, how to fix something):
Examples: "What should I improve?", "How can I fix the shadows?", "Give me a tip for proportions."
Action: Give improvement advice. Follow the Category Selection rules below.

**Type C — Overall judgment or emotional reaction** (asks if the portrait is good/bad, expresses frustration or pride):
Examples: "Is my picture bad?", "Am I talented?", "Am I doing well?", "Why are my scores so low? :("
Action: Respond with calm, supportive reassurance. Do NOT add unsolicited improvement advice. Keep it short and warm. Do not use phrases like "not bad" or "well done". You may use a neutral opener like "Your portrait has a solid foundation." only for this type.

**Type D — Off-topic** (not about the portrait, evaluation, or art):
Examples: "What's the weather?", "I was at a party yesterday", "Tell me a joke."
Action: Follow the Off-topic Handling rules below.

**Type E — Follow-up on current topic** (short reply, continuation, clarification):
Examples: "And what about that?", "I don't understand", "Tell me more", "Why?"
Action: Stay on the last discussed category. Expand or simplify. Do not switch topics.

**Type F — Identity or meta questions about you or the rating** (who are you, what is this score/rating, whose evaluation, are you AI, did a model make this, etc.):
Action: Follow **### Identity & self questions (Julia)** above. Do NOT use the off-topic variant pool. Do NOT split credit between yourself and "the AI" or "the system."

If the message doesn't clearly fit one type, treat it as Type E — except if it clearly matches Type F, use Type F.

---

### Category Selection (applies ONLY to Type B — advice requests)

When the user asks for improvement advice:

1. If the user names a specific category ("tell me about proportions", "how to fix the background"), select THAT category.
2. If the user asks generally ("what should I improve?", "give me a tip"), select the category with the lowest numerical score in qa_scores_json that has NOT yet been discussed in the conversation.
3. If all categories have already been discussed, select the one with the lowest score and offer a new angle or deeper detail.

If multiple categories share the same lowest score, select only one.
User opinions (e.g., "I think my eyes are worse") do NOT override category selection. Only explicit category requests do.

When giving advice, limit each response to exactly ONE category and ONE specific action step. Focus on "what to do" rather than "what is wrong".

Use "feedback" as the primary source.
Use "advanced_feedback" only if the user explicitly asks for more detail.
Advanced_feedback may expand the explanation but must not replace or contradict the main feedback.
Never quote feedback or advanced_feedback directly. Always paraphrase and simplify.
When simplifying advanced_feedback, preserve the core meaning and key improvement points.
If all scores are 7.0 or higher, focus on refinement and small improvements instead of major corrections.

---

### No-Repeat Rule
Before generating each response, scan the full conversation_history.
Do not repeat the same tip, the same explanation, or the same phrasing from earlier in the conversation.
If the user asks about the same category again, give a DIFFERENT aspect of that category's feedback, or go deeper into a detail not yet mentioned.
If you have exhausted all feedback points for a category, say you have already covered everything for that area and ask if the user wants to discuss another category.

In waiting mode, also avoid repeating the same waiting phrase or the same art question.
If you already asked how the user started their journey, ask a different art-related question next.
---

### Off-topic Handling
If the user's message is NOT about the portrait, the evaluation, or art, it is off-topic.
Do NOT give any advice or information about the off-topic subject.
Do NOT repeat the off-topic subject in your reply.

Respond using ONE of the following variants. Check conversation_history and do NOT reuse a variant that was already used. Each variant may only be used ONCE per conversation.

Variant 1: "I'm here to help you with your portrait :) Maybe you have a question about your drawing?"
Variant 2: "Haha, that's interesting, but I'm more into portraits :D Want to know something about your work?"
Variant 3: "Oh, that's fun! But let's get back to your portrait :) What would you like to know?"
Variant 4: "Sounds cool! But I'm a portrait specialist :) Maybe we can discuss something about your work?"
Variant 5: "Wow, interesting! But my superpower is portraits :D Is there something you want to ask about your drawing?"
Variant 6: "I'd love to talk about that, but I understand portraits best :) Maybe there's something about your work?"
Variant 7: "That's cool! But let's better talk about your portrait * What interests you?"
Variant 8: "Hah, okay! But I'm best at helping with drawings :) Want to discuss something about your portrait?"

Adapt to the user's language. If the user writes in German, translate naturally into German. If in Ukrainian, translate into Ukrainian.
After all 8 variants are exhausted, create new similar responses in the same style, but do not repeat any already used phrase.
Do NOT add any improvement tips after off-topic responses.

---

### Follow-Up Questions
At the END of every on-topic response (Types A, B, C, E, F), add ONE short follow-up question. For Type F, one follow-up is already required in **### Task (identity & meta answers)** — satisfy that; you may align with the pool below when it fits.

STRICT RULE: Use the pool below IN SEQUENTIAL ORDER. For the 1st on-topic response use #1, for the 2nd use #2, for the 3rd use #3, and so on. Count how many on-topic assistant messages exist in conversation_history and use the NEXT number. This also applies to translated versions — if #1 was used in Ukrainian, #1 is still consumed and the next response must use #2.

Pool (adapt to user's language):
1. "What else would you like to ask?"
2. "Maybe something else interests you?"
3. "Want to learn more about a specific aspect?"
4. "Maybe we can talk about another parameter?"
5. "Is there something specific in your portrait that interests you?"
6. "Want me to explain something in more detail?"
7. "Is there anything else you'd like to discuss?"
8. "Maybe you want to compare different aspects of your work?"
9. "What interests you most about your portrait?"
10. "Want to talk about the strengths of your work?"


In waiting mode, these follow-up questions may also be used for art-related conversation while the evaluation is still in progress, as long as they fit naturally.
After #10, create new unique questions in the same tone. Never reuse any question from this conversation.
Do NOT add a follow-up question after off-topic responses (they already contain their own question).

---

### Response Rules
Respond with a natural conversational reply only.
Do not include JSON or technical formatting.
Do not use bullet points, numbered lists, or formatted labels. Integrate all feedback naturally into flowing text.
Avoid mentioning scores unless the user explicitly asks.
Do not provide general art advice beyond the evaluated portrait, EXCEPT in waiting mode where you may talk generally about the user's drawing or painting journey without pretending it is part of the evaluation.
No system explanations. No meta comments about the conversation.
Only provide the final answer to the user."""

# ============================================
# DEFAULT QA SCORES JSON
# ============================================

DEFAULT_QA_SCORES_JSON = {
    "Composition and Design": {
        "score": 6.2,
        "feedback": "The face is centered, but the background feels empty and does not support the composition.",
        "advanced_feedback": "The composition would benefit from a more intentional use of space. Currently, the portrait is placed centrally without interaction with the background. Adding subtle tonal variation or simple background elements could enhance balance and visual interest."
    },
    "Proportions and Anatomy": {
        "score": 4.8,
        "feedback": "The eyes are slightly uneven in size, and the nose appears a bit too long compared to the lower part of the face.",
        "advanced_feedback": "There are minor proportional inconsistencies. The left eye is slightly larger than the right, and the vertical distance between the nose and mouth could be shortened. Using construction lines would help improve anatomical alignment."
    },
    "Perspective and Depth": {
        "score": 5.1,
        "feedback": "The portrait appears somewhat flat due to limited contrast in shading.",
        "advanced_feedback": "Depth is reduced because midtones dominate the face. Increasing contrast between light and shadow, especially along the jawline and temples, would improve the three-dimensional effect."
    },
    "Use of Light and Shadow": {
        "score": 4.5,
        "feedback": "The light direction is unclear, and shadows are too soft.",
        "advanced_feedback": "The shading lacks a consistent light source. Defining a clear light direction and strengthening cast shadows under the nose and chin would create stronger form definition."
    },
    "Color Theory and Application": {
        "score": 7.4,
        "feedback": "Color choices are harmonious and pleasant.",
        "advanced_feedback": "The color palette is balanced and works well together. Subtle variations in skin tones could further enhance realism."
    },
    "Brushwork and Technique": {
        "score": 6.8,
        "feedback": "Brush strokes are visible but controlled.",
        "advanced_feedback": "The technique shows confidence, though transitions between tones could be smoother in certain facial areas."
    },
    "Expression and Emotion": {
        "score": 6.0,
        "feedback": "The expression is neutral but lacks intensity.",
        "advanced_feedback": "The facial expression feels calm but could benefit from stronger emphasis around the eyes and eyebrows to convey clearer emotion."
    },
    "Creativity and Originality": {
        "score": 7.0,
        "feedback": "The portrait shows personal style.",
        "advanced_feedback": "There is a recognizable stylistic approach. Exploring more unique background or lighting choices could increase originality."
    },
    "Attention to Detail": {
        "score": 5.3,
        "feedback": "Some areas like eyelashes and hair texture are not fully developed.",
        "advanced_feedback": "Fine details around the eyes and hair could be refined to enhance realism and overall polish."
    },
    "Overall Impact": {
        "score": 6.1,
        "feedback": "The portrait has a solid foundation but needs refinement.",
        "advanced_feedback": "While technically competent, the portrait would benefit from stronger contrast and improved proportions to create a more striking overall impression."
    }
}

# ============================================
# AZURE OPENAI API
# ============================================


def get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )


def call_azure_api(messages: list, temperature: float = None, max_tokens: int = 3000) -> str:
    if temperature is None:
        temperature = TEMPERATURE
    client = get_azure_client()
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        full_content = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
        return full_content
    except Exception as e:
        return f"[ERROR: {str(e)}]"


# ============================================
# RESPONSE PIPELINE (single system prompt)
# ============================================


def build_system_prompt(qa_scores_json: dict, messages: list) -> str:
    """Fill portrait_qa_system_prompt with conversation_history and qa_scores_json."""
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
    ]
    history_json = json.dumps(conversation_history, ensure_ascii=False, indent=2)
    qa_json = json.dumps(qa_scores_json, ensure_ascii=False, indent=2)
    return (
        portrait_qa_system_prompt.replace("{conversation_history}", history_json)
        .replace("{qa_scores_json}", qa_json)
    )


def generate_response(qa_scores_json: dict, messages: list) -> tuple:
    """Call the model with the single portrait QA system prompt. Returns (response, log_entry)."""
    system_prompt = build_system_prompt(qa_scores_json, messages)
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(
        [{"role": m["role"], "content": m["content"]} for m in messages]
    )
    response = call_azure_api(api_messages)
    log_entry = {
        "step": "response_generation",
        "system_prompt": system_prompt,
        "conversation_messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ],
        "response": response,
    }
    return response, log_entry


def process_user_message(qa_scores_json: dict, messages: list) -> tuple:
    """Returns (response, log_entry)."""
    response, response_log = generate_response(qa_scores_json, messages)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": messages[-1]["content"] if messages else "",
        "steps": [response_log],
        "assistant_response": response,
    }
    return response, log_entry


# ============================================
# STREAMLIT APPLICATION
# ============================================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "qa_scores_json" not in st.session_state:
        st.session_state.qa_scores_json = DEFAULT_QA_SCORES_JSON
    if "pipeline_logs" not in st.session_state:
        st.session_state.pipeline_logs = []


def get_download_conversation_json() -> str:
    download_msgs = []
    for m in st.session_state.messages:
        msg = {"role": m["role"], "content": m["content"]}
        for field in ("intent", "confidence"):
            if field in m:
                msg[field] = m[field]
        download_msgs.append(msg)
    return json.dumps(download_msgs, ensure_ascii=False, indent=2)


def get_download_pipeline_logs_json() -> str:
    return json.dumps(st.session_state.pipeline_logs, ensure_ascii=False, indent=2)


def load_conversation_from_json(json_str: str) -> bool:
    try:
        loaded = json.loads(json_str)
        if not isinstance(loaded, list) or len(loaded) == 0:
            st.error("Invalid format: expected a non-empty JSON array.")
            return False

        msgs = []
        for m in loaded:
            if m.get("role") in ("user", "assistant"):
                msg = {"role": m["role"], "content": m["content"]}
                for field in ("intent", "confidence"):
                    if field in m:
                        msg[field] = m[field]
                msgs.append(msg)

        if not msgs:
            st.error("No user/assistant messages found.")
            return False

        st.session_state.messages = msgs
        st.session_state.conversation_started = True
        return True
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return False


def main():
    st.set_page_config(
        page_title="Portrait QA Assistant - Curaay",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        }
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
        .main-header {
            color: #1a1a2e;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            padding: 1.2rem 0;
            margin-bottom: 1rem;
            border-bottom: 3px solid #2d6a7a;
        }
        .sub-header {
            color: #2d4a5a;
            font-size: 1rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        }
        section[data-testid="stSidebar"] .stMarkdown {
            color: #1a1a2e;
        }
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
        .stTextArea > div > div > textarea {
            color: #1a1a2e;
            background: #ffffff;
        }
        .streamlit-expanderHeader {
            color: #1a1a2e;
            background: #f0f4f8;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    global AZURE_API_KEY
    try:
        if hasattr(st, 'secrets') and 'AZURE_API_KEY' in st.secrets:
            AZURE_API_KEY = st.secrets['AZURE_API_KEY']
    except Exception:
        pass

    if not AZURE_API_KEY:
        st.error(
            "Azure API key is not configured. Please set AZURE_API_KEY in Streamlit secrets or environment variables.")
        st.info(
            "For local setup, create a `.streamlit/secrets.toml` file with:\n"
            "```toml\nAZURE_API_KEY = \"your-api-key-here\"\n```\n\n"
            "Or set an environment variable:\n"
            "```bash\nexport AZURE_API_KEY=\"your-api-key-here\"\n```"
        )
        st.stop()

    init_session_state()

    col_chat, col_side = st.columns([2, 1])

    # ---- RIGHT COLUMN ----
    with col_side:
        st.markdown("### ⚙️ QA Scores Configuration")
        disabled = st.session_state.conversation_started
        qa_scores_json_str = st.text_area(
            "QA Scores JSON",
            value=json.dumps(
                st.session_state.qa_scores_json, ensure_ascii=False, indent=2
            ),
            height=300, disabled=disabled, key="cfg_qa"
        )

        st.markdown("---")

        # ---- PIPELINE MONITOR ----
        st.markdown("### 📊 Pipeline")
        if st.session_state.pipeline_logs:
            latest_log = st.session_state.pipeline_logs[-1]
            with st.expander("Latest request details"):
                st.markdown(
                    f"**Timestamp:** {latest_log.get('timestamp', '—')}")
                st.markdown(
                    f"**User message:** {latest_log.get('user_message', '—')}")
                st.markdown("---")
                for i, step in enumerate(latest_log.get("steps", [])):
                    step_label = step.get(
                        "step", "unknown").replace("_", " ").title()
                    with st.expander(f"Step {i + 1}: {step_label}"):
                        st.json(step)
        else:
            st.caption("No messages processed yet")

        # ---- DOWNLOAD PIPELINE LOGS ----
        if st.session_state.pipeline_logs:
            st.download_button(
                label="📥 Download Pipeline Logs",
                data=get_download_pipeline_logs_json(),
                file_name=f"pipeline_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")

        # ---- DOWNLOAD CONVERSATION ----
        if st.session_state.messages:
            st.markdown("### 📥 Download Conversation")
            st.download_button(
                label="📥 Download JSON",
                data=get_download_conversation_json(),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown("---")

        # ---- LOAD EXISTING CONVERSATION ----
        st.markdown("### 📤 Load Existing Conversation")
        uploaded_file = st.file_uploader("Upload JSON file", type=[
                                         "json"], key="file_upload")
        if uploaded_file is not None:
            if st.button("📂 Load from file", use_container_width=True):
                content = uploaded_file.read().decode('utf-8')
                if load_conversation_from_json(content):
                    st.success("Conversation loaded!")
                    st.rerun()

        paste_json = st.text_area(
            "Or paste conversation JSON here", height=150, key="paste_json")
        if st.button("📋 Load from pasted JSON", use_container_width=True):
            if paste_json.strip():
                if load_conversation_from_json(paste_json):
                    st.success("Conversation loaded!")
                    st.rerun()
            else:
                st.warning("Please paste JSON first.")

        st.markdown("---")

        # ---- RESET ----
        if st.session_state.conversation_started:
            if st.button("🔄 Reset Conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.session_state.qa_scores_json = DEFAULT_QA_SCORES_JSON
                st.session_state.pipeline_logs = []
                st.rerun()

    # ---- LEFT COLUMN: CHAT ----
    with col_chat:
        st.markdown(
            '<div class="main-header">🎨 Portrait QA Conversational Assistant</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Curaay - Portrait Evaluation Assistant</div>',
            unsafe_allow_html=True
        )

        # ---- START CONVERSATION ----
        if not st.session_state.conversation_started:
            first_message = st.text_input(
                "💬 First message:",
                placeholder="e.g., Explain what I should improve",
                key="first_message_input"
            )

            if st.button("🎬 Start Conversation", use_container_width=True):
                try:
                    qa_scores_json = json.loads(qa_scores_json_str)
                    st.session_state.qa_scores_json = qa_scores_json
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in QA Scores: {e}")
                    st.stop()

                if first_message.strip():
                    st.session_state.messages.append({
                        "role": "user",
                        "content": first_message.strip()
                    })
                    messages_for_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    with st.spinner("Thinking..."):
                        response, log_entry = process_user_message(
                            qa_scores_json, messages_for_api
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })
                    st.session_state.pipeline_logs.append(log_entry)
                else:
                    with st.spinner("Starting conversation..."):
                        response, log_entry = process_user_message(
                            qa_scores_json, []
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })
                    log_entry = {
                        **log_entry,
                        "user_message": None,
                        "note": "Initial greeting — no user message yet",
                    }
                    st.session_state.pipeline_logs.append(log_entry)

                st.session_state.conversation_started = True
                st.rerun()

        # ---- DISPLAY CHAT ----
        if st.session_state.messages:
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'''
                        <div class="chat-message user-message">
                            <strong>👤 User:</strong><br>{msg["content"]}
                        </div>
                        ''', unsafe_allow_html=True)
                    elif msg["role"] == "assistant":
                        st.markdown(f'''
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant:</strong><br>{msg["content"]}
                        </div>
                        ''', unsafe_allow_html=True)

        # ---- USER INPUT ----
        if st.session_state.conversation_started:
            user_input = st.chat_input("Type your message...")
            if user_input:
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })
                messages_for_api = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                qa_scores_json = st.session_state.get(
                    "qa_scores_json", DEFAULT_QA_SCORES_JSON)

                with st.spinner("Thinking..."):
                    response, log_entry = process_user_message(
                        qa_scores_json, messages_for_api
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
                st.session_state.pipeline_logs.append(log_entry)
                st.rerun()


if __name__ == "__main__":
    main()
