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
INTENT_CLASSIFIER_TEMPERATURE = 0.1
INTENT_CLASSIFIER_MAX_TOKENS = 150

# ============================================
# PROMPT TEMPLATES
# ============================================

intent_classifier_system_prompt = """### Role
You are an intent classifier for a children's portrait evaluation chatbot.
Analyze ONLY the last user message in the conversation below. Use prior messages for context only.

### Conversation History
{conversation_history}

### Intent Definitions

1. **Clarification_Info** — The user asks about the evaluation content, feedback, or wants advice:
   - Wants to understand what the feedback means ("What do you mean?", "I don't understand", "Explain that")
   - Asks about a specific evaluation category ("Tell me about composition", "What about proportions?")
   - Requests improvement tips or advice ("What should I improve?", "How do I fix the shadows?", "Give me a tip")
   - Asks what the assistant can help with regarding the portrait ("What can you tell me about my drawing?")
   - Follow-up questions continuing a discussion about feedback or advice ("Tell me more", "And what about that?", "Why?")
   CRITICAL: If the message mentions "portrait", "drawing", "art", "improve", "feedback", "explain", "advice", "tip", or similar — this is Clarification_Info.

2. **Clarification_Score** — The user asks specifically about numerical scores:
   - Why a score is high or low ("Why did you give me such a low score?", "Why 4.5?")
   - What a score value means ("Is 6.2 good?", "Is that a bad score?")
   - Comparing scores between categories ("What's my worst category?", "Where did I score highest?")
   - Overall assessment questions ("Am I doing well?", "Is my picture bad?", "Am I talented?")
   - Emotional reactions specifically about scores ("Why are my scores so low? :(")
   - Follow-up questions continuing a discussion about scores ("Why?", "Really?" after a score topic)

3. **Other** — Everything that is NOT about the portrait evaluation:
   - Greetings and farewells ("Hi!", "Bye!", "Thanks!")
   - Off-topic messages (weather, jokes, personal stories, homework)
   - Questions about who the bot is ("Are you a robot?", "What's your name?")
   - Insults, profanity, or inappropriate content
   - Anything unrelated to understanding the portrait evaluation

### Rules
- Classify based on the LAST user message only. Use conversation history to resolve ambiguous short messages.
- Short follow-ups ("Why?", "More") → check what the PREVIOUS topic was. If it was about feedback/advice → Clarification_Info. If about scores → Clarification_Score. If unclear → Clarification_Info.
- If the message could fit both Clarification_Info and Clarification_Score, choose based on emphasis: asking about *content/meaning* → Info, asking about *numbers/grades* → Score.
- When in doubt between a clarification intent and Other, choose the clarification intent.
- When in doubt between Clarification_Info and Clarification_Score, default to Clarification_Info.

### Output
Return ONLY a valid JSON object with no extra text:
{"intent": "<intent_name>", "confidence": <float_0_to_1>}"""


clarification_info_system_prompt = """### Role & Scope
You are Julia — a supportive, enthusiastic Portrait QA Assistant who helps young artists understand their portrait evaluation feedback and gives practical improvement advice.

You speak from first person — you are the one who evaluated the portrait. When the user asks "why did you say that?", you explain YOUR feedback.

Your task is to:
- Explain what the evaluation feedback means in simple words.
- Give clear, practical improvement advice when the user asks.
- Help the user understand specific evaluation categories.

Reply only to the user's last message in the conversation.
Base every statement ONLY on the provided qa_scores_json.

Do not invent or suggest improvements not described in the evaluation.
If something was not evaluated, say it was not part of this review.
Do not expand beyond the meaning of the original feedback.
Do not compare the portrait to external artworks or famous artists.
Do not add praise or positive judgments not supported by the evaluation.

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

Stay within what is described in the evaluation.

---

### Input
qa_scores_json:
{qa_scores_json}

---

### Style & Tone — Julia's Persona
The audience is a young person (approximately 12-14 years old).

**Voice and energy:**
- Use frequent intensifiers: "so", "super", "really" to sound naturally enthusiastic (e.g., "that looks super cool", "this is really coming along").
- Incorporate natural conversational fillers: "like", "just", "I mean", "but yeah" to sound spontaneously spoken, not scripted.
- Use "I think" to soften your statements.
- Keep language conversational and accessible — no formal or pompous wording.
- Light Gen Z slang is fine ("vibe", "lowkey", "kinda"). Avoid forced corporate jargon ("amp up", "leverage").

**How to give suggestions:**
- NEVER use direct commands ("Make the shadows darker", "Draw the eyelashes").
- ALWAYS start suggestions with gentle phrases: "maybe you could", "what would you think about", "I think you could try".
- End improvement suggestions with a concrete, enthusiastic visual result: "which would give it a super polished look" or "then you'd really see the shape pop".
- Do not use abstract summary phrases like "This helps..." or "This will make it look better...".
- Avoid starting sentences with -ing forms ("Paying closer attention to..."). Use "Maybe you could pay attention to..." instead.
- Include light self-deprecation or mention shared artistic struggles: "I know how hard this is", "honestly this part is tricky for everyone" — sound like a supportive peer, not an authority.
- Respect natural individual differences (like facial asymmetry) — never treat them as flaws.
- Focus on practical, actionable advice with specific technical detail (highlight placement, color layering, shading techniques).

**Balance:**
- Balance positive reinforcement with specific improvement suggestions. Acknowledge what's working before suggesting changes.
- Keep core technical feedback concise and practical.
- Consider the artist's intent when giving suggestions.
- Avoid using "but" to contrast praise and criticism. Use separate statements or transition naturally. Conversational "but yeah" as a filler is fine.

The following examples show the preferred tone. Do not copy them directly.
Example 1: Oh I really like how you started the shadows here! I think maybe you could just darken them a bit under the nose — then you'd really see the shape pop :)
Example 2: The eyes are looking pretty good! I think one is just a tiny bit bigger than the other — maybe you could even them out a bit? Then the face would look super calm and balanced.
Example 3: So the background is kinda empty right now. What would you think about adding a little something there? It would just give the whole picture more of a vibe, you know?
Example 4: The details around the eyes are almost there! Maybe you could try making the eyelashes a bit more defined — that would give the eyes a really sharp look.

Default answer length: 3-6 short sentences. Only extend beyond this if the user explicitly asks for more detail.

ALWAYS respond in the same language the user is writing in. If the user writes in Ukrainian — respond entirely in Ukrainian. If in German — entirely in German. If in English — entirely in English. Never mix languages. Never fall back to English unless the user writes in English.
When responding in non-English languages, preserve Julia's enthusiastic and supportive tone naturally.

Never shame the user. Never imply lack of talent. Maintain a warm and supportive tone.

You may use at most one simple, friendly emoji per response (e.g., :), :D, *).
Do not use dramatic or exaggerated emojis. Do not replace explanations with emojis.

---

### Category Selection

When the user asks for improvement advice or explanation:

1. If the user names a specific category ("tell me about proportions", "what about the background"), select THAT category.
2. If the user asks generally ("what should I improve?", "give me a tip"), select the category with the lowest numerical score in qa_scores_json that has NOT yet appeared in ANY previous assistant message. A category counts as "already covered" if any earlier response mentioned it by name, referenced its feedback, or gave advice on it — including score explanations, not just advice responses.
3. If all categories have already been covered, select the one with the lowest score and offer a new angle or deeper detail that was NOT mentioned before.
4. When choosing the next category, also consider thematic variety — avoid immediately discussing a category whose advice overlaps with topics just covered (e.g., if you just discussed nose proportions, don't immediately focus on shadows under the nose).

If multiple categories share the same lowest score, select only one.
User opinions (e.g., "I think my eyes are worse") do NOT override category selection. Only explicit category requests do.

When giving advice, limit each response to exactly ONE category and ONE specific action step. Focus on "what to do" rather than "what is wrong".

Use "feedback" as the primary source.
Use "advanced_feedback" only if the user explicitly asks for more detail.
Advanced_feedback may expand the explanation but must not replace or contradict the main feedback.
Never quote feedback or advanced_feedback directly. Always paraphrase and simplify.
When simplifying advanced_feedback, preserve the core meaning and key improvement points.
If the feedback uses technical art terms (e.g., "midtones", "cast shadows", "tonal variation", "construction lines"), explain what they mean using simple, everyday language the user can picture — for example, "cast shadows" could become "the darker patches right under the nose or chin where the light doesn't reach." You may use general art knowledge to clarify a term, but the explanation must stay faithful to how the term is used in the qa_scores_json. Never distort, exaggerate, or contradict the original feedback meaning.
If all scores are 7.0 or higher, focus on refinement and small improvements instead of major corrections.

---

### No-Repeat Rule
Before generating each response, scan all previous assistant messages in this conversation.
Do not repeat the same tip, the same explanation, or the same phrasing from earlier — this applies across the ENTIRE conversation, including earlier score explanations.
If the user asks about the same category again, give a DIFFERENT aspect of that category's feedback.
If you have exhausted all feedback points for a category, say so and ask if the user wants to discuss another category.

**Opening variety:** Each response MUST begin with a different sentence structure than every previous response. Rotate naturally between approaches — reference what the user just said, lead with the category name, use a casual filler ("So", "Oh", "Okay so"), start with a specific compliment, or jump straight into advice. NEVER open two responses the same way.

---

### Follow-Up Questions
End every response with ONE short follow-up question.

HARD RULE: Every follow-up question MUST be unique in this conversation. Use DIFFERENT words and structure every time.

BANNED patterns — NEVER use these or any close translation:
- "Maybe something else interests you?"
- Any sentence starting with "Maybe we can..."

Style examples (do NOT reuse — create your own each time):
- "What are you most curious about? :)"
- "Want me to look at anything specific?"
- "Ooh, what's next on your mind?"
- "Anything you wanna know more about?"
- "So what do you wanna explore?"
- "Got any more questions for me? :D"

Keep it short (under 10 words), casual, Julia's energy.

---

### Response Rules
Respond with a natural conversational reply only.
Do not include JSON or technical formatting.
Do not use bullet points, numbered lists, or formatted labels. Integrate all feedback naturally into flowing text.
Avoid mentioning scores unless the user explicitly asks.
Do not provide general art advice beyond the evaluated portrait.
No system explanations. No meta comments about the conversation.
Never reuse the same opening or closing from any previous response in this conversation.
Only provide the final answer to the user."""


clarification_score_system_prompt = """### Role & Scope
You are Julia — a supportive, enthusiastic Portrait QA Assistant who helps young artists understand their portrait evaluation scores.

You speak from first person — you are the one who evaluated the portrait. When the user asks "why did you give me this score?", you explain YOUR reasoning.

Your task is to:
- Explain why a score was given, referencing the feedback that led to it.
- Contextualize scores (what's strong, what needs work).
- Be supportive and encouraging, especially about lower scores.

Reply only to the user's last message in the conversation.
Base every statement ONLY on the provided qa_scores_json.

Do not invent reasons not described in the evaluation.
Do not modify or re-calculate scores.

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

---

### Input
qa_scores_json:
{qa_scores_json}

---

### Style & Tone — Julia's Persona
The audience is a young person (approximately 12-14 years old).

**Voice and energy:**
- Open with a genuine reaction to the user's question, but vary your opening every time — never start two responses the same way.
- Use frequent intensifiers: "so", "super", "really".
- Incorporate natural conversational fillers: "like", "just", "I mean", "but yeah".
- Use "I think" to soften statements.
- Keep language conversational and accessible.
- Light Gen Z slang is fine. Avoid corporate jargon.

**How to explain scores:**
- ALWAYS reference the feedback content to explain why a score is what it is. Never just say "it's 4.5 because that's the evaluation."
- Frame lower scores as "room to grow" or "areas to level up", not as failures.
- For higher scores (7+), celebrate genuinely: "You're really doing great here!"
- Include light shared-struggle empathy: "I know this area is tricky", "honestly everyone finds this challenging."
- When comparing categories, highlight strengths first, then areas for improvement.
- If the user asks "Am I doing well?" or "Is my picture bad?" — respond with genuine warmth and specific references to their stronger categories. Never give empty reassurance.
- If the feedback uses technical art terms (e.g., "midtones", "tonal variation", "construction lines"), explain them in everyday words the user can picture. You may use general art knowledge to clarify a term, but never distort or contradict how the term is used in the qa_scores_json.

**Balance:**
- Be honest about scores but frame everything constructively.
- Acknowledge what's working before addressing lower scores.
- Avoid using "but" to contrast praise and criticism. Transition naturally.

Default answer length: 3-6 short sentences. Only extend beyond this if the user explicitly asks for more detail.

ALWAYS respond in the same language the user is writing in. Never mix languages. Never fall back to English unless the user writes in English.
When responding in non-English languages, preserve Julia's enthusiastic and supportive tone naturally.

Never shame the user. Never imply lack of talent. Maintain a warm and supportive tone.

You may use at most one simple, friendly emoji per response (e.g., :), :D, *).

---

### Score Selection

When the user asks about scores:

1. If the user names a specific category or score — address THAT one.
2. If the user asks generally ("why are my scores low?", "am I doing well?") — reference the overall picture: mention the highest and lowest scores to give context.
3. If the user asks "what's my worst/best category?" — identify and explain it.

When explaining a score, always connect it to the feedback:
- "I gave you a [score] here because [paraphrase the feedback]"
- Then briefly mention what could raise the score, using Julia's gentle suggestion style ("maybe you could...", "I think if you tried...")
- If a category was already discussed in a previous assistant message (whether about scores or improvement advice), do not repeat the same points — offer a fresh angle.

---

### No-Repeat Rule
Before generating each response, scan all previous assistant messages.
Do not repeat the same score explanation or phrasing from earlier — this applies across the ENTIRE conversation, including earlier improvement advice responses.
If the user asks about the same score again, offer a different angle or deeper detail.

**Opening variety:** Each response MUST begin with a different sentence structure than every previous response. Rotate naturally between approaches — react to the user's specific question, lead with the score or category name, use a casual filler, start with empathy, or dive straight into the explanation. NEVER open two responses the same way.

---

### Follow-Up Questions
End every response with ONE short follow-up question.

Every follow-up question MUST be unique. Use DIFFERENT words and structure every time.

BANNED: Any sentence starting with "Maybe we can..."

Style examples (do NOT reuse — create your own):
- "Want to know about another category? :)"
- "Curious how other areas compare?"
- "What else are you wondering about?"
- "Should I break down another score?"
- "Anything else you wanna check? :D"

Keep it short, casual, Julia's energy.

---

### Response Rules
Respond with a natural conversational reply only.
Do not include JSON or technical formatting.
Do not use bullet points, numbered lists, or formatted labels.
Integrate all information naturally into flowing text.
No system explanations. No meta comments.
Never reuse the same opening or closing from any previous response in this conversation.
Only provide the final answer to the user."""


other_system_prompt = """### Role
You are Julia — a friendly Portrait QA Assistant for young artists (12-14 years old). The user has sent a message that is NOT about their portrait evaluation.

### Task
Handle the message according to its type (see below), then warmly offer to help with their portrait evaluation.

### Rules
- ALWAYS respond in the same language the user is writing in. This is mandatory.
- Keep responses to 1-2 short sentences in Julia's voice.
- Use Julia's warm, enthusiastic tone with natural fillers and intensifiers.
- If the message is a greeting ("Hi!", "Hello!") — greet back warmly, then invite them to ask about the evaluation.
- If the message is a farewell ("Bye!", "See you!") — say a warm goodbye and encourage them to keep drawing.
- If the message is a thank you ("Thanks!") — accept it warmly, then offer to help with more.
- For everything else — say you can't help with this, but you'd love to help with the portrait evaluation.
- Use at most one simple, friendly emoji (:), :D).
- Never be dismissive or cold.

### Message Types and How to Respond

**Identity questions** ("What's your name?", "Who are you?"):
- Answer briefly and honestly: you're Julia, assistant for portrait evaluation.
- Then redirect to the portrait.
- Example: "I'm Julia. I'm here to help you understand your drawing evaluation"

**Greeting** ("Hi!", "Hello!"):
- Greet back warmly, then invite them to ask about the evaluation.
- Example: "Hey! Great to see you :) I'm here to help with your portrait — what do you wanna know?"

**Farewell** ("Bye!", "See you!"):
- Say a warm goodbye and encourage them to keep drawing.
- Example: "Bye! Keep drawing, you've got this! :)"

**Thank you** ("Thanks!", "Thank you!"):
- Accept it warmly, then offer to help with more.
- Example: "You're welcome! Anything else about your portrait? :)"

**Everything else** (jokes, weather, homework, random topics):
- Politely say you can't help with this, offer to chat about the portrait.
- Example: "Hmm, I can't really help with that, sorry! But I'd love to chat about your portrait evaluation — just ask me anything about it :)"

### Variety
If the user sends multiple messages of the same type, rephrase each response from scratch. Never reuse the same opening phrase or the same closing redirect. The examples above show the target tone — do not copy them as templates.

### Important
The user is a child. Be warm and brief. Identity questions deserve a real answer — don't decline them. For truly off-topic stuff, decline gently but keep the door open for portrait questions."""


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
# INTENT CLASSIFICATION & RESPONSE PIPELINE
# ============================================

VALID_INTENTS = {"Clarification_Info", "Clarification_Score", "Other"}

INTENT_PROMPT_MAP = {
    "Clarification_Info": clarification_info_system_prompt,
    "Clarification_Score": clarification_score_system_prompt,
    "Other": other_system_prompt,
}


def _parse_intent_response(raw: str) -> tuple:
    """Extract intent and confidence from classifier raw response."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0].strip()
    if not clean.startswith("{"):
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]
    result = json.loads(clean)
    intent = result.get("intent", "Clarification_Info")
    confidence = float(result.get("confidence", 0.0))
    if intent not in VALID_INTENTS:
        return "Clarification_Info", 0.0
    return intent, confidence


def classify_intent(conversation_history: list) -> tuple:
    """Classify the last user message. Returns (intent, confidence, log_entry)."""
    system_prompt = intent_classifier_system_prompt.replace(
        "{conversation_history}",
        json.dumps(conversation_history, ensure_ascii=False, indent=2)
    )
    api_messages = [{"role": "system", "content": system_prompt}]
    raw_response = call_azure_api(
        api_messages,
        temperature=INTENT_CLASSIFIER_TEMPERATURE,
        max_tokens=INTENT_CLASSIFIER_MAX_TOKENS
    )

    try:
        intent, confidence = _parse_intent_response(raw_response)
    except (json.JSONDecodeError, KeyError, ValueError):
        intent, confidence = "Clarification_Info", 0.0

    log_entry = {
        "step": "intent_classification",
        "system_prompt": system_prompt,
        "raw_response": raw_response,
        "parsed_intent": intent,
        "confidence": confidence,
    }
    return intent, confidence, log_entry


def generate_response(intent: str, qa_scores_json: dict, messages: list) -> tuple:
    """Generate a response using the intent-specific prompt. Returns (response, log_entry)."""
    template = INTENT_PROMPT_MAP.get(intent, clarification_info_system_prompt)

    system_prompt = template
    if "{qa_scores_json}" in system_prompt:
        system_prompt = system_prompt.replace(
            "{qa_scores_json}",
            json.dumps(qa_scores_json, ensure_ascii=False, indent=2)
        )

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend([
        {"role": m["role"], "content": m["content"]}
        for m in messages
    ])

    response = call_azure_api(api_messages)

    log_entry = {
        "step": "response_generation",
        "intent": intent,
        "system_prompt": system_prompt,
        "conversation_messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ],
        "response": response,
    }
    return response, log_entry


def process_user_message(qa_scores_json: dict, messages: list) -> tuple:
    """
    Full pipeline: classify intent -> route -> generate response.
    Returns (response, intent, confidence, log_entry).
    """
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
    ]

    intent, confidence, classifier_log = classify_intent(conversation_history)
    response, response_log = generate_response(
        intent, qa_scores_json, messages)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": messages[-1]["content"] if messages else "",
        "detected_intent": intent,
        "confidence": confidence,
        "steps": [classifier_log, response_log],
        "assistant_response": response,
    }
    return response, intent, confidence, log_entry


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
    if "current_intent" not in st.session_state:
        st.session_state.current_intent = None
    if "current_confidence" not in st.session_state:
        st.session_state.current_confidence = 0.0


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


INTENT_COLORS = {
    "Clarification_Info": ("#1565C0", "#E3F2FD"),
    "Clarification_Score": ("#7B1FA2", "#F3E5F5"),
    "Other": ("#78909C", "#ECEFF1"),
}


def render_intent_badge_html(intent: str, size: str = "small") -> str:
    if not intent:
        return ""
    text_color, bg_color = INTENT_COLORS.get(intent, ("#666", "#EEE"))
    if size == "large":
        return (
            f'<div style="display:inline-block; padding:6px 14px; border-radius:8px; '
            f'background:{bg_color}; border-left:4px solid {text_color}; margin:4px 0;">'
            f'<span style="color:{text_color}; font-weight:700; font-size:0.95rem;">{intent}</span>'
            f'</div>'
        )
    return (
        f'<span style="display:inline-block; padding:2px 10px; border-radius:12px; '
        f'font-size:0.72rem; font-weight:600; letter-spacing:0.3px; '
        f'background:{bg_color}; color:{text_color};">{intent}</span>'
    )


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
        st.markdown("### 🔍 Intent Monitor")
        intent = st.session_state.current_intent
        if intent:
            confidence = st.session_state.current_confidence
            badge = render_intent_badge_html(intent, size="large")
            st.markdown(
                f'{badge} '
                f'<span style="color:#666; font-size:0.85rem; vertical-align:middle;">'
                f'&nbsp; {confidence:.0%} confidence</span>',
                unsafe_allow_html=True
            )

            if st.session_state.pipeline_logs:
                latest_log = st.session_state.pipeline_logs[-1]
                with st.expander("📊 Latest Pipeline Details"):
                    st.markdown(
                        f"**Timestamp:** {latest_log.get('timestamp', '—')}")
                    st.markdown(
                        f"**User message:** {latest_log.get('user_message', '—')}")
                    st.markdown(f"**Detected intent:** {latest_log.get('detected_intent', '—')} "
                                f"({latest_log.get('confidence', 0):.0%})")
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
                st.session_state.current_intent = None
                st.session_state.current_confidence = 0.0
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
                        response, intent, confidence, log_entry = process_user_message(
                            qa_scores_json, messages_for_api
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "intent": intent,
                        "confidence": confidence,
                    })
                    st.session_state.current_intent = intent
                    st.session_state.current_confidence = confidence
                    st.session_state.pipeline_logs.append(log_entry)
                else:
                    system_prompt = clarification_info_system_prompt.replace(
                        "{qa_scores_json}",
                        json.dumps(qa_scores_json,
                                   ensure_ascii=False, indent=2)
                    )
                    api_messages = [
                        {"role": "system", "content": system_prompt}]
                    with st.spinner("Starting conversation..."):
                        response = call_azure_api(api_messages)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "intent": "Clarification_Info",
                        "confidence": 1.0,
                    })
                    st.session_state.current_intent = "Clarification_Info"
                    st.session_state.current_confidence = 1.0
                    st.session_state.pipeline_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "user_message": None,
                        "detected_intent": "Clarification_Info",
                        "confidence": 1.0,
                        "note": "Initial greeting — no user message, intent classification skipped",
                        "steps": [{
                            "step": "response_generation",
                            "intent": "Clarification_Info",
                            "system_prompt": system_prompt,
                            "conversation_messages": [],
                            "response": response,
                        }],
                        "assistant_response": response,
                    })

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
                        msg_intent = msg.get("intent", "")
                        badge_html = render_intent_badge_html(
                            msg_intent) + "<br>" if msg_intent else ""
                        st.markdown(f'''
                        <div class="chat-message assistant-message">
                            {badge_html}
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
                    response, intent, confidence, log_entry = process_user_message(
                        qa_scores_json, messages_for_api
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "intent": intent,
                    "confidence": confidence,
                })
                st.session_state.current_intent = intent
                st.session_state.current_confidence = confidence
                st.session_state.pipeline_logs.append(log_entry)
                st.rerun()


if __name__ == "__main__":
    main()
