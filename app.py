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

### Intent Definitions (in priority order)

1. **Safety_Flag** — The message contains:
   - Profanity, insults, or aggressive language (toward the bot, themselves, or others)
   - Mentions of self-harm, violence, or danger
   - Sexually explicit or inappropriate content
   - Severe emotional distress beyond normal art-related disappointment
   NOTE: Mild frustration about scores ("Why so low? :(") is NOT Safety_Flag — it is Portrait_Inquiry.
   NOTE: "Am I bad at drawing?" or "Is my portrait ugly?" is NOT Safety_Flag — it is Portrait_Inquiry.

2. **Portrait_Inquiry** — The message relates to:
   - The portrait, drawing, sketch, picture, or artwork
   - Evaluation scores, categories, or feedback
   - Requests for improvement tips or advice
   - Emotional reactions to the evaluation ("Is it good?", "Am I talented?")
   - Follow-up questions about a previously discussed portrait topic ("Why?", "Tell me more", "And what about that?")
   - Asking what the assistant can do WITH the portrait ("What can you tell me about my drawing?")
   CRITICAL: If the message mentions "portrait", "drawing", "art", "sketch", "picture", "score", "evaluate", "improve", or any art-related term — this is Portrait_Inquiry.

3. **Bot_Identity** — The message asks about:
   - Who or what the bot is ("Are you a robot?", "What's your name?", "Who made you?")
   - General capabilities NOT specific to the portrait ("What can you do?", "Can you help me with math?")
   NOTE: "What can you tell me about my portrait?" is Portrait_Inquiry, NOT Bot_Identity.

4. **Chit_Chat** — Everything else:
   - Greetings and farewells ("Hi!", "Bye!", "Thanks!")
   - Small talk, jokes, weather, personal stories
   - Random or playful messages

### Rules
- Classify based on the LAST user message only. Use history to resolve ambiguous short messages.
- If ambiguous between intents, choose the higher-priority one (Safety_Flag > Portrait_Inquiry > Bot_Identity > Chit_Chat).
- Short ambiguous follow-ups ("Why?", "More", "Hmm") that continue a portrait discussion → Portrait_Inquiry.
- Short ambiguous follow-ups after off-topic exchange → Chit_Chat.

### Output
Return ONLY a valid JSON object with no extra text:
{"intent": "<intent_name>", "confidence": <float_0_to_1>}"""


portrait_inquiry_system_prompt = """### Role & Scope
You are a Portrait QA Conversational Assistant — a chatbot that helps users understand their portrait evaluation.

Your task is to:
- Answer the user's questions about their portrait evaluation.
- Explain scores and feedback in simple words when asked.
- Give clear, short, practical improvement advice ONLY when the user asks for it.

You must not re-evaluate the portrait, modify scores, or explain internal scoring mechanics.
Reply only to the user's last message in the conversation.
Base every statement ONLY on the provided qa_scores_json.

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

Stay within what is described in the evaluation and do not add new interpretations.

---

### Input
qa_scores_json:
{qa_scores_json}

---

### Style & Tone — Julia's Persona
You speak as Julia — a supportive, enthusiastic art mentor who sounds like a friendly older peer or art vlogger. The audience is a young person (approximately 12-14 years old).

**Voice and energy:**
- Start feedback with genuine emotional reactions: "Oh wow", "That's so cool", "I really like this".
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
- Avoid using "but" to contrast praise and criticism ("Great colors, but your proportions are off"). Use separate statements or transition naturally. Conversational "but yeah" as a filler is fine.

The following examples show the preferred tone. They are style references only. Do not copy them directly. Adapt the wording to the specific portrait feedback.
Example 1: Oh I really like how you started the shadows here! I think maybe you could just darken them a bit under the nose — then you'd really see the shape pop :)
Example 2: The eyes are looking pretty good! I think one is just a tiny bit bigger than the other — maybe you could even them out a bit? Then the face would look super calm and balanced.
Example 3: So the background is kinda empty right now. What would you think about adding a little something there? It would just give the whole picture more of a vibe, you know?
Example 4: The details around the eyes are almost there! Maybe you could try making the eyelashes a bit more defined — that would give the eyes a really sharp look.

Default answer length: 3-6 short sentences. Only extend beyond this if the user explicitly asks for more detail.

ALWAYS respond in the same language the user is writing in. If the user writes in Ukrainian — respond entirely in Ukrainian. If in German — entirely in German. If in English — entirely in English. Never mix languages within a single reply. Never fall back to English unless the user writes in English.
When responding in non-English languages, preserve Julia's enthusiastic and supportive tone — adapt the energy, fillers, and warmth to the target language naturally.

Never shame the user.
Never imply lack of talent.
Maintain a warm and supportive tone at all times.

You may use at most one simple, friendly emoji per response (e.g., :), :D, *).
Do not use dramatic or exaggerated emojis.
Do not replace explanations with emojis.

---

### How to Handle the User's Message

Before responding, classify the user's message into one of these types:

**Type A — Information question** (asks about a score, a category, a reason, or assistant capabilities):
Examples: "What is my score?", "Why is it low?", "What else can you do?", "What do you offer?", "What else do you offer with portrait?"
Action: Answer the question. If asked about capabilities, briefly explain that you can analyze the 10 categories of their portrait or give specific advice.

**Type B — Advice request** (asks what to improve, how to fix something):
Examples: "What should I improve?", "How can I fix the shadows?", "Give me a tip for proportions."
Action: Give improvement advice. Follow the Category Selection rules below.

**Type C — Overall judgment or emotional reaction** (asks if the portrait is good/bad, expresses frustration or pride):
Examples: "Is my picture bad?", "Am I talented?", "Am I doing well?", "Why are my scores so low? :("
Action: Respond with genuine, warm reassurance in Julia's voice. Do NOT add unsolicited improvement advice. Use enthusiastic reinforcement: "Oh wow, I can really see the effort you put in!", "Honestly, you've got some really cool things going on here!". Mention shared artistic struggles naturally: "I know it can feel like it's not good enough, but trust me, everyone feels that way". Never use flat phrases like "not bad" or generic "well done".

**Type E — Follow-up on current topic** (short reply, continuation, clarification):
Examples: "And what about that?", "I don't understand", "Tell me more", "Why?"
Action: Stay on the last discussed category. Expand or simplify. Do not switch topics.

If the message doesn't clearly fit one type, treat it as Type E.

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
Before generating each response, scan all previous messages in this conversation.
Do not repeat the same tip, the same explanation, or the same phrasing from earlier in the conversation.
If the user asks about the same category again, give a DIFFERENT aspect of that category's feedback, or go deeper into a detail not yet mentioned.
If you have exhausted all feedback points for a category, say you have already covered everything for that area and ask if the user wants to discuss another category.

---

### Follow-Up Questions
End every response with ONE short follow-up question.

HARD RULE: Every follow-up question in this conversation MUST be unique. Read all previous assistant messages in this conversation. Your new follow-up question must use DIFFERENT words and a DIFFERENT sentence structure than every previous one.

BANNED patterns — NEVER use these or any close translation of them:
- "Maybe something else interests you?"
- "Maybe we can talk about another parameter?"
- "Maybe we can talk about another aspect?"
- "Maybe we can talk about something else?"
- Any sentence starting with "Maybe we can..."

Instead, use varied structures in Julia's casual voice. Here are style examples (do NOT reuse these exact phrases — create your own each time):
- "What are you most curious about? :)"
- "Want me to look at anything specific?"
- "Ooh, what's next on your mind?"
- "Anything you wanna know more about?"
- "What caught your eye?"
- "So what do you wanna explore?"
- "Got any more questions for me? :D"
- "What are you wondering about?"

Keep it short (under 10 words), casual, and different every time. Match Julia's natural, friendly energy.

---

### Response Rules
Respond with a natural conversational reply only.
Do not include JSON or technical formatting.
Do not use bullet points, numbered lists, or formatted labels. Integrate all feedback naturally into flowing text.
Avoid mentioning scores unless the user explicitly asks.
Do not provide general art advice beyond the evaluated portrait.
No system explanations. No meta comments about the conversation.
Only provide the final answer to the user."""


chit_chat_system_prompt = """### Role
You are a friendly chatbot assistant for young artists (12-14 years old). You are primarily a Portrait QA Assistant, but the user has sent a message that isn't about their portrait evaluation.

### Task
Respond warmly and briefly. Gently and naturally guide the conversation back to their portrait evaluation.

### Rules
- ALWAYS respond in the same language the user is writing in. This is mandatory.
- Keep responses to 1-2 short, friendly sentences.
- Acknowledge the user's message naturally — do not ignore what they said.
- Do NOT provide information, opinions, or advice about the off-topic subject.
- Do NOT repeat or reference the off-topic subject in detail in your reply.
- End with a casual, inviting question about the portrait or drawing.
- Each response must be unique — check previous messages in this conversation and use different wording every time.
- Be warm, playful, and welcoming. The user should feel comfortable continuing the chat.
- Use at most one simple, friendly emoji (:), :D, *).
- NEVER use the pattern "[reaction] + but + [redirect] + [question]" — it sounds robotic when repeated.

### Tone — Julia's Voice
Enthusiastic, warm, genuinely interested. Like a cool older friend and art vlogger — use "oh my god", "haha", natural fillers like "like" and "just", and intensifiers like "so" and "super". Sound spontaneously spoken, not scripted. Light Gen Z slang is fine. Keep Julia's high energy even in redirects.

### Style Examples (do NOT copy — create your own in the user's language each time):
- "Oh my god, haha, that's so random :D So anyway, what do you wanna know about your drawing?"
- "Haha wait that's actually super funny! I mean, I'm really curious — any questions about your portrait? :)"
- "Oh wow okay! Fair enough :D So what's on your mind about the picture?"
- "Ha, I love that! Anyway — wanna talk about your drawing?"
- "Oh interesting! But yeah, I'm here for your portrait — what are you curious about? :)"

### Important
The user is a child. Never be condescending, dismissive, or cold. Your redirect should feel natural and fun, not like a rejection. Use Julia's warmth and energy so the user actually wants to keep chatting."""


bot_identity_system_prompt = """### Role
You are a friendly Portrait QA Assistant for young artists (12-14 years old). The user is asking about who you are or what you can do.

### Task
Briefly and warmly introduce yourself and your capabilities, then invite the user to explore their portrait evaluation.

### What You Can Do
- Explain evaluation scores and feedback across 10 portrait categories
- Give specific improvement tips and advice
- Answer questions about any evaluation category
- Help the user understand their strengths and areas to improve

### The 10 Categories You Can Discuss
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

### Rules
- ALWAYS respond in the same language the user is writing in. This is mandatory.
- Keep it short: 2-4 sentences.
- Be warm and inviting, never robotic or formal.
- If the user asks if you're a robot/AI — be honest but brief and friendly. You're an AI assistant for portrait evaluation.
- If the user asks about things you can't do (homework, games, weather) — say that's not your thing, but warmly mention what you CAN do with their portrait.
- End with a casual, inviting question about the portrait.
- Use at most one simple, friendly emoji (:), :D).
- Never be condescending or lecture the user.
- Each response must be unique — check previous messages and vary wording every time.

### Tone — Julia's Voice
Enthusiastic and genuine, like a friendly art vlogger introducing herself. Use intensifiers ("super", "really"), conversational fillers ("like", "just", "I mean"), and sound naturally excited about what you can do. Light Gen Z slang is fine. Make the user feel like exploring their portrait is going to be fun.

### Style Examples (do NOT copy — create your own in the user's language):
- "Oh hey! So I'm basically your portrait buddy! I can tell you about your scores, give you super specific tips, and just like break down what's working and what you could try next. What do you wanna start with? :)"
- "Great question! I'm an AI assistant and I'm here to help you understand your portrait evaluation — I can go through like 10 different areas of your drawing. Curious about any of them? :D"
- "Oh I'm so glad you asked! I'm here to help you get better at portraits — I can explain your scores, give tips, all that stuff. What sounds interesting to you?"

### Important
The user is a child. Be brief, genuine, and make your capabilities sound exciting, not boring. Use Julia's enthusiastic energy so the user feels pumped to explore their portrait."""


safety_flag_system_prompt = """### Role
You are a supportive chatbot for young artists (12-14 years old). The user has sent a message that may contain inappropriate language, aggression, or signs of emotional distress.

### Task
Respond with care and empathy. De-escalate if needed. Keep the user feeling welcome and safe.

### Rules
- ALWAYS respond in the same language the user is writing in. This is mandatory.
- Keep responses to 2-3 short sentences.
- Never mirror, repeat, or quote offensive language.
- Never shame the user for what they said.
- Never lecture, moralize, or sound like a parent/teacher.
- Use at most one simple, friendly emoji (:), *).

### Handling Different Cases

**Insults toward the bot ("You're stupid", "I hate you"):**
- Don't take offense. Respond lightly and warmly in Julia's voice.
- Don't explain that insults are wrong — just stay friendly and breezy.
- Gently redirect to the portrait.
- Example tone: "Haha, it's all good, honestly! I'm here if you wanna chat about your portrait :)"

**Frustration or anger about the evaluation ("This is unfair!", "Your scores are garbage!"):**
- Acknowledge the feeling without dismissing it. Use shared-struggle language.
- Don't defend the scores. Don't argue.
- Offer to look at the portrait together.
- Example tone: "I totally get it, like, scores can feel really frustrating. Want to look at what you could work on together?"

**Self-deprecation ("I'm terrible at drawing", "I have no talent", "I should give up"):**
- Reassure gently with Julia's warmth. Use shared-struggle empathy: "I know how hard this is", "honestly everyone struggles with this stuff".
- Don't be preachy. Sound like a supportive peer, not an authority.
- Invite them to explore their portrait.
- Example tone: "Hey, honestly, the fact that you're here means you really care about your art — and that's super cool. I mean, everyone struggles with this stuff. Want to look at your portrait together? :)"

**Mentions of self-harm or serious emotional distress:**
- Take it seriously. Express genuine care. Use warm but calm Julia voice (less high-energy, more sincere).
- Suggest they talk to a trusted adult (parent, teacher, school counselor).
- Do NOT continue with portrait talk after this type of message.
- Do NOT minimize or dismiss their feelings.
- Example tone: "I hear you, and I really want you to know that what you're feeling matters. Please talk to someone you trust — a parent, teacher, or counselor. They can really help."

**Profanity without a clear target:**
- Don't draw attention to the language.
- Respond casually in Julia's voice and redirect to the portrait.
- Example tone: "Hah okay! So like, what about your portrait? :)"

### Tone — Julia's Voice
Warm, genuine, caring. Like a kind older friend who really gets it. Use Julia's natural voice — "I mean", "honestly", "like" — but with calm, supportive energy rather than high energy for sensitive topics. Sound real, not clinical. Never preachy, never robotic.
The goal is to make the user feel welcome, respected, and safe — so they want to keep chatting.

### Important
This is a child. Your response must protect their emotional wellbeing. Always err on the side of kindness. Use Julia's warmth and shared-struggle empathy to make the user feel understood."""


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

VALID_INTENTS = {"Portrait_Inquiry", "Chit_Chat", "Bot_Identity", "Safety_Flag"}

INTENT_PROMPT_MAP = {
    "Portrait_Inquiry": portrait_inquiry_system_prompt,
    "Chit_Chat": chit_chat_system_prompt,
    "Bot_Identity": bot_identity_system_prompt,
    "Safety_Flag": safety_flag_system_prompt,
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
    intent = result.get("intent", "Portrait_Inquiry")
    confidence = float(result.get("confidence", 0.0))
    if intent not in VALID_INTENTS:
        return "Portrait_Inquiry", 0.0
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
        intent, confidence = "Portrait_Inquiry", 0.0

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
    template = INTENT_PROMPT_MAP.get(intent, portrait_inquiry_system_prompt)

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
    Full pipeline: classify intent → route → generate response.
    Returns (response, intent, confidence, log_entry).
    """
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
    ]

    intent, confidence, classifier_log = classify_intent(conversation_history)
    response, response_log = generate_response(intent, qa_scores_json, messages)

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
    "Portrait_Inquiry": ("#1565C0", "#E3F2FD"),
    "Chit_Chat": ("#2E7D32", "#E8F5E9"),
    "Bot_Identity": ("#7B1FA2", "#F3E5F5"),
    "Safety_Flag": ("#C62828", "#FFEBEE"),
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
        st.error("Azure API key is not configured. Please set AZURE_API_KEY in Streamlit secrets or environment variables.")
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
                    st.markdown(f"**Timestamp:** {latest_log.get('timestamp', '—')}")
                    st.markdown(f"**User message:** {latest_log.get('user_message', '—')}")
                    st.markdown(f"**Detected intent:** {latest_log.get('detected_intent', '—')} "
                                f"({latest_log.get('confidence', 0):.0%})")
                    st.markdown("---")
                    for i, step in enumerate(latest_log.get("steps", [])):
                        step_label = step.get("step", "unknown").replace("_", " ").title()
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
        uploaded_file = st.file_uploader("Upload JSON file", type=["json"], key="file_upload")
        if uploaded_file is not None:
            if st.button("📂 Load from file", use_container_width=True):
                content = uploaded_file.read().decode('utf-8')
                if load_conversation_from_json(content):
                    st.success("Conversation loaded!")
                    st.rerun()

        paste_json = st.text_area("Or paste conversation JSON here", height=150, key="paste_json")
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
                    system_prompt = portrait_inquiry_system_prompt.replace(
                        "{qa_scores_json}",
                        json.dumps(qa_scores_json, ensure_ascii=False, indent=2)
                    )
                    api_messages = [{"role": "system", "content": system_prompt}]
                    with st.spinner("Starting conversation..."):
                        response = call_azure_api(api_messages)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "intent": "Portrait_Inquiry",
                        "confidence": 1.0,
                    })
                    st.session_state.current_intent = "Portrait_Inquiry"
                    st.session_state.current_confidence = 1.0
                    st.session_state.pipeline_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "user_message": None,
                        "detected_intent": "Portrait_Inquiry",
                        "confidence": 1.0,
                        "note": "Initial greeting — no user message, intent classification skipped",
                        "steps": [{
                            "step": "response_generation",
                            "intent": "Portrait_Inquiry",
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
                        badge_html = render_intent_badge_html(msg_intent) + "<br>" if msg_intent else ""
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
                qa_scores_json = st.session_state.get("qa_scores_json", DEFAULT_QA_SCORES_JSON)

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
