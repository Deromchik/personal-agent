# AI Journalist Interview Agent

A provocative AI journalist that conducts personalized interviews based on information about a specific person.

## Features

- **Journalist Agent**: Asks provocative, investigative questions in German based on personal history
- **Verification Agent**: Checks each assistant message for factual accuracy against source data
- **Streamlit UI**: Interactive chat interface with real-time fact verification
- **Logging**: Complete logs of all interactions available for download

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the application by editing the variables at the top of `app.py`:
```python
OPENAI_API_KEY = "your-openai-api-key-here"  # Your OpenAI API key
MODEL = "gpt-4o"                               # Model to use
PERSON_INFO_FILE_PATH = "person_info.txt"      # Path to person info file
```

3. Create or edit `person_info.txt` with information about the interview subject.

4. Run the application:
```bash
streamlit run app.py
```

## File Structure

```
Personal_agent/
├── app.py              # Main application with agents and Streamlit UI
├── person_info.txt     # Information about the person (input data)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How It Works

### Journalist Agent
- Takes person information and conversation history as input
- Generates provocative, journalist-style questions in German
- References specific events, dates, and decisions from the person's life

### Verification Agent
- Takes person information and assistant's last message
- Outputs JSON with verification result:
```json
{
    "truth": "true/false",
    "description": "Evidence or contradiction from source data"
}
```

## UI Features

- **Chat Panel**: Main conversation area with auto-updating messages
- **Verification Panel**: Shows fact-check results for each assistant message
- **Download Logs**: Export complete interaction logs as JSON
- **Reset Interview**: Start a new conversation

## Language

- **Prompts**: English
- **Assistant Responses**: German (Deutsch)
- **Verification Output**: English

