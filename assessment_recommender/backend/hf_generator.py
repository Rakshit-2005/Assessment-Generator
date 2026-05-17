"""Hugging Face text-generation adapter using the Inference API.

Provides a simple `generate_assessment` function that prompts the model
to produce a JSON-formatted assessment given a job description (JD).
"""
from typing import Optional
import os
import json
import requests

HF_TOKEN = os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
# Use an instruction-tuned small model available on HF Inference API
DEFAULT_MODEL = 'google/flan-t5-small'


def _make_prompt(jd: str) -> str:
    # Instruction: output a single JSON object following the schema exactly.
    prompt = (
        "You are an expert assessment designer. Given the job description, produce a single JSON object (no extra text) "
        "with the following fields: title (string), instructions (string), duration_minutes (int), difficulty (one of \"easy\", \"medium\", \"hard\"), "
        "tags (array of strings), and questions (array). Each question must be an object with fields: id (string), type (one of 'mcq','short','coding'), "
        "stem (string), options (array of strings or empty), answer (string or array), and rubric (string). Make 4-6 questions appropriate for the JD. "
        "Keep language concise. Respond ONLY with valid JSON.\n\n"
    )
    prompt += f"Job Description:\n{jd}\n\nJSON:" 
    return prompt


def generate_assessment(jd: str, model: Optional[str] = None, max_length: int = 512):
    if not HF_TOKEN:
        raise RuntimeError('HF_API_TOKEN not set')
    model = model or DEFAULT_MODEL
    endpoint = f'https://api-inference.huggingface.co/pipeline/text-generation/{model}'
    headers = {'Authorization': f'Bearer {HF_TOKEN}', 'Content-Type': 'application/json'}
    prompt = _make_prompt(jd)
    payload = {
        'inputs': prompt,
        'parameters': { 'max_new_tokens': max_length, 'do_sample': False },
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f'HF generation failed: {resp.status_code} {resp.text}')
    data = resp.json()
    # The HF Inference API may return [{'generated_text': '...'}] or plain text; handle both.
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'generated_text' in data[0]:
        text = data[0]['generated_text']
    elif isinstance(data, dict) and 'generated_text' in data:
        text = data['generated_text']
    else:
        # fallback: convert to string
        text = json.dumps(data)

    # attempt to extract a JSON object from the generated text
    try:
        # find first '{' and last '}' to extract JSON substring
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            j = text[start:end+1]
            return json.loads(j)
        # otherwise try to parse full text
        return json.loads(text)
    except Exception:
        # if parsing fails, return raw text under a key
        return {'raw': text}
