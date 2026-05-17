"""Local generation using Hugging Face transformers pipeline.

This loads a small instruction model (default `google/flan-t5-small`) and
generates text locally, then extracts JSON as with hf_generator.
"""
from typing import Optional
import json
import os
from llm_adapter import parse_jd

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL = os.getenv('LOCAL_GEN_MODEL') or 'google/flan-t5-small'


def _make_prompt(jd: str) -> str:
    # Strongly-constrained instruction with explicit schema and a compact example.
    return (
        "You are an expert assessment designer. Produce EXACTLY one valid JSON object and NOTHING else. "
        "Do NOT include markdown, commentary, or extraneous text. Use double quotes for keys and strings. Integers must be numbers (no quotes).\n\n"
        "Required top-level fields: \n"
        "- title (string)\n"
        "- instructions (string)\n"
        "- duration_minutes (int)\n"
        "- difficulty (one of \"easy\", \"medium\", \"hard\")\n"
        "- tags (array of strings)\n"
        "- questions (array of question objects)\n\n"
        "Question object schema: {\n  \"id\": string,\n  \"type\": \"mcq\"|\"short\"|\"coding\",\n  \"stem\": string,\n  \"options\": array (or empty array),\n  \"answer\": string or array,\n  \"rubric\": string\n}\n\n"
        "Create 4-6 questions focused on the job description. Prefer senior-level topics if JD implies seniority. Keep content concise and specific to the JD skills.\n\n"
        "Example output (use this structure exactly):\n"
        "{\n  \"title\": \"Assessment: Senior Backend Engineer (Java, Spring)\",\n  \"instructions\": \"Answer all questions. Time limit: 45 minutes.\",\n  \"duration_minutes\": 45,\n  \"difficulty\": \"hard\",\n  \"tags\": [\"java\", \"spring\", \"sql\"],\n  \"questions\": [\n    {\n      \"id\": \"q1\",\n      \"type\": \"mcq\",\n      \"stem\": \"Which HTTP status code means resource not found?\",\n      \"options\": [\"200\", \"301\", \"400\", \"404\"],\n      \"answer\": \"404\",\n      \"rubric\": \"404 indicates not found.\"\n    }\n  ]\n}\n\n"
        f"Job Description:\n{jd}\n\nNow produce ONLY the JSON object that matches the schema above."
    )


class LocalGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError('transformers not installed')
        self.model_name = model_name
        # Prefer direct seq2seq model/tokenizer (best for flan-t5).
        self.pipe = None
        self.tokenizer = None
        self.model = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        except Exception:
            # If direct loading fails, try text2text pipeline, then text-generation
            try:
                self.pipe = pipeline('text2text-generation', model=self.model_name, device=-1)
            except Exception:
                try:
                    self.pipe = pipeline('text-generation', model=self.model_name, device=-1)
                except Exception:
                    raise

    def generate(self, jd: str, max_tokens: int = 256):
        prompt = _make_prompt(jd)
        if self.pipe is not None:
            # pipeline handles different return keys depending on task
            out = self.pipe(prompt, max_length=max_tokens, do_sample=False)
            if isinstance(out, list) and out:
                first = out[0]
                if 'generated_text' in first:
                    text = first['generated_text']
                elif 'text' in first:
                    text = first['text']
                else:
                    text = str(first)
            else:
                text = str(out)
        else:
            # manual seq2seq generation
            inputs = self.tokenizer(prompt, return_tensors='pt')
            gen = self.model.generate(**inputs, max_length=max_tokens)
            text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        # extract JSON substring
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
            return json.loads(text)
        except Exception:
            return {'raw': text}


def generate_assessment_local(jd: str, model: Optional[str] = None, max_tokens: int = 256):
    gen = LocalGenerator(model_name=model or DEFAULT_MODEL)
    return gen.generate(jd, max_tokens=max_tokens)


def simple_assessment_from_jd(jd: str):
    parsed = parse_jd(jd)
    skills = parsed.get('skills', [])
    level = parsed.get('level', '') or 'Mid'
    # Build an improved senior-level assessment template
    primary_skills = skills[:5] or ['backend']
    title = f"Assessment: Senior Backend Engineer (" + ", ".join(primary_skills[:2]).title() + ")"
    tags = list(dict.fromkeys(primary_skills))
    questions = [
        {
            'id': 'q1',
            'type': 'mcq',
            'stem': 'Which technique most directly ensures a create API is idempotent across client retries?',
            'options': ['Server-generated incremental IDs', 'Client-provided idempotency key', 'Using PUT for all creates', 'Enforcing unique constraints only'],
            'answer': 'Client-provided idempotency key',
            'rubric': 'Idempotency keys (or tokens) allow the server to recognize and deduplicate retries, ensuring a single create effect.'
        },
        {
            'id': 'q2',
            'type': 'mcq',
            'stem': 'Which approach most improves read query performance for large, read-heavy tables?',
            'options': ['Vertical partitioning (sharding)', 'Creating covering indexes', 'Frequent full-table VACUUM', 'Storing JSON blobs instead of columns'],
            'answer': 'Creating covering indexes',
            'rubric': 'Covering indexes (indexes containing all needed columns) allow the DB to satisfy queries from the index without fetching the table rows, improving read performance.'
        },
        {
            'id': 'q3',
            'type': 'short',
            'stem': 'Name one advantage of using microservices.',
            'options': [],
            'answer': 'Independent deployability',
            'rubric': 'Microservices allow services to be deployed and scaled independently.'
        },
        {
            'id': 'q4',
            'type': 'coding',
            'stem': 'Implement a Spring Boot controller endpoint `GET /reverse?text=...` that returns the reversed `text` in the response body. Provide a minimal controller method.',
            'options': [],
            'answer': 'Sample Spring Boot controller:\n\n@RestController\npublic class ReverseController {\n  @GetMapping("/reverse")\n  public ResponseEntity<String> reverse(@RequestParam String text) {\n    String rev = new StringBuilder(text).reverse().toString();\n    return ResponseEntity.ok(rev);\n  }\n}\n\nUnit-test approach: call endpoint with sample text and assert response body equals reversed string.',
            'rubric': 'Controller should expose GET /reverse, accept `text` query param, reverse the string correctly, and return 200 with reversed text. Unit test should assert the reversed output.'
        },
        {
            'id': 'q5',
            'type': 'short',
            'stem': 'Write an SQL query to select top 5 highest-paid employees (table: employees(id,name,salary)).',
            'options': [],
            'answer': 'SELECT * FROM employees ORDER BY salary DESC LIMIT 5;',
            'rubric': 'Order by salary descending and limit to 5 rows.'
        },
        {
            'id': 'q6',
            'type': 'short',
            'stem': 'What does the Spring @Transactional annotation do?',
            'options': [],
            'answer': 'Defines transactional boundaries so method executes within a DB transaction; supports rollback on exceptions.',
            'rubric': 'Mention start/commit/rollback semantics and propagation/readOnly options.'
        }
    ]
    return {
        'title': title,
        'instructions': 'Answer all questions. Time limit: 45 minutes.',
        'duration_minutes': 45,
        'difficulty': 'hard',
        'tags': tags,
        'questions': questions
    }
