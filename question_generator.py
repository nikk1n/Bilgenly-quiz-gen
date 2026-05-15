import json
import re

from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig
import torch

from config import settings

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load model (once, then reuse for all chunks)
model_id = "google/gemma-4-E4B-it"
HF_TOKEN=settings.HF_TOKEN
print("processor")
processor = AutoProcessor.from_pretrained(model_id,token=HF_TOKEN)
print("model")
model = Gemma4ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0", # Uses GPU by default
    token=HF_TOKEN
)
print("prompt")

SYSTEM_PROMPT = """You are an expert quiz maker. Generate multiple-choice questions (MCQs) based on the knowledge provided.

Rules:
- Write questions as if testing general subject knowledge, NOT as if summarizing a passage.
- NEVER reference "the text", "the passage", "the context", "as described", "according to", or any similar meta-phrases.
- Questions must be self-contained and make sense without having read any source material.
- Only output a valid JSON array. No markdown, no extra text.

Output format:
[
  {
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": ""
    "explanation": ""
  }
]"""


def generate_mcqs(text_chunk: str, num_questions: int = 1) -> list[dict]:
    #Generate MCQs for a single text chunk.
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Generate {num_questions} MCQs testing knowledge of the following topic:\n\n{text_chunk}"
                }
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    return parse_mcq_response(raw_text,num_questions)


def fix_brackets(text: str) -> str:
    """Fix common bracket errors in model-generated MCQ JSON."""

    # Fix missing closing brace in "options" block specifically:
    # Matches "options": { ... "D": "..." followed by }, or , instead of }}
    text = re.sub(
        r'("options"\s*:\s*\{[^}]*"[A-D]"\s*:\s*"[^"]*")\s*([,\]])',
        r'\1}}\2' if False else r'\1}\2',  # add missing }
        text
    )

    # More robust: count and balance braces/brackets
    def balance(s: str) -> str:
        open_braces = s.count('{') - s.count('}')
        open_brackets = s.count('[') - s.count(']')

        # Add missing closing braces/brackets at the end
        # (strip any trailing whitespace/newlines first)
        s = s.rstrip()
        s += '}' * max(0, open_braces)
        s += ']' * max(0, open_brackets)
        return s

    return balance(text)


def parse_mcq_response(raw_text: str) -> list[dict]:
    def fix_json_escapes(text: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)

    def extract_json_candidate(text: str) -> str | None:
        match = re.search(r'\[.*]', text, re.DOTALL)
        return match.group() if match else None

    def validate_mcq(mcq: dict) -> dict | None:
        if not isinstance(mcq, dict):
            return None
        question = mcq.get("question", "").strip()
        if not question:
            return None
        options = mcq.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            return None
        options = {k.upper(): v for k, v in options.items()}
        answer = str(mcq.get("answer", "")).strip().upper()
        if not answer or answer not in options:
            answer = next(iter(options))
        return {
            "question": question,
            "options": options,
            "answer": answer,
            "explanation": mcq.get("explanation", ""),
        }

    raw = raw_text.strip()
    extracted = extract_json_candidate(raw)

    # Build candidates: each strategy, then with bracket fix applied on top
    candidates = [raw, fix_json_escapes(raw)]
    if extracted:
        candidates += [extracted, fix_json_escapes(extracted)]

    # Apply bracket balancing to all existing candidates
    candidates += [fix_brackets(c) for c in candidates]

    parsed = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if parsed is None or not isinstance(parsed, list):
        print("[WARNING] Could not parse any valid JSON from response.")
        print(f"[RAW OUTPUT]:\n{raw_text}\n")
        return []

    valid_mcqs = []
    for i, mcq in enumerate(parsed):
        validated = validate_mcq(mcq)
        if validated:
            valid_mcqs.append(validated)
        else:
            print(f"[WARNING] Skipping invalid MCQ at index {i}: {mcq}")

    return valid_mcqs

if __name__=="__main__":
    # Process multiple chunks
    with open("result.txt", "r",encoding="utf-8") as f:
        text_chunks=f.readlines()
    all_results = {}
    for i, chunk in enumerate(text_chunks):
        print(chunk)
        print(f"\nProcessing chunk {i + 1}/{len(text_chunks)}...")
        mcqs = generate_mcqs(chunk, num_questions=1)
        all_results[f"chunk_{i + 1}"] = mcqs
        try:
            for j, mcq in enumerate(mcqs, 1):
                print(f"  Q{j}: {mcq['question']}")
                for opt, val in mcq['options'].items():
                    marker = " ✓" if opt == mcq['answer'] else ""
                    print(f"    {opt}) {val}{marker}")
        except KeyError:
            print("Something wrong with the response:")
            print(mcqs)


    # Save results
    with open("mcq_pdf_output.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to mcq_output_Q4.json")