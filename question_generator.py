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
- Only output a valid JSON array. No explanation, no markdown, no extra text.

Output format:
[
  {
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": ""
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


def parse_mcq_response(raw_text: str, num_questions: int) -> list[dict]:
    # Parse and validate MCQ JSON response with multiple fallback strategies.

    def fix_json_escapes(text: str) -> str:
        # Fix invalid escape sequences like \e, \a, etc.
        return re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)

    def extract_json_candidate(text: str) -> str | None:
        # Try to extract a JSON array from messy text.
        # Try to find [...] block
        match = re.search(r'\[.*?]', text, re.DOTALL)
        return match.group() if match else None

    def validate_mcq(mcq: dict) -> dict | None:
        # Validate a single MCQ and fill missing fields if possible.
        if not isinstance(mcq, dict):
            return None

        # Must have a non-empty question
        question = mcq.get("question", "").strip()
        if not question:
            return None

        # Options must be a dict with at least 2 entries
        options = mcq.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            return None

        # Normalize option keys to uppercase
        options = {k.upper(): v for k, v in options.items()}

        # Answer: must exist and match one of the option keys
        answer = str(mcq.get("answer", "")).strip().upper()
        if not answer or answer not in options:
            # Fall back to first option key if answer is missing/invalid
            answer = next(iter(options))

        return {
            "question": question,
            "options": options,
            "answer": answer,
        }

    # Attempt 1: Parse as-is
    candidates = [raw_text.strip()]

    # Attempt 2: Fix bad escape sequences
    candidates.append(fix_json_escapes(raw_text.strip()))

    # Attempt 3: Extract [...] block, then fix escapes
    extracted = extract_json_candidate(raw_text)
    if extracted:
        candidates.append(extracted)
        candidates.append(fix_json_escapes(extracted))

    parsed = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break  # Stop at first successful parse
        except json.JSONDecodeError:
            continue

    if parsed is None or not isinstance(parsed, list):
        print("[WARNING] Could not parse any valid JSON from response.")
        print(f"[RAW OUTPUT]:\n{raw_text}\n")
        return []

    # Validate each MCQ and filter out invalid ones
    valid_mcqs = []
    for i, mcq in enumerate(parsed):
        if len(valid_mcqs) >= num_questions:
            break
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