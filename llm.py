
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Small Qwen instruction model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = None
model = None


def load_qwen():

    global tokenizer, model

    if model is None:

        print("Loading Qwen model...")
        print("The first download may take some time.")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model.eval()

        print("Qwen model loaded successfully!")

    return model, tokenizer


def generate_answer(prompt):

    model, tokenizer = load_qwen()

    messages = [
        {
            "role": "system",
            "content": (
                "You are FractureAI, an educational bone health "
                "assistant. Give clear, simple and evidence-based "
                "information. Do not claim to diagnose fractures. "
                "Encourage professional medical assessment."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


if __name__ == "__main__":

    print("Testing FractureAI Qwen...")

    question = "What is a bone fracture?"

    answer = generate_answer(question)

    print("\nQWEN ANSWER:")
    print(answer)
