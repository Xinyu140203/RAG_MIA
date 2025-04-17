import openai
from openai import OpenAI
import os

def call_gpt4(query, api_key):
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {str(e)}")
        return "error"


def append_output(file_path, text):
    with open(file_path, 'a', encoding='utf-8') as f:
        cleaned = text.replace("\n", " ").strip()
        f.write(cleaned + "\n\n")


def read_inputs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip().split("\n\n")


if __name__ == "__main__":
    # Get API key from environment variable instead of hardcoding
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: No API key provided")
    
    # Use relative or configurable paths
    input_path = "input.txt"  # Replace with actual path
    output_path = "output.txt"  # Replace with actual path

    texts = read_inputs(input_path)
    for text in texts:
        word_count = len(text.split())
        perturbation_magnitude=0.03 
        replace_count = int(perturbation_magnitude * word_count) or 1

        prompt = (
            f"Replace {replace_count} key adjectives or adverbs in noticeable positions "
            f"with their antonyms in the following text, ensuring the modified text remains logically correct:\n"
            f"{text}\n"
            f"Return only the modified text."
        )

        result = call_gpt4(prompt, api_key)
        if result and result != "error":
            append_output(output_path, result)
        else:
            print("Failed to process text segment")

    print("Processing complete.")
