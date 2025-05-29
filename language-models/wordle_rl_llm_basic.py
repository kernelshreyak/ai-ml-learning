import os
import re
from openai import OpenAI
import time

# Detailed system prompt for the Wordle assistant
SYSTEM_PROMPT = """
You are playing Wordle, a challenging 5-letter English word guessing game. Follow these guidelines precisely:

1. **Objective**
   - Guess the secret 5-letter target word in up to 6 attempts.
   - Each guess must be a valid English word of exactly five letters.

2. **Feedback Symbols**
   - **✓** : Letter is in the word and in the CORRECT position.
   - **–** : Letter is in the word but in the WRONG position.
   - **✗** : Letter is NOT in the word.

3. **Strategy**
   - Prioritize high-frequency letters early to narrow possibilities.
   - Avoid repeating letters that have been marked ✗ (absent).
   - Use positional feedback to eliminate or confirm letter placements.

4. **Response Format**
   - **Thoughts**: Wrap your step-by-step reasoning in `<think>...</think>` tags. Explain why you choose each letter.
   - **Guess**: After reasoning, output exactly one 5-letter word wrapped in `<guess>...</guess>` tags.
   - Do **not** include any additional commentary outside these tags.

5. **Error Handling**
   - If your candidate is not a valid 5-letter English word, respond with `<error>Invalid guess</error>`.

Example:
<think>
I know 'A' and 'R' are present but in wrong spots, and 'T' is correct at position 3...
</think>
<guess>arise</guess>
"""

# Initialize OpenAI client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_guess(guess: str, target: str) -> list:
    """
    Evaluate a guess against the target word.
    Returns a list of feedback for each letter:
    - 'correct': letter in correct position
    - 'present': letter in word but wrong position
    - 'absent': letter not in word
    """
    feedback = [''] * len(guess)
    target_chars = list(target)

    # First pass: mark correct letters
    for i, char in enumerate(guess):
        if char == target[i]:
            feedback[i] = 'correct'
            target_chars[i] = None

    # Second pass: mark present or absent
    for i, char in enumerate(guess):
        if feedback[i]:
            continue
        if char in target_chars:
            feedback[i] = 'present'
            target_chars[target_chars.index(char)] = None
        else:
            feedback[i] = 'absent'

    return feedback


def next_turn(history: list, feedbacks: list) -> str:
    """
    Generate the next 5-letter word guess using the detailed system prompt.
    """
    # Build the dynamic user prompt with history
    prompt = "You are playing Wordle. Based on the feedback so far, suggest the next guess.\n\n"
    for guess, fb in zip(history, feedbacks):
        visual = ['✓' if f == 'correct' else '–' if f == 'present' else '✗' for f in fb]
        prompt += f"Guess: {guess.upper()}, Feedback: {''.join(visual)}\n"
    prompt += "Next guess:"

    # LLM call using the new SYSTEM_PROMPT constant
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    full = response.choices[0].message.content.strip()
    # Expect format: <think>...</think><guess>word</guess>
    m = re.search(r"<guess>([a-zA-Z]{5})</guess>", full)
    if not m:
        # Optionally print full for debugging
        raise ValueError(f"Invalid or missing <guess> tag: '{full}'")
    return m.group(1).lower()


def play_wordle(target: str, N: int = 6):
    """
    Play a Wordle-like game for up to N turns or until solved.
    """
    history = []
    feedbacks = []

    for turn in range(1, N+1):
        try:
            guess = next_turn(history, feedbacks)
            time.sleep(2)
        except Exception as e:
            print(f"Error generating next guess: {e}")
            break

        print(f"Turn {turn}: Guessing '{guess.upper()}'")
        fb = evaluate_guess(guess, target)
        feedbacks.append(fb)
        history.append(guess)
        visual = ['✓' if f=='correct' else '–' if f=='present' else '✗' for f in fb]
        print(f"Feedback: {''.join(visual)}")

        if guess == target:
            print(f"Solved! The word was '{target.upper()}' in {turn} turns.")
            return

    print(f"Failed to guess the word within {N} turns. The word was '{target.upper()}'.")

if __name__ == "__main__":
    # Example usage
    target_word = input("Enter the target word (5 letters): ").strip().lower()
    if len(target_word) != 5 or not target_word.isalpha():
        print("Please provide a valid 5-letter word.")
    else:
        play_wordle(target_word, N=10)
