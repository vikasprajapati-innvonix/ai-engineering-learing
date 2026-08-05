import os

from dotenv import load_dotenv
from groq import Groq

# Load the .env file
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write a one-paragraph story about a robot learning to paint.",
        }
    ],
    model="llama-3.3-70b-versatile",
    # Example:
    # Prompt: "Complete this sentence: The sky is ..."
    # The model predicts the next token with probabilities:
    # - blue    → 70%
    # - clear   → 20%
    # - falling → 10%
    # - purple  → 5%
    # Temperature controls how random the selection is:
    # - Low temperature (0.1–0.3): More deterministic and factual outputs.
    # - High temperature (0.7–1.5): More diverse and creative outputs.
    # top_p (nucleus sampling) controls which predicted tokens are eligible for selection.
    # - top_p = 1.0 → Consider all predicted tokens.
    # - top_p = 0.1 → Consider only the highest-probability tokens.
    # - top_p = 0.9 → Keep adding the most likely tokens until their cumulative
    #   probability reaches 90%.
    # For top_p = 0.9:
    # - blue  = 70% (included)
    # - clear = 20% (70% + 20% = 90%, included)
    # - Cut off here.
    # - "falling" and "purple" are excluded because the cumulative probability
    #   threshold has already been reached.
    # temperature decide the randomness of the model's output
    # low temperature (0.1 - 0.3) = deterministic output (factual), high temperature (0.7 - 1.5) = more random (creativity)
    temperature=1.6,
    # top_p controls the diversity of the model's output
    # this decide how many tokens to consider for the output
    # top_p=1.0 means consider all tokens, top_p=0.1 means consider only the most likely tokens
    # E.g top_p=0.7 means consider the top 70% of the most likely tokens
    top_p=0.9,
    # max_tokens controls the maximum number of tokens to generate
    max_tokens=150,
)

print(chat_completion.choices[0].message.content)
