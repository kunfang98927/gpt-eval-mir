import time
import tiktoken
from openai import OpenAI

# Configuration constants
MAX_TOKENS_CONTEXT = 16385  # Context limit for GPT-3.5-turbo-16k-0613
API_KEY = "YOUR_API_KEY"


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo-16k-0613") -> int:
    """
    Count the number of tokens in a given text based on the specified model's encoding.

    Parameters:
        text (str): The text to tokenize.
        model_name (str): The model whose encoding is used for tokenization.

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def initialize_client(api_key: str) -> OpenAI:
    """
    Initialize and return the OpenAI client using the provided API key.

    Parameters:
        api_key (str): OpenAI API key.

    Returns:
        OpenAI: Initialized OpenAI client.
    """
    return OpenAI(api_key=api_key)


def create_message(prompt: str) -> list:
    """
    Construct the message payload for the OpenAI API call.

    Parameters:
        prompt (str): The user prompt.

    Returns:
        list: A list of message dictionaries formatted for the OpenAI API.
    """
    return [
        {"role": "system", "content": "You are a specialist in data pattern analysis."},
        {"role": "user", "content": prompt},
    ]


def call_chat_api(prompt: str, model_name: str = "gpt-3.5-turbo-1106") -> tuple:
    """
    Call the OpenAI chat completion API with retry logic for rate-limiting errors.

    Parameters:
        prompt (str): The user prompt.
        model_name (str): Model name for the chat completion.

    Returns:
        tuple: The result from the API and the messages used in the API call.
    """
    client = initialize_client(API_KEY)
    messages = create_message(prompt)

    # Calculate total tokens for input
    token_count = sum(count_tokens(m["content"], model_name) for m in messages)
    print(f"Total input tokens: {token_count}")

    result = ""
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=MAX_TOKENS_CONTEXT - token_count - 500,
                temperature=0.0,
                stream=True,
            )
            # Collect response content from the streamed completion
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    result += content
            break
        except Exception as e:
            error_message = str(e).lower()
            if (
                "requests per min (rpm)" in error_message
                or "tokens per min (tpm)" in error_message
            ):
                print("Rate limit exceeded. Waiting for 5 seconds...")
                time.sleep(5)
            else:
                print("Error encountered. Exiting...")
                print(e)
                return result, messages

    return result, messages
