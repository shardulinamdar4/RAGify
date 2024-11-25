import tiktoken

def tokenize_text(text, model="gpt-3.5-turbo"):
    """
    Tokenizes the text using the tiktoken library.
    
    Args:
        text (str): The input text to tokenize.
        model (str): The OpenAI model name (affects tokenization logic).

    Returns:
        list: A list of token strings.
        int: Total token count.
    """
    try:
        # Load the tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model(model)
        
        # Tokenize the input text
        tokens = tokenizer.encode(text)
        
        # Return tokenized data and token count
        return tokens, len(tokens)
    except Exception as e:
        raise RuntimeError(f"Error during tokenization: {e}")
