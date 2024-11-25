from evaluation import evaluate_chunks


def optimize_chunk_size(text, strategy, callback=None):
    """
    Dynamically determines the best chunk size iteratively with a callback for updates.

    Args:
        text (str): Input text.
        strategy (function): The chunking strategy to optimize.
        callback (function, optional): Function to call after each iteration.

    Returns:
        dict: Best chunk size and evaluation results.
    """
    initial_size = len(text.split()) // 10 or 50  # Start with 10% of the text or minimum 50 tokens
    step_size = 50  # Step size for incrementing chunk size
    tolerance = 0.01  # Stop optimization when improvements are small

    best_chunk_size = initial_size
    best_evaluation = None
    iteration = 1

    while True:
        chunks = strategy(text, chunk_size=best_chunk_size)
        evaluation = evaluate_chunks(chunks)

        # Use the callback to pass iteration updates
        if callback:
            callback(iteration, evaluation, best_chunk_size)

        if (
            best_evaluation is None or 
            (evaluation["avg_similarity"] > best_evaluation["avg_similarity"] and
             evaluation["chunk_count"] <= best_evaluation["chunk_count"])
        ):
            # Update best evaluation if conditions are met
            best_evaluation = evaluation
            best_chunk_size += step_size
        else:
            break

        iteration += 1

    return {
        "best_chunk_size": best_chunk_size - step_size,
        "best_evaluation": best_evaluation
    }