import streamlit as st
from ragify.optimizer import optimize_chunk_size
from ragify.chunker import dynamic_chunking, adaptive_overlap_chunking, hierarchical_chunking

# Title of the app
st.title("Dynamic Chunk Sizing with Iterative Evaluation")

# File uploader
uploaded_file = st.file_uploader("Upload your text file:", type=["txt"])
if uploaded_file is not None:
    # Read the content of the file
    text = uploaded_file.read().decode("utf-8")
    st.write("**Uploaded File Content:**")
    st.text_area("", value=text[:500] + "...", height=150, disabled=True)  # Show preview

    if st.button("Optimize Chunking"):
        if not text.strip():
            st.error("File content is empty.")
        else:
            # Chunking strategies
            strategies = {
                "Dynamic Heuristic": dynamic_chunking,
                "Adaptive Overlap": adaptive_overlap_chunking,
                "Hierarchical": hierarchical_chunking
            }

            results = {}
            # Iterate through strategies
            for strategy_name, strategy_func in strategies.items():
                st.subheader(f"Optimizing: {strategy_name}")
                progress_bar = st.progress(0)
                strategy_results = []

                # Custom optimization logic to show iteration
                def optimization_callback(iteration, evaluation, best_chunk_size):
                    strategy_results.append({
                        "Iteration": iteration,
                        "Chunk Size": best_chunk_size,
                        "Avg Similarity": evaluation["avg_similarity"],
                        "Chunk Count": evaluation["chunk_count"]
                    })
                    # Update progress bar
                    progress_bar.progress(min(iteration * 10, 100))

                # Run optimization
                result = optimize_chunk_size(
                    text=text,
                    strategy=strategy_func,
                    callback=optimization_callback
                )

                results[strategy_name] = {
                    "Best Chunk Size": result["best_chunk_size"],
                    "Evaluation": result["best_evaluation"],
                    "Iterations": strategy_results
                }

                # Show iteration details
                for iteration_result in strategy_results:
                    st.write(f"Iteration {iteration_result['Iteration']}:")
                    st.write(f"- Chunk Size: {iteration_result['Chunk Size']}")
                    st.write(f"- Avg Similarity: {iteration_result['Avg Similarity']:.2f}")
                    st.write(f"- Chunk Count: {iteration_result['Chunk Count']}")
                    st.divider()

            # Determine the best strategy
            best_strategy = max(
                results.items(),
                key=lambda x: x[1]["Evaluation"]["avg_similarity"]
            )[0]

            # Display results summary
            st.subheader("Optimization Summary")
            for strategy_name, result in results.items():
                st.write(f"**{strategy_name}**")
                st.write(f" - Best Chunk Size: {result['Best Chunk Size']}")
                st.write(f" - Avg Similarity: {result['Evaluation']['avg_similarity']:.2f}")
                st.write(f" - Chunk Count: {result['Evaluation']['chunk_count']}")
                st.divider()

            st.success(f"The best strategy is **{best_strategy}**!")
