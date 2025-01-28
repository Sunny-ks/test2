from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# Load the SentenceTransformer model
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)

# Optional: Adjust the maximum sequence length
model.max_seq_length = 8192

# Define your ground truth and generated response columns
ground_truth = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

generated_responses = [
    "Women aged 19 to 70 need an average of 46 grams of protein per day according to the CDC. If you are pregnant or training for an event, this requirement increases. Refer to the chart for more details.",
    "A summit is the highest point of a mountain, the topmost level, or a meeting between leaders of governments."
]

# Encode both the ground truth and generated responses
truth_embeddings = model.encode(ground_truth, prompt_name="ground_truth")
generated_embeddings = model.encode(generated_responses, prompt_name="generated")

# Compute similarity scores using different metrics
cosine_scores = cosine_similarity(truth_embeddings, generated_embeddings)
euclidean_scores = euclidean_distances(truth_embeddings, generated_embeddings)
manhattan_scores = manhattan_distances(truth_embeddings, generated_embeddings)

# Print similarity scores for each pair
for i in range(len(ground_truth)):
    print(f"Pair {i+1}:")
    print(f"  Cosine Similarity: {cosine_scores[i][i] * 100:.2f}%")
    print(f"  Euclidean Distance: {euclidean_scores[i][i]:.2f}")
    print(f"  Manhattan Distance: {manhattan_scores[i][i]:.2f}\n")
