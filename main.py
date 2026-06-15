from similarity import cosine_similarity, jaccard_similarity
from candidate_generator import generate_candidates
from scorer import score_candidates
from evaluator import precision_at_k

# Sample Dataset
USER_DATA = {
    "Alice": {"Toy Story": 5.0, "Finding Nemo": 5.0, "Up": 4.5, "Interstellar": 4.0, "The Matrix": 3.5},
    "Bob": {"Interstellar": 5.0, "The Matrix": 5.0, "Iron Man": 4.5, "The Dark Knight": 4.0, "Avatar": 3.5},
    "Charlie": {"Toy Story": 4.0, "The Notebook": 5.0, "La La Land": 4.5, "About Time": 4.0},
    "David": {"Iron Man": 5.0, "The Avengers": 4.5, "The Notebook": 3.0, "Titanic": 4.0, "Gladiator": 4.5},
    "Eve": {"Toy Story": 4.5, "Finding Nemo": 4.0, "Up": 5.0, "Interstellar": 4.5, "Avatar": 4.0},
    "Frank": {"Iron Man": 4.0, "The Dark Knight": 5.0, "The Avengers": 4.0, "Interstellar": 4.0, "The Matrix": 4.5},
    "Grace": {"The Dark Knight": 5.0},
    "Henry": {"The Notebook": 4.5, "La La Land": 5.0, "Titanic": 3.5, "About Time": 4.5}
}

GROUND_TRUTH = {
    "Alice": ["Monsters Inc.", "Spider-Man: Into the Spider-Verse", "The Avengers"],
    "Bob": ["Blade Runner 2049", "Inception", "Gladiator"],
    "Charlie": ["Titanic", "Up", "Monsters Inc."],
    "David": ["The Dark Knight", "La La Land", "Blade Runner 2049"],
    "Eve": ["Monsters Inc.", "Spider-Man: Into the Spider-Verse", "The Matrix"],
    "Frank": ["Inception", "Avatar", "Gladiator"],
    "Grace": ["Iron Man", "The Avengers", "The Matrix"],
    "Henry": ["Toy Story", "Up", "La La Land"]
}

def run_simulation(target_user="Alice", similarity_type="cosine", K=3, N=5, eval_k=3):
    print("="*60)
    print(f"RECOMMENDATION ENGINE SIMULATION FOR USER: {target_user}")
    print(f"Settings: Metric={similarity_type.upper()}, Neighbors={K}, Top={N}, Eval@K={eval_k}")
    print("="*60)
    
    # ----------------------------------------------------
    # STEP 1: SIMILARITY CALCULATION
    # ----------------------------------------------------
    print("\n[STEP 1] Similarity Calculator")
    print("-" * 35)
    similarities = []
    target_profile = USER_DATA[target_user]
    
    for other_user, other_profile in USER_DATA.items():
        if other_user == target_user:
            continue
            
        if similarity_type == "cosine":
            sim = cosine_similarity(target_profile, other_profile)
        else:
            s1 = set(target_profile.keys())
            s2 = set(other_profile.keys())
            sim = jaccard_similarity(s1, s2)
            
        similarities.append((other_user, round(sim, 4)))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for user, sim in similarities:
        is_neighbor = "*" if sim > 0 and similarities.index((user, sim)) < K else " "
        print(f" {is_neighbor} Sim({target_user}, {user:<8}) = {sim:.4f}")
        
    # Get top K neighbors
    neighbors = [s for s in similarities if s[1] > 0][:K]
    print(f"\nSelected Top {K} Neighbors: {', '.join([n[0] for n in neighbors])}")
    
    # ----------------------------------------------------
    # STEP 2: CANDIDATE GENERATION
    # ----------------------------------------------------
    print("\n[STEP 2] Candidate Generator")
    print("-" * 35)
    candidates = generate_candidates(target_user, USER_DATA, similarity_type, K)
    print(f"Generated Candidates (unseen items from neighbors):")
    print(f" {candidates}")
    
    # ----------------------------------------------------
    # STEP 3: CANDIDATE SCORER
    # ----------------------------------------------------
    print("\n[STEP 3] Scorer & Ranker")
    print("-" * 35)
    recommendations = score_candidates(target_user, candidates, USER_DATA, similarity_type, N)
    print("Top Recommendations:")
    for idx, (item, score) in enumerate(recommendations, 1):
        print(f" {idx}. {item:<25} (Predicted Score: {score:.4f}★)")
        
    # ----------------------------------------------------
    # STEP 4: EVALUATOR
    # ----------------------------------------------------
    print("\n[STEP 4] Evaluator")
    print("-" * 35)
    user_ground_truth = GROUND_TRUTH.get(target_user, [])
    precision = precision_at_k(recommendations, user_ground_truth, eval_k)
    
    print(f"Ground Truth (Movies actually liked later): {user_ground_truth}")
    print(f"Top {eval_k} Recs Evaluated: {[r[0] for r in recommendations[:eval_k]]}")
    
    # Calculate hits
    hits = [r[0] for r in recommendations[:eval_k] if r[0] in user_ground_truth]
    print(f"Hits: {hits}")
    print(f"Precision@{eval_k} = {len(hits)} / {eval_k} = {precision * 100:.1f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Run a test for Alice using Cosine Similarity
    run_simulation(target_user="Alice", similarity_type="cosine", K=3, N=5, eval_k=3)
    # Run a test for Bob using Jaccard Similarity
    run_simulation(target_user="Bob", similarity_type="jaccard", K=3, N=5, eval_k=3)
