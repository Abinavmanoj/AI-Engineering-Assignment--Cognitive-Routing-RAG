"""
phase1/router.py
----------------
Phase 1 – Vector-Based Persona Matching (The Router)

Steps:
1. Embed three bot personas using sentence-transformers (local, no API key needed).
2. Store the embeddings in an in-memory ChromaDB collection.
3. route_post_to_bots() embeds an incoming post, queries ChromaDB, and returns
   only the bots whose cosine similarity to the post exceeds `threshold`.

NOTE: ChromaDB uses cosine distance internally but returns a `distance` in
      [0, 2].  We convert to cosine similarity: sim = 1 - (distance / 2).
      Adjust THRESHOLD below if you get no matches with your embedding model.
"""

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Bot personas – these are the "souls" of each agent
# ---------------------------------------------------------------------------
BOT_PERSONAS = {
    "bot_a_tech_maximalist": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space "
        "exploration. I dismiss regulatory concerns."
    ),
    "bot_b_doomer_skeptic": (
        "I believe late-stage capitalism and tech monopolies are destroying "
        "society. I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c_finance_bro": (
        "I strictly care about markets, interest rates, trading algorithms, "
        "and making money. I speak in finance jargon and view everything "
        "through the lens of ROI."
    ),
}

# ---------------------------------------------------------------------------
# Embedding model – all-MiniLM-L6-v2 runs locally with no API key
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cosine similarity threshold.  Lower this (e.g. 0.30) if you get no matches;
# raise it (e.g. 0.50) for stricter routing.
THRESHOLD = 0.35


def _build_vector_store() -> chromadb.Collection:
    """
    Create an in-memory ChromaDB collection and insert persona embeddings.
    Returns the populated collection.
    """
    # In-memory client – nothing is persisted to disk
    client = chromadb.Client()

    # Use sentence-transformers locally (no API calls)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.create_collection(
        name="bot_personas",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # tells ChromaDB to use cosine distance
    )

    ids = list(BOT_PERSONAS.keys())
    docs = list(BOT_PERSONAS.values())

    collection.add(documents=docs, ids=ids)
    print(f"[Phase 1] ✅  Loaded {len(ids)} bot personas into vector store.")
    return collection


# Build once at import time so the store is ready for all queries
_COLLECTION = _build_vector_store()


def route_post_to_bots(post_content: str, threshold: float = THRESHOLD) -> list[dict]:
    """
    Embed `post_content` and return bots whose persona similarity exceeds `threshold`.

    Parameters
    ----------
    post_content : str   – The incoming social-media post text.
    threshold    : float – Minimum cosine similarity (0–1) to include a bot.

    Returns
    -------
    List of dicts: [{"bot_id": str, "persona": str, "similarity": float}, ...]
    Sorted by similarity descending.
    """
    # Query all 3 bots so we can inspect every similarity score
    results = _COLLECTION.query(
        query_texts=[post_content],
        n_results=len(BOT_PERSONAS),
        include=["documents", "distances"],
    )

    matched_bots = []

    # ChromaDB cosine distance ∈ [0, 2];  similarity = 1 - distance/2
    for bot_id, doc, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        similarity = 1.0 - distance / 2.0
        print(
            f"  [Router] {bot_id:<30}  similarity={similarity:.4f}  "
            f"({'✅ MATCHED' if similarity >= threshold else '❌ skipped'})"
        )
        if similarity >= threshold:
            matched_bots.append(
                {"bot_id": bot_id, "persona": doc, "similarity": similarity}
            )

    # Sort best match first
    matched_bots.sort(key=lambda x: x["similarity"], reverse=True)
    return matched_bots


# ---------------------------------------------------------------------------
# Quick smoke test – run this file directly to verify routing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits a new all-time high; ETF inflows surge past $500 million.",
        "Study shows social media is destroying teen mental health. Big Tech must be regulated.",
        "The Fed raised interest rates again; bond yields spike across the curve.",
    ]

    for post in test_posts:
        print(f"\n📨  Post: \"{post}\"")
        matches = route_post_to_bots(post)
        if matches:
            print(f"  ➡️  Routed to: {[m['bot_id'] for m in matches]}")
        else:
            print("  ➡️  No bots matched (try lowering the threshold).")
