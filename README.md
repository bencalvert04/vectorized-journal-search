# Personal Journal Vector Search

My dad has 20 years of journaling every day stored on his computer. I was inspired by him to build a semantic search engine that could search through my own apple journal entries. Soon, I'd like to use it to be able to help my dad search through his own entries and especially, to allow an LLM to remember his experiences and analyze patterns in his thinking. The notebook walks through learning to understand and build a vector store — starting with a naive letter-frequency embedder and progressing to a real sentence embedding model — then uses it to search through my Apple Journal exports.

## What it does

1. **Parse** — Strips Apple Journal HTML exports down to plain text using BeautifulSoup.
2. **Embed** — Converts each journal entry into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`. Every vector is L2-normalized so cosine similarity reduces to a dot product.
3. **Store** — Saves the full vector matrix as a `.npy` file and the raw texts as JSON. Loading the whole matrix at once means search never needs a Python loop.
4. **Search** — Multiplies the query vector against the entire matrix in one NumPy operation (`vectors @ query_vector`), ranks by cosine similarity, and returns the top entries.
5. **Deduplicate** — Before adding a new entry, `threshold_check()` flags anything already in the store with similarity > 0.95 and asks for confirmation.
6. **Filtered search** — `search_v2()` lets you narrow results by year and month before doing semantic ranking, useful when you remember roughly *when* something happened.

## Evolution of the embedder

The notebook keeps the old versions around as commented-out code so you can see the progression:

| Version | Approach | Dimension | Notes |
|---------|----------|-----------|-------|
| v1 | Letter-frequency histogram | 26 | No word order, no semantics |
| v2 | `all-MiniLM-L6-v2` via SentenceTransformers | 384 | Captures meaning; current version |

## Project layout

```
vector_store.ipynb      # All code and explanation lives here
AppleJournalEntries/    # Raw HTML exports from Apple Journal (not tracked)
text_entries/           # Parsed plain-text entries, one .txt per day (not tracked)
texts.json              # Text corpus loaded at search time (not tracked)
vectors.npy             # Saved embedding matrix (not tracked)
```

The data files are excluded from version control because they contain private journal content.

## Setup

```bash
pip install numpy sentence-transformers beautifulsoup4
```

Then open `vector_store.ipynb` in Jupyter and run the cells in order.

To use your own journal data, export entries from Apple Journal as HTML files, place them in `AppleJournalEntries/Entries/`, and run the `folder_converter()` cell. The notebook will parse them, embed each entry, and save the vector store.

## Key ideas worth understanding

- **Why normalize vectors?** Once every vector has length 1, `a · b = cos(θ)`. This means the most-similar entry is just `argmax(matrix @ query)` — no division needed.
- **Why a matrix instead of a list?** A single matrix multiply (`vectors @ query_vector`) computes all cosine similarities at once using optimized BLAS routines. A Python `for` loop over the same data is orders of magnitude slower.
- **Why 384 dimensions?** That's the output size of `all-MiniLM-L6-v2`. It's a distilled BERT model trained specifically to produce good sentence embeddings at a small size.
