# Benchmarks

## LongMemEval retrieval (Track A)

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is a benchmark for
chat-assistant memory. Each question comes with a haystack of past chat
sessions, and the sessions that hold the answer are labelled.

This run measures **retrieval only**. Each question's past messages are
ranked against the question, and the run checks whether the labelled
evidence comes out near the top. No LLM answers anything.

- **What it covers:** the layer every KIPP recall rests on, exact cosine
  search over a local embedding model.
- **What it doesn't cover:** answer accuracy. The figures vendors publish for
  LongMemEval (Mem0, Zep, Letta and others) are end-to-end question
  answering scored by an LLM judge, so they **are not comparable** with the
  numbers here.

Harness: [`bench/longmemeval/run_retrieval.py`](../bench/longmemeval/run_retrieval.py).
Raw results: [`bench/longmemeval/results/`](../bench/longmemeval/results/). Issue #81.

### Setup

- **Dataset:** LongMemEval_S, cleaned release (MIT licence). 500 questions;
  the 30 abstention questions have no evidence, so 470 are scored. Each
  haystack has about 50 sessions and about 500 messages.
- **Index units:** rounds, each a user message plus the assistant reply that
  follows it. Two texts per round:
  - **round:** `User: …\n\nAssistant: …`, exactly what `kipp-capture` stores
    for a turn;
  - **user:** the user message alone.
- **Scoring:** a session's score is its best round's score.
  - `session@k`: an answer session is in the top k sessions, out of about 50.
  - `all@k`: every answer session is in the top k.
  - `round@k`: a round holding a labelled evidence message is in the top k
    rounds, out of about 500. This is the strictest measure.
- **Models**, each with its documented query instruction:

  | Model | Size | Window |
  |---|---|---|
  | `all-MiniLM-L6-v2` (KIPP default) | 22M | 256 tokens |
  | `BAAI/bge-small-en-v1.5` | 33M | 512 tokens |
  | `BAAI/bge-base-en-v1.5` | 109M | 512 tokens |

- **Hardware:** CPU only, an AMD Ryzen 7 5825U.

### Results (all 470 questions)

| Model / text | session@1 | session@5 | all@10 | round@5 | round@10 |
|---|---|---|---|---|---|
| MiniLM / round | 0.874 | 0.977 | 0.951 | 0.817 | 0.900 |
| MiniLM / user | 0.847 | 0.966 | 0.932 | 0.832 | 0.900 |
| bge-small / round | 0.857 | 0.970 | 0.945 | 0.834 | 0.919 |
| bge-small / user | 0.885 | 0.979 | 0.964 | 0.885 | 0.932 |
| bge-base / user | 0.904 | 0.979 | 0.979 | 0.904 | 0.943 |

`round@5` by question type, which shows where the misses are:

| Type (n) | MiniLM round | MiniLM user | bge-base user |
|---|---|---|---|
| knowledge-update (72) | 0.917 | 0.944 | 0.972 |
| multi-session (121) | 0.802 | 0.860 | 0.950 |
| single-session-assistant (56) | 0.964 | 0.857 | 0.857 |
| single-session-preference (30) | 0.733 | 0.700 | 0.800 |
| single-session-user (64) | 0.828 | 0.922 | 0.969 |
| temporal-reasoning (127) | 0.724 | 0.717 | 0.835 |

Truncation: 74% of round texts run past MiniLM's 256-token window, and 39%
past bge-small's 512. Only 1% of user messages do.

Embedding time for the about 90,000 unique texts, on that CPU:

| Run | Time |
|---|---|
| MiniLM / user | 8 min |
| MiniLM / round | 38 min |
| bge-small / user | 17 min |
| bge-base / user | 57 min |
| bge-small / round | 2 h 20 min |

### What it says

- **Finding the right session is mostly solved at this scale.** Every
  configuration puts an answer session in the top 5 of about 50 for at
  least 97% of questions. The headroom is in ranking the exact message
  (`round@k`) and in getting every session for multi-session questions.
- **A bigger model buys a few points, at a steep cost.**
  - **Gain:** going from MiniLM to bge-base on user text lifts `round@5`
    from 0.83 to 0.90, and `session@1` from 0.85 to 0.90.
  - **Cost:** about 7× the embedding time and 5× the model size.
  - **Verdict:** the gain is real, but not enough on its own to change
    KIPP's default.
- **What to embed matters as much as the model.**
  - **Answer in the assistant's reply:** the round text wins (0.96 against
    0.86 on single-session-assistant), because user text can't contain it.
  - **Answer stated by the user:** user text wins (0.92 against 0.83 on
    single-session-user), because the long assistant reply dilutes the
    vector and is mostly truncated anyway.
  - **Implication:** a round embedded both ways, or the user message kept
    separate from the reply, would likely beat either. That's a cheaper
    lever than a bigger model.
- **Truncation hurts less than expected.** bge-small reads twice as much of
  each round as MiniLM, but scores no better on round text. Evidence tends
  to sit near the start of a round.
- **Weakest types:** temporal reasoning and preference. Both ask about
  things the question doesn't name (a date, or a taste implied by an
  earlier request), and cosine similarity alone ranks those poorly.

### Caveats

- **Small haystacks.** Ranking out of about 50 sessions is easier than
  ranking a real memory store that has grown for months. `round@k` (about
  500 candidates) is the more honest figure.
- **Raw messages, not entities.** This ranks raw messages. KIPP's
  `memory_search` ranks consolidated entity facts, which the LLM extraction
  step writes from events. Track B (end-to-end, with extraction and an LLM
  judge) would measure that. It isn't run yet.
- **No reranking.** Search is pure cosine. KIPP's search also fuses keyword
  matches over facts, which this harness doesn't model.

### Reproducing

```bash
mkdir -p ~/.cache/kipp-bench
curl -L -o ~/.cache/kipp-bench/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python bench/longmemeval/run_retrieval.py --model minilm --unit round
```

- **Arguments:** `--model` is `minilm`, `bge-small` or `bge-base`; `--unit`
  is `round` or `user`.
- **Caching:** embeddings are cached under `~/.cache/kipp-bench/emb/`, so a
  stopped run resumes.
- **Smoke test:** `--limit N` scores the first N questions and writes
  nothing.
