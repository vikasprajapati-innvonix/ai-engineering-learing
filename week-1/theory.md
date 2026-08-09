# Week 1 — AI Engineering Foundations: Notes

Overview-level summary of everything covered this week. Tokenizer and Transformer sections include detailed examples since those are the core mechanics everything else builds on.

---

## 1. What is AI Engineering?

Building applications **on top of** existing LLMs, rather than training models from scratch. Sits between traditional software engineering and ML research.

**Key terms:**
- **Generative AI** — the capability to create new content (text, images, code)
- **AI Agent** — an LLM that pauses, uses a tool, and completes **one bounded task** (single think → act → respond loop)
- **Agentic AI** — an LLM system given an open-ended **goal**, planning and chaining multiple steps/tools autonomously
- **AI Engineer** — the person building these systems

**Core distinction:** traditional software is deterministic (same input → same output). LLM apps are probabilistic (same input → possibly different output). This is the root cause of most challenges unique to this field (evaluation, guardrails, prompt design).

**Prompt engineering ≠ AI Engineering.** Real apps also need context management, reliability handling, retrieval, tool integration, evaluation, and cost/latency control.

---

## 2. Tokens & Tokenization (Detailed)

LLMs don't read words — they read **tokens**, chunks of text mapped to numeric IDs via a fixed vocabulary.

### Why not whole words or single characters?
| Approach | Problem |
|---|---|
| Whole word | Vocabulary explodes (500K+ words, names, slang) — still hits unknown words |
| Single character | Sequences become very long — wastes context window, harder to learn meaning |
| **Subword (BPE)** | Balances both — common words = 1 token, rare words = split into pieces |

### How BPE (Byte Pair Encoding) builds its vocabulary — step by step

Training corpus (toy example): `"low low low lower lowest"`

```
Step 1 — Start at character level:
   l, o, w, e, r, s, t, (space)

Step 2 — Count most frequent adjacent character pairs:
   "l"+"o" appears most often (in low, lower, lowest)

Step 3 — Merge the most frequent pair into one token:
   "l" + "o" → "lo"

Step 4 — Repeat merging on the next most frequent pair:
   "lo" + "w" → "low"

Step 5 — Keep repeating (on real data: billions of merges)
   until vocabulary reaches target size (e.g., 100,000 tokens)
```

Result: common patterns become single tokens; rare/unseen text falls back to smaller pieces (never fails, always has a fallback down to raw characters).

### Applying it to new text (inference time)

```
Input: "unbelievable"
Apply learned merge rules in order:
  → ["un", "believ", "able"]

Input: "zxqoblorp" (never seen in training)
No merges match most of it, so it falls back to small pieces:
  → ["z", "x", "q", "ob", "lo", "r", "p"]  (illustrative — exact split depends on real vocab)
```

### Token → Number → Meaning

```
Vocabulary = a lookup table built during training:
   "un"     → ID 4245
   "believ" → ID 12893
   "able"   → ID 522

"unbelievable" → ["un","believ","able"] → [4245, 12893, 522]
```

The ID alone is meaningless — it's just an index. Real "meaning" comes later via **embeddings** (see Transformer section).

### Practical consequences
- API cost = billed per token (input + output), not words
- Context window limits = measured in tokens
- Non-English text often costs **more tokens** for the same sentence (tokenizers are trained on English-majority data)
- Explains why LLMs sometimes miscount letters — they don't see individual characters, they see tokens

**Rule of thumb (English):** 1 token ≈ 4 characters ≈ ¾ of a word.

---

## 3. Context Window

Max number of tokens a model can hold at once:
```
Context Window = System Prompt + Conversation History + Current Input + Output
```

Exists because processing cost grows steeply with sequence length (attention compares every token to every other token).

**Important correction:** exceeding the limit does **not** get silently trimmed by the API — it throws an **error**. Managing history (sliding window, summarization, truncation, RAG) is the **developer's job**, not automatic.

---

## 4. Temperature, Top-p, Max Tokens

At every step, the model outputs a probability distribution over its vocabulary for the next token. These three parameters control how a token gets picked from that distribution.

| Parameter | Controls | Low | High |
|---|---|---|---|
| **Temperature** | Randomness of selection | Deterministic, safe, repetitive | Creative, varied, more hallucination-prone |
| **Top-p** | Size of the candidate pool (cumulative probability cutoff) | Narrow — only top likely tokens | Wide — includes unusual tokens |
| **Max tokens** | Hard ceiling on output length | Short, may cut off mid-sentence | Long, room to fully answer |

**Use low temperature for:** factual Q&A, code, math, extraction.
**Use high temperature for:** creative writing, brainstorming.

---

## 5. Transformer Architecture & Attention (Detailed)

### The problem it solved
Older models (RNNs) read text one word at a time, in order, relying on fading memory — struggled with long-distance relationships and couldn't be parallelized (slow to train).

**Transformers (2017, "Attention Is All You Need")** let the model look at **all tokens simultaneously** and decide how much each token should relate to every other token — regardless of distance.

### Attention — worked example

```
Sentence: "The cat sat on the mat because it was tired."
Question: what does "it" refer to?

Attention weights when processing "it":
  "The"     → low
  "cat"     → HIGH   ← model focuses here
  "sat"     → low
  "mat"     → medium
  "tired"   → medium
  (others)  → low
```

The model learned this relevance from patterns across massive training data — not hardcoded.

### Query, Key, Value (the mechanism, via analogy)

| Vector | Analogy | Role |
|---|---|---|
| Query (Q) | "What am I looking for?" | What this token needs to understand itself |
| Key (K) | "What do I offer?" | Searchable label for every token |
| Value (V) | "Here's my content" | Actual info contributed if selected |

```
"it" generates a Query → compared against every token's Key
→ produces attention weights (cat=0.7, mat=0.2, others≈0.1)
→ weighted blend of Values → "it" now has a context-aware representation
```

### Multi-head attention
Runs attention multiple times in parallel ("heads"), each potentially specializing in a different relationship type (grammar, coreference, topic) — this specialization **emerges** during training, isn't hand-programmed.

### Full flow

```
Text → Tokenization → Token IDs
     → Embedding Lookup (base meaning as a vector)
     → + Positional Encoding (adds word ORDER info)
     → [Multi-Head Self-Attention → Feed-Forward] × N layers
     → Probability distribution over vocabulary for next token
     → Sampling (temperature/top-p applied here) → next token chosen
     → repeat until max_tokens or stop token
```

### Why positional encoding is needed
Attention treats input as a set, not a sequence — no inherent sense of order. "Dog bites man" vs "Man bites dog" would look identical without it. Positional encoding adds a "this is my position" signal to each token.

### Static vs. Contextual Embeddings

| | Static Embedding (Word2Vec, GloVe) | Contextual Embedding (Transformers) |
|---|---|---|
| One word, one vector? | Yes — always identical everywhere | No — recalculated per sentence |
| Handles multiple meanings? | No — blends all meanings into one | Yes — shaped by surrounding context |
| Example | "bank" = same vector in "river bank" and "money bank" | "bank" gets a different vector in each sentence, via attention |

### Why it won over RNNs
Parallelizable (fast to train on GPUs), handles long-distance relationships directly via attention — this scalability is *why* today's massive models (GPT, Claude, Gemini) became possible.

---

## 6. Hallucination

**Not "the model lying"** — it has no concept of true/false. It's a next-token predictor: at every step it picks the statistically most probable next token, given training data patterns.

- Strong training data → confident, usually correct output
- Sparse/no training data → still outputs *something* fluent and confident-sounding, because there's no "I don't know" token that wins by default

**Triggers that increase hallucination:** high temperature, obscure topics, very recent events (outside training data), requests for exact numbers/citations, leading/loaded questions.

**Important nuance:** low temperature reduces *randomness*, not *incorrectness* — a wrong "most probable" answer just becomes consistently wrong instead of randomly wrong. Fixed by grounding techniques like **RAG** (feeding real documents at request time), not by parameter tuning.

---

## 7. Reasoning Models vs. Chat Models

| | Chat Model | Reasoning Model |
|---|---|---|
| Approach | Answers fairly directly | Generates internal reasoning steps first, then answers |
| Speed/Cost | Faster, cheaper | Slower, more expensive (reasoning tokens) |
| Best for | Simple Q&A, chat, creative writing | Multi-step math, logic, coding, planning |

**Key distinction:** reasoning models are better *thinkers*, not better *knowers*. They reduce logic/multi-step errors — they do **not** fix hallucination caused by missing knowledge (more reasoning steps can't invent a real fact that was never in training data).

---

## 8. LLM Provider Comparison Framework

Six dimensions to evaluate any provider (not a static table — this goes stale fast, re-verify current state):

1. **Response quality / reasoning** — test with hard, ambiguous, self-critique prompts
2. **Speed** — latency (time to first token) vs. throughput (tokens/sec); e.g. Groq uses custom LPU chips for very fast inference regardless of model quality
3. **Coding ability** — test debugging, not just generation
4. **Context length** — advertised max ≠ effective max ("lost in the middle" effect)
5. **Cost** — priced per token; output tokens usually cost 3–5x more than input tokens
6. **Specialization/philosophy** — general purpose vs. safety-focused vs. multimodal vs. inference-speed-focused vs. open-weight/self-hostable

**Fair comparison method:** same prompt, same parameters, multiple prompt categories (factual/coding/creative/reasoning), measure your own timing — don't trust marketing claims.

---

## 9. LangChain — Why It Exists

**Problem:** every LLM provider has a different SDK (different function names, parameter structures, response formats) — makes switching/comparing providers a rewrite, not a config change.

**What LangChain does:** provides one consistent interface across providers.

```python
model_openai = ChatOpenAI(model="gpt-4")
model_claude = ChatAnthropic(model="claude-sonnet-4-6")
# same .invoke() call works for both
```

**Philosophy:** real LLM apps are rarely one call — they're **chains** of steps (retrieve → build prompt → call LLM → parse output). LangChain provides pre-built building blocks (prompt templates, memory, retrievers, output parsers) for these common patterns.

**Honest trade-off:**
- ✅ Good for: provider flexibility, complex chains, standard patterns (RAG, agents, memory)
- ⚠️ Criticism: over-abstracted for simple use cases, harder to debug (errors happen inside the framework layer), a single direct API call can be simpler for basic single-provider apps

---

## Quick Reference Table — Full Pipeline

```
Prompt (text)
   → Tokenizer (BPE) splits into tokens → converts to token IDs
   → Embedding lookup → each token becomes a vector
   → + Positional encoding → adds order info
   → Transformer layers (multi-head attention + feed-forward) × N
   → Probability distribution over vocabulary
   → Temperature / Top-p shape the sampling
   → Next token selected → repeat until max_tokens or stop
```