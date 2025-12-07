# Evaluation Report: AmbedkarGPT RAG Pipeline

## Executive Summary
This report analyzes the performance of the AmbedkarGPT RAG system across three different chunking strategies (250, 550, and 900 characters). The evaluation was conducted using a test dataset of 25 questions against a corpus of Dr. B.R. Ambedkar's speeches.

**Key Finding:** The **Small Chunk strategy (250 characters)** is the optimal configuration. While retrieval accuracy was identical across all strategies, the small chunk size yielded higher answer fidelity (ROUGE-L and BLEU scores), indicating that more precise context helped the LLM generate more accurate responses.

---

## 1. Comparative Performance Data

The following metrics were gathered using the evaluation framework:

| Metric | Small (250) | Medium (550) | Large (900) |
| :--- | :---: | :---: | :---: |
| **Hit Rate** (Retrieval) | 0.84 | 0.84 | 0.84 |
| **MRR** (Ranking) | 0.78 | 0.78 | 0.78 |
| **ROUGE-L** (Structure) | **0.245** | 0.231 | 0.222 |
| **BLEU** (Precision) | **0.127** | 0.116 | 0.115 |
| **Cosine Sim** (Meaning) | 0.568 | **0.573** | 0.572 |

---

## 2. Analysis of Results

### Q1. Which chunking strategy works best?
**Winner: Small Chunks (250 chars)**

* **Retrieval Stability:** Interestingly, the `Hit Rate` (0.84) was consistent across all sizes. This suggests the `all-MiniLM-L6-v2` embedding model is robust enough to match query keywords to documents regardless of segment length.
* **Generation Quality:** The Small Chunk strategy achieved the highest **ROUGE-L (0.245)** and **BLEU (0.127)** scores.
    * *Interpretation:* Smaller chunks provide the LLM with focused, "noise-free" context. Large chunks (900 chars) likely introduced irrelevant text that confused the Mistral 7B model or diluted the specific answer, leading to lower lexical overlap scores.

### Q2. System Accuracy Assessment
The system demonstrates a strong baseline performance:
* **Retrieval Capability:** With a Hit Rate of **84%**, the system successfully retrieves the correct source document for 21 out of 25 questions.
* **Semantic Understanding:** An average Cosine Similarity of **~0.57** indicates the generated answers are semantically aligned with the ground truth, even if the exact wording (BLEU) varies.

### Q3. Common Failure Modes
Based on the metric analysis, the following failure patterns were observed:
1.  **Paraphrasing Penalty:** The low BLEU scores (~0.12) despite decent Cosine Similarity suggest the model is paraphrasing answers heavily rather than using the exact terminology from the text.
2.  **Unanswerable Questions:** The 16% miss rate (1.0 - 0.84) likely correlates with "Unanswerable" questions (e.g., "Favorite food"). If the system retrieves *any* document for these instead of determining it cannot answer, it counts as a retrieval miss or a hallucination.
3.  **Context Dilution:** In the 900-character strategy, ROUGE scores dropped. This indicates that providing too much context caused the model to lose focus on the specific fact requested.

---

## 3. Recommendations for Improvement

To boost performance beyond the current prototype, I recommend the following:

1.  **Implement a Re-ranking Step:**
    * *Current:* Uses raw vector similarity (Cosine).
    * *Fix:* Add a Cross-Encoder (e.g., `ms-marco-MiniLM`) to re-rank the top 5 retrieved chunks. This would likely push the MRR closer to 0.90.

2.  **Hybrid Search:**
    * *Current:* Dense vector retrieval only.
    * *Fix:* Combine Vector search with Keyword search (BM25). This helps significantly with specific entity questions (e.g., specific dates or names) where semantic search sometimes struggles.

3.  **Prompt Engineering for "Unanswerable" Queries:**
    * Update the system prompt to explicitly instruct the model: *"If the answer is not in the context, reply 'Information not available'."* This will improve faithfulness scores on negative constraints.

4.  **Metadata Filtering:**
    * For questions specifying "In Document 1...", we can parse the query to filter the retriever by filename metadata, ensuring 100% precision for document-specific questions.