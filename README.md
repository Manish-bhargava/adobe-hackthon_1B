# Persona-Driven Document Intelligence

## 1. What does it do?
Our system acts as an **intelligent analyst** that combs through any PDF collection and surfaces exactly the
content a specific *persona* needs to accomplish a *job-to-be-done* – whether that’s a dentist learning new
spacer designs, a traveller planning a trip, or a student mastering Python.

## 2. Why is it different?
* **Domain-agnostic**: hybrid BM25 + SBERT retrieval adapts to research papers, travel guides, clinical
  reviews – no finetuning required.
* **Persona-aware scoring**: query expansion & skill-level heuristics boost the parts of a document that
  matter most to the stated user role (e.g., beginner vs. expert).
* **Layout-aware extraction**: pdfminer.six preserves natural reading order, eliminating coordinate garbage
  that plagues naïve parsers.
* **Docker-first**: one command spins up everything – no local Python wrangling.

## 3. Quick start (⏱ < 2 min)
```bash
# 1. build
docker build -t doc-analyst .

# 2. run on a sample collection (JSON input mounted from host)
docker run --rm -v "$PWD/Challenge_1b/Collection\ 1":/app/collection doc-analyst \
    collection/challenge1b_input.json > collection/challenge1b_output.json
```
Running locally?
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # installed inside Dockerfile
python run_analysis.py Challenge_1b/Collection\ 1/challenge1b_input.json
```

## 4. Input / Output schema
<img width="1061" height="601" alt="Screenshot 2025-07-27 at 10 41 37 PM" src="https://github.com/user-attachments/assets/cbd7c09b-1b06-41b2-a296-367b22d2c8d2" />

*Input (`challenge1b_input.json`)*

```json
{
  "documents": [ {"filename": "file.pdf", "title": "optional"} ],
  "persona": {"role": "Dentist"},
  "job_to_be_done": {"task": "Learn spacer designs"}
}
```
*Output (`challenge1b_output.json`)*
```json
{
  "metadata": { ... },
  "extracted_sections": [
    {"document": "file.pdf", "page_number": 4, "section_title": "INTRODUCTION", "importance_rank": 1}
  ],
  "subsection_analysis": [
    {"document": "file.pdf", "page_number": 4, "refined_text": "Short summary ..."}
  ]
}


```

## 5. Architecture ⚙️
```
PDFs → pdfminer extraction → heuristic section splitter
     ↘                                 ↘
      BM25 tokenizer                 SBERT embeddings
            ↘                           ↘
           Hybrid scorer  ← persona/query expansion & weighting
                    ↘
         Top-k extractive TextRank summariser
                    ↘
                 JSON writer
```
Key files:
* `run_analysis.py` – single entry point; **core logic untouched** per brief.
* `Dockerfile`      – reproducible environment (Python 3.9 slim).

## 6. Extending
* Swap SBERT model via env `SBERT_MODEL`.
* Add new heuristics in `BEGINNER_KEYWORDS` or `ADVANCED_KEYWORDS`.
* Plug different summariser by modifying `extractive_summarization_textrank_like()`.

## 7. Hackathon check-list ✅
- [x] Generic across domains / personas
- [x] Structured JSON I/O
- [x] Dockerised & offline-ready
- [x] Unicode-safe output
- [x] Battle-tested on three diverse collections

