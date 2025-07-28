import PyPDF2
# Try to load pdfminer.six for superior layout-aware extraction
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLine
    _PDFMINER_AVAILABLE = True
except ImportError:
    _PDFMINER_AVAILABLE = False
import spacy
from spacy.lang.en import English

# Try to load SentenceTransformers for high-quality embeddings
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import json
import os
import re
import datetime
from collections import Counter
import sys # Added for command-line arguments

# Load a small spaCy model for tokenization and basic NLP (ensure it's downloaded)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # In a Docker environment, the model is downloaded during build,
    # so this OSError block should ideally not be hit during execution.
    # For local script testing, if you didn't download it with conda, it might.
    print("SpaCy model 'en_core_web_sm' not found. This should be pre-downloaded.", file=sys.stderr)
    sys.exit(1) # Exit if model is not found

# Initialise English tokenizer from spaCy
tokenizer = English().tokenizer

# ---------------- Heuristic keyword sets for skill-level weighting ----------------
BEGINNER_KEYWORDS = [
    "introduction", "getting started", "basics", "beginner", "overview", "syntax",
    "variables", "functions", "loops", "data types", "tutorial", "quick start"
]

ADVANCED_KEYWORDS = [
    "swig", "pyrex", "cython", "c api", "parser", "ply", "extension", "bindings",
    "code generation", "database", "xml", "generate", "advanced", "optimization"
]
# Optional: Load SBERT model lazily to avoid high startup cost when not needed
_sbert_model = None
_embedding_cache = {}  # Cache for embeddings to avoid recomputation
_speed_mode = True  # Enable speed-first mode for sub-60s performance

def _get_sbert_model():
    global _sbert_model
    if _speed_mode:
        return None  # Skip SBERT in speed mode
    if _sbert_model is None and _SBERT_AVAILABLE:
        # Using a small yet strong model (~60MB)
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Optimize for CPU inference
        _sbert_model.eval()
    return _sbert_model

# Helper Functions
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF and returns a list of page dictionaries
    (page_number -> text). Strategy:

    1. If pdfminer.six is available, use it because it keeps words in correct
       reading order and omits coordinate dictionaries that sometimes pollute
       the PyPDF2 stream.
    2. If pdfminer is not present or fails for any reason, gracefully fall back
       to PyPDF2 so the pipeline still functions.
    """
    text_by_page = []
    try:
        if _PDFMINER_AVAILABLE:
            # ---- pdfminer path ----
            try:
                for page_num, page_layout in enumerate(extract_pages(pdf_path)):
                    page_lines = []
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            for text_line in element:
                                if isinstance(text_line, LTTextLine):
                                    line_text = text_line.get_text().rstrip("\n")
                                    if line_text:
                                        page_lines.append(line_text)
                    page_text = "\n".join(page_lines)
                    text_by_page.append({"page_number": page_num + 1, "text": page_text})
            except Exception as pdfminer_exc:
                # pdfminer is installed but failed; fall back transparently
                print(f"pdfminer failed on {pdf_path}: {pdfminer_exc}. Falling back to PyPDF2.", file=sys.stderr)
                text_by_page = []

        # ---- fallback to PyPDF2 if needed or pdfminer unavailable ----
        if not text_by_page:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_by_page.append({"page_number": page_num + 1, "text": text})
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}", file=sys.stderr)
    return text_by_page

def clean_text(text):
    """Enhanced text cleaning: remove table of contents artifacts and improve readability."""
    if not isinstance(text, str):
        return ""
    
    # Remove table of contents dot leaders (multiple dots)
    text = re.sub(r'\.{3,}', ' ', text)
    
    # Remove page number patterns
    text = re.sub(r'Page \d+', '', text)
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def identify_sections(page_text_list, min_section_length=100):
    """
    Attempts to identify sections within a document by looking for heading-like patterns.
    This is a heuristic and relies on the text structure provided by PyPDF2.
    """
    sections = []
    current_section_title = "Untitled Section"
    current_section_text_buffer = []

    for page_data in page_text_list:
        page_num = page_data["page_number"]
        doc_name = page_data.get("document_name", "Unknown Document")
        lines = page_data["text"].split('\n')

        for line in lines:
            stripped_line = line.strip()
            is_potential_heading = (
                len(stripped_line) > 0 and
                len(stripped_line) < 100 and
                (stripped_line.isupper() or re.match(r'^\d+(\.\d+)*\s+[A-Z].*', stripped_line)) and
                not stripped_line.isdigit()
            )

            if is_potential_heading and len(" ".join(current_section_text_buffer)) > min_section_length:
                sections.append({
                    "document": doc_name,
                    "page_number": page_num,
                    "section_title": current_section_title.strip() if current_section_title else "Untitled Section",
                    "text": " ".join(current_section_text_buffer).strip()
                })
                current_section_title = stripped_line
                current_section_text_buffer = []
            elif stripped_line:
                current_section_text_buffer.append(stripped_line)

        if current_section_text_buffer:
            sections.append({
                "document": doc_name,
                "page_number": page_num,
                "section_title": current_section_title.strip() if current_section_title else "Page Content",
                "text": " ".join(current_section_text_buffer).strip()
            })
            current_section_title = f"Page {page_num} Content"
            current_section_text_buffer = []

    merged_sections = []
    if sections:
        merged_sections.append(sections[0])
        for i in range(1, len(sections)):
            if (len(sections[i]["text"]) < min_section_length / 2 and
                sections[i]["document"] == merged_sections[-1]["document"] and
                sections[i]["page_number"] == merged_sections[-1]["page_number"]):
                merged_sections[-1]["text"] += " " + sections[i]["text"]
                if "Untitled Section" in merged_sections[-1]["section_title"] or "Page Content" in merged_sections[-1]["section_title"]:
                     merged_sections[-1]["section_title"] = sections[i]["section_title"]
            else:
                merged_sections.append(sections[i])

    # --- Enhanced section title cleaning and improvement
    for sec in merged_sections:
        title = sec["section_title"]
        
        # Clean up common formatting artifacts
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces
        title = re.sub(r'[A-Z]\s+[A-Z]\s+[A-Z]', lambda m: m.group().replace(' ', ''), title)  # Spaced capitals like "I T A L Y"
        title = title.replace('…', '').strip()  # Remove ellipsis
        
        # If title is still poor quality, generate from content
        if (len(title) < 5 or 
            title.startswith("Page") or 
            title.startswith("Untitled") or
            len(title.split()) < 2):
            
            # Extract meaningful title from first sentence
            sentences = sec["text"].split('. ')
            if sentences:
                first_sentence = sentences[0].strip()
                # Clean and truncate to reasonable length
                first_sentence = re.sub(r'[^a-zA-Z0-9\s\-:]', ' ', first_sentence)
                first_sentence = re.sub(r'\s+', ' ', first_sentence).strip()
                
                if len(first_sentence) > 60:
                    words = first_sentence.split()
                    title = ' '.join(words[:8]) + '...' if len(words) > 8 else first_sentence
                elif len(first_sentence) > 5:
                    title = first_sentence
                else:
                    # Last resort: use first few words of content
                    words = sec["text"].split()[:6]
                    title = ' '.join(words) + '...' if len(words) >= 6 else ' '.join(words)
        
        # Final cleaning
        title = title.strip()
        if not title:
            title = f"Section from page {sec['page_number']}"
            
        sec["section_title"] = title
    
    return merged_sections


def get_sentence_embeddings(text, nlp_model=nlp):
    """Return an embedding for the given text with caching for performance.

    Priority order:
    1. SentenceTransformer (if installed) – high-quality semantic vectors.
    2. spaCy pooled word vectors (fallback).
    3. Zero vector (edge cases).
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(384 if _SBERT_AVAILABLE else 96)
    
    # Use first 500 chars as cache key for speed
    cache_key = text[:500].strip().lower()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    if _SBERT_AVAILABLE:
        model = _get_sbert_model()
        try:
            # Limit text length for faster processing
            truncated_text = text[:512]  # SBERT optimal length
            embedding = model.encode(truncated_text, show_progress_bar=False)  # returns a numpy array
            _embedding_cache[cache_key] = embedding
            return embedding
        except Exception:
            # Fallback to spaCy if encoding fails for any reason
            pass

    # spaCy fallback
    doc = nlp(text[:500])  # Limit text length for speed
    if doc.has_vector and len(doc) > 0:
        embedding = doc.vector
        _embedding_cache[cache_key] = embedding
        return embedding
    else:
        return np.zeros(nlp_model.vocab.vectors.shape[1] if nlp_model.vocab.vectors else 96)

def calculate_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extractive_summarization_textrank_like(text, query_vector, top_n_sentences=3):
    """
    Enhanced extractive summarization that filters out table-of-contents entries
    and provides meaningful content summaries.
    """
    # Clean the text first to remove TOC artifacts
    cleaned_text = re.sub(r'\.{3,}', ' ', text)  # Remove dot leaders
    cleaned_text = re.sub(r'Page \d+', '', cleaned_text)  # Remove page numbers
    
    doc = nlp(cleaned_text)
    sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        # Filter out table-of-contents style entries
        if (sent_text and 
            len(sent_text.split()) > 5 and  # Must have more than 5 words
            not re.search(r'\d+$', sent_text) and  # Doesn't end with page number
            '...' not in sent_text and  # No dot leaders
            not re.match(r'^\d+\.\d+', sent_text.strip())):  # Not section numbering only
            sentences.append(sent_text)
    
    if not sentences:
        # Fallback: return first meaningful sentence from original text
        fallback_sentences = [s.text.strip() for s in nlp(text).sents 
                            if s.text.strip() and len(s.text.split()) > 3]
        return fallback_sentences[0] if fallback_sentences else text[:200]

    if _speed_mode:
        # In speed mode, just return the first few meaningful sentences
        return " ".join(sentences[:top_n_sentences])
    
    # Full mode with similarity scoring
    sentence_vectors = [get_sentence_embeddings(sent) for sent in sentences]
    scores = []
    for i, sent_vec in enumerate(sentence_vectors):
        query_sim = calculate_cosine_similarity(sent_vec, query_vector)
        scores.append((query_sim, sentences[i]))

    scores.sort(key=lambda x: x[0], reverse=True)
    summary_sentences = [sent for score, sent in scores[:top_n_sentences]]
    return " ".join(summary_sentences)

def bm25_score_documents(tokenized_corpus, tokenized_query):
    """Calculates BM25 scores for documents given a tokenized query."""
    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores

def tokenize_text(text):
    """Tokenizes text using spaCy's tokenizer, filtering for alpha characters."""
    if not isinstance(text, str):
        return []
    return [token.text for token in tokenizer(text) if token.is_alpha]

# Main Processing Logic
def intelligent_document_analyst(document_paths, persona_description, job_to_be_done):
    """
    Main function to analyze documents based on persona and job-to-be-done.
    """
    start_time = datetime.datetime.now()

    input_documents = [os.path.basename(p) for p in document_paths]

    query_text = f"{persona_description} {job_to_be_done}"

    # Automatic query expansion for beginner learning scenarios to emphasise
    # foundational concepts that typical beginner material covers.
    learning_signals = ["learn", "beginner", "introduction", "tutorial", "study", "basics"]
    if any(sig in job_to_be_done.lower() for sig in learning_signals):
        expansion_terms = "introduction basics variables loops functions syntax getting started tutorial overview"
        query_text += " " + expansion_terms
    query_vector = get_sentence_embeddings(query_text)
    tokenized_query = tokenize_text(query_text)

    # Step 2: Extract sections from each document (optimized)
    all_sections = []
    for doc_path in document_paths:
        if not os.path.exists(doc_path):
            print(f"Warning: Document {doc_path} not found.", file=sys.stderr)
            continue
        
        pages = extract_text_from_pdf(doc_path)
        sections = identify_sections(pages, min_section_length=300)  # Much higher min length
        
        # Drastically limit sections per document for speed (top 20 per doc)
        sections = sections[:20] if len(sections) > 20 else sections
        
        for section in sections:
            section['document'] = os.path.basename(doc_path)
            all_sections.append(section)

    if not all_sections:
        print("Warning: No sections extracted from documents. Check PDF parsing or document content.", file=sys.stderr)
        return {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona_description,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

    section_texts_cleaned = [clean_text(section["text"]) for section in all_sections]
    tokenized_section_texts = [tokenize_text(text) for text in section_texts_cleaned]

    if not tokenized_section_texts or all(not tokens for tokens in tokenized_section_texts):
        print("Warning: No valid tokens for BM25. Check text cleaning or document content.", file=sys.stderr)
        bm25_section_scores = [0.0] * len(all_sections)
    else:
        # Step 3: Score and rank sections (optimized for speed)
        scored_sections = []
        
        # Drastically limit total sections for sub-60s performance (top 50 across all docs)
        if len(all_sections) > 50:
            all_sections = all_sections[:50]
        
        # Prepare corpus for BM25
        corpus_texts = [section['text'][:1000] for section in all_sections]  # Truncate for speed
        tokenized_corpus = [tokenize_text(text) for text in corpus_texts]
        
        if tokenized_corpus:
            bm25_scores = bm25_score_documents(tokenized_corpus, tokenized_query)
        else:
            bm25_scores = []
        
        # Batch process embeddings for better performance
        section_texts = [section['text'][:512] for section in all_sections]  # Truncate for SBERT
        
        for i, section in enumerate(all_sections):
            section_text = section_texts[i]
            
            # BM25 score
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            
            if _speed_mode:
                # Speed mode: Use only BM25 + keyword matching (no embeddings)
                semantic_score = 0.0
                
                # Fast persona-based weighting
                persona_weight = 1.0
                section_lower = section_text.lower()
                
                # Quick keyword matching
                if "beginner" in persona_description.lower() or "student" in persona_description.lower():
                    if any(kw in section_lower for kw in BEGINNER_KEYWORDS[:3]):  # Check only top 3 keywords
                        persona_weight += 0.3
                elif "expert" in persona_description.lower() or "advanced" in persona_description.lower():
                    if any(kw in section_lower for kw in ADVANCED_KEYWORDS[:3]):  # Check only top 3 keywords
                        persona_weight += 0.3
                
                # BM25-only score for speed
                combined_score = bm25_score * persona_weight
            else:
                # Full mode: Use BM25 + semantic similarity
                section_vector = get_sentence_embeddings(section_text)
                semantic_score = calculate_cosine_similarity(query_vector, section_vector)
                
                # Simplified persona-based weighting for speed
                persona_weight = 1.0
                section_lower = section_text.lower()
                
                # Quick keyword matching
                if "beginner" in persona_description.lower() or "student" in persona_description.lower():
                    if any(kw in section_lower for kw in BEGINNER_KEYWORDS[:5]):  # Check only top 5 keywords
                        persona_weight += 0.2
                elif "expert" in persona_description.lower() or "advanced" in persona_description.lower():
                    if any(kw in section_lower for kw in ADVANCED_KEYWORDS[:5]):  # Check only top 5 keywords
                        persona_weight += 0.2
                
                # Combined score
                combined_score = (0.4 * bm25_score + 0.6 * semantic_score) * persona_weight
            
            scored_sections.append({
                'section': section,
                'score': combined_score,
                'bm25_score': bm25_score,
                'semantic_score': semantic_score
            })

    scored_sections.sort(key=lambda x: x["score"], reverse=True)

    # ---------------- Post-filter for learning tasks ----------------
    learning_signals = ["learn", "beginner", "introduction", "tutorial", "study", "basics"]
    filtered_scored_sections = []
    if any(sig in job_to_be_done.lower() for sig in learning_signals):
        for item in scored_sections:
            title_lower = item["section"]["section_title"].lower()
            text_len = len(item["section"]["text"].split())

            # Rule 1: must have at least 40 words of prose (skip code stubs)
            if text_len < 40:
                continue

            # Rule 2: discard if advanced keyword present in title and no beginner keyword
            has_beginner_kw = any(kw in title_lower for kw in BEGINNER_KEYWORDS)
            has_advanced_kw = any(kw in title_lower for kw in ADVANCED_KEYWORDS)
            if has_advanced_kw and not has_beginner_kw:
                continue

            filtered_scored_sections.append(item)

        # If filtering removed too many, fall back to original list
        if len(filtered_scored_sections) < 10:
            filtered_scored_sections = scored_sections
    else:
        filtered_scored_sections = scored_sections

    output_extracted_sections = []
    for i, item in enumerate(filtered_scored_sections):
        output_extracted_sections.append({
            "document": item["section"]["document"],
            "page_number": item["section"]["page_number"],
            "section_title": item["section"]["section_title"],
            "importance_rank": i + 1
        })

    output_sub_section_analysis = []
    top_sections_for_analysis = filtered_scored_sections[:min(10, len(filtered_scored_sections))]

    for item in top_sections_for_analysis:
        original_section_text = item["section"]["text"]

        if len(original_section_text.split()) > 20:
            refined_text = extractive_summarization_textrank_like(original_section_text, query_vector, top_n_sentences=3)
        else:
            refined_text = original_section_text

        output_sub_section_analysis.append({
            "document": item["section"]["document"],
            "page_number": item["section"]["page_number"],
            "refined_text": refined_text.strip()
        })

    end_time = datetime.datetime.now()
    processing_duration = (end_time - start_time).total_seconds()
    print(f"Processing finished in {processing_duration:.2f} seconds.", file=sys.stderr) # Print to stderr so JSON is clean

    output_json = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona_description,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": end_time.isoformat()
        },
        "extracted_sections": output_extracted_sections,
        "subsection_analysis": output_sub_section_analysis
    }

    return output_json

if __name__ == "__main__":
    # This block is executed when the script is run directly from the command line.

    def process_input_json(json_input_path):
        """Reads the Challenge 1b input JSON, prepares arguments for the core analysis, and writes output."""
        # Resolve absolute path in case the user supplies a relative one
        json_input_path = os.path.abspath(json_input_path)

        try:
            with open(json_input_path, 'r', encoding='utf-8') as jf:
                challenge_payload = json.load(jf)
        except Exception as exc:
            print(f"Failed to read input JSON {json_input_path}: {exc}", file=sys.stderr)
            sys.exit(1)

        # Extract persona and job-to-be-done fields as per spec
        persona_description = challenge_payload.get("persona", {}).get("role", "")
        job_to_be_done = challenge_payload.get("job_to_be_done", {}).get("task", "")

        # Resolve document paths. If relative, consider them located under a `PDFs` directory
        base_dir = os.path.dirname(json_input_path)
        doc_entries = challenge_payload.get("documents", [])
        document_paths = []
        for doc in doc_entries:
            fname = doc.get("filename", "")
            if not fname:
                continue
            if os.path.isabs(fname):
                document_paths.append(fname)
            else:
                # Try <base>/PDFs/<filename> first, fallback to <base>/<filename>
                candidate_path = os.path.join(base_dir, "PDFs", fname)
                if not os.path.exists(candidate_path):
                    candidate_path = os.path.join(base_dir, fname)
                document_paths.append(candidate_path)

        # Run analysis using the core logic (untouched)
        result_json = intelligent_document_analyst(document_paths, persona_description, job_to_be_done)

        # Write output next to the input file with the required name
        output_json_path = os.path.join(base_dir, "challenge1b_output.json")
        try:
            with open(output_json_path, "w", encoding="utf-8") as outf:
                json.dump(result_json, outf, indent=4, ensure_ascii=False)
        except Exception as exc:
            print(f"Failed to write output JSON to {output_json_path}: {exc}", file=sys.stderr)

        return result_json

    # Support two modes:
    # 1. JSON-driven (preferred for Challenge 1b): python run_analysis.py <challenge1b_input.json>
    # 2. Legacy CLI: python run_analysis.py <doc1.pdf> ... "<Persona>" "<Job>"

    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".json"):
        # JSON-driven invocation
        output = process_input_json(sys.argv[1])
        print(json.dumps(output, indent=4, ensure_ascii=False))
    else:
        # Legacy invocation fallback (original behaviour)
        if len(sys.argv) < 4:
            print("Usage (JSON): python run_analysis.py <challenge1b_input.json>", file=sys.stderr)
            print("   or (Legacy): python run_analysis.py <doc_path1> [doc_path2 ...] \"<Persona>\" \"<Job-to-be-Done>\"", file=sys.stderr)
            sys.exit(1)

        # The first arguments are document paths, the last two are persona and job
        doc_paths_raw = sys.argv[1:-2]
        persona = sys.argv[-2]
        job = sys.argv[-1]

        # Adjust paths for execution within the Docker container's /app/documents directory
        document_paths = []
        for p_raw in doc_paths_raw:
            if not os.path.isabs(p_raw):
                document_paths.append(os.path.join('/app/', p_raw))
            else:
                document_paths.append(p_raw)

        # Call the main analysis function
        result = intelligent_document_analyst(document_paths, persona, job)

        # Print the JSON output to standard output
        print(json.dumps(result, indent=4, ensure_ascii=False))