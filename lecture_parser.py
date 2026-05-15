#Requires poppler-utils and tesseract-ocr
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTAnno, LTChar
import numpy as np
import os
import re

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_structured_elements(pdf_path=None, pdf=None):
    data = []

    source = pdf_path if pdf_path else pdf

    for page_layout in extract_pages(source):
        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue

            text = element.get_text().strip()
            if not text or len(text) < 10:
                continue

            # Filter headers/footers by vertical position on page
            if element.y0 < 50 or element.y1 > page_layout.height - 50:
                continue

            # Filter common garbage
            if re.match(r"^(page\s*\d+|©|copyright|\d+$)", text.lower()):
                continue

            data.append({"type": "NarrativeText", "text": text})

    if not data:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned or image-only document."
        )

    return data


def merge_and_clean(elements, min_length=50):
    #Merge small text fragments (titles, captions) with nearby paragraphs to make smoother chunks.
    merged = []
    buffer = []

    for el in elements:
        if isinstance(el,dict):
            t = el["text"]
        else:
            t = el
        if len(t) < min_length:
            buffer.append(t)
        else:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            merged.append(t)

    if buffer:
        merged.append(" ".join(buffer))

    return merged


def semantic_split(text_blocks, threshold=0.5):
    # Group text blocks into coherent chunks based on semantic similarity.
    if not text_blocks:
        return []
    embeddings = model.encode(text_blocks, convert_to_numpy=True)

    chunks = []
    current_chunk = [text_blocks[0]]

    for i in range(1, len(text_blocks)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [text_blocks[i]]
        else:
            current_chunk.append(text_blocks[i])

    chunks.append(" ".join(current_chunk))
    return chunks

def process_pdf_to_chunks(pdf_path=None,pdf_file=None):
    elements = extract_structured_elements(pdf_path,pdf_file)
    merged_texts = merge_and_clean(elements)
    chunks = semantic_split(merged_texts)
    return chunks

def cluster_chunks(chunks, num_clusters=20):
    if len(chunks) < 2:
        return chunks

        # Can't have more clusters than samples
    num_clusters = min(num_clusters, len(chunks))
    # Get embeddings and cluster to ~30 groups
    embeddings = model.encode(chunks, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)
    clustering = AgglomerativeClustering(num_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(1 - sim_matrix)

    # Pick one representative per cluster
    final_chunks = [chunks[np.where(labels == label)[0][0]] for label in set(labels)]
    return final_chunks


def extract_context_from_pdf(pdf_file_path=None,pdf_file=None,num_chunks=20):
    chunks = process_pdf_to_chunks(pdf_file_path,pdf_file)
    print(f"Extracted {len(chunks)} coherent sections:\n")
    for i, c in enumerate(chunks[:5], 1):
        print(f"--- Section {i} ---\n{c[:500]}...\n")

    counter = 0
    for i, c in enumerate(chunks, 1):
        if len(c) >= 200:
            counter += 1
            print(f"\n--- Chunk {i} ---\n{c[:1000]}...")
    print(counter)

    filtered = [c for c in chunks if len(c) > 200]

    final_chunks=cluster_chunks(filtered,num_chunks)

    for i, c in enumerate(final_chunks, 1):
        print(f"\n--- Chunk {i} ---\n{c[:1000]}...")
    print(counter)
    return final_chunks

def extract_context_from_text(text,num_chunks=20):
    # Split into paragraphs/sentences first, same as PDF elements
    blocks = [b.strip() for b in re.split(r'\n{2,}|\n', text) if b.strip()]

    # Fall back to sentence splitting if no newlines (e.g. one big paragraph)
    if len(blocks) < 3:
        blocks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    merged_texts = merge_and_clean(blocks)
    chunks = semantic_split(blocks)
    print(f"Extracted {len(chunks)} coherent sections:\n")
    for i, c in enumerate(chunks[:5], 1):
        print(f"--- Section {i} ---\n{c[:500]}...\n")

    counter = 0
    for i, c in enumerate(chunks, 1):
        if len(c) >= 200:
            counter += 1
            print(f"\n--- Chunk {i} ---\n{c[:1000]}...")
    print(counter)

    filtered = [c for c in chunks if len(c) > 200]
    if not filtered:
        filtered = [c for c in chunks if len(c) > 50]
    if not filtered:
        filtered = chunks  # last resort: use everything

        # Guard: nothing usable at all
        if not filtered:
            return []

    final_chunks=cluster_chunks(filtered,num_clusters=min(num_chunks, len(filtered)))
    for i, c in enumerate(final_chunks, 1):
        print(f"\n--- Chunk {i} ---\n{c[:1000]}...")
    print(counter)
    return final_chunks

if __name__=="__main__":
    extract_context_from_pdf("test_files/Lecture5.pdf")