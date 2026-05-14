#Requires poppler-utils and tesseract-ocr
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import re

model = SentenceTransformer("all-MiniLM-L6-v2")
def extract_structured_elements(pdf_path=None,pdf=None):
    # "hi_res" for better layout parsing, "fast" for speed
    elements = partition_pdf(filename=pdf_path,file=pdf, strategy="auto")

    data = []
    for el in elements:
        text = el.text.strip()
        if not text:
            continue

        # Filter: only keep meaningful narrative elements
        if el.category in ("NarrativeText", "ListItem", "UncategorizedText"):
            # Remove common footer/header garbage
            if not re.match(r"^(page\s*\d+|©|copyright)", text.lower()):
                data.append({
                    "type": el.category,
                    "text": text
                })
    return data


def merge_and_clean(elements, min_length=50):
    #Merge small text fragments (titles, captions) with nearby paragraphs to make smoother chunks.
    merged = []
    buffer = []

    for el in elements:
        t = el["text"]
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
    chunks = semantic_split(text)
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

