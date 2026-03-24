

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


# ─────────────────────────────────────────────────────────────────────────────
# Data model for a chunk
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """
    A piece of text with attached metadata.
    This is what gets embedded and stored in the vector DB.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, chars={len(self.text)}, '{preview}...')"


# ─────────────────────────────────────────────────────────────────────────────
# Base class all splitters inherit from
# ─────────────────────────────────────────────────────────────────────────────
class BaseSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, metadata: dict = None) -> list[Chunk]:
        raise NotImplementedError

    def _add_overlap(self, pieces: list[str]) -> list[str]:
        """
        Overlap: the last N chars of chunk[i] are prepended to chunk[i+1].
        WHY: If a sentence is split across two chunks, overlap ensures
             neither chunk is missing crucial context.
        """
        if self.overlap <= 0 or len(pieces) <= 1:
            return pieces
        result = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = pieces[i - 1][-self.overlap:]
            result.append(tail + pieces[i])
        return result

    def _attach_metadata(self, pieces: list[str], base_meta: dict) -> list[Chunk]:
        """Wrap each text piece in a Chunk with metadata + position info."""
        chunks = []
        for i, text in enumerate(pieces):
            if not text.strip():
                continue
            meta = {**base_meta, "chunk_index": i, "chunk_total": len(pieces)}
            chunks.append(Chunk(text=text.strip(), metadata=meta))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTER 1: Character Splitter (naive baseline)
# ─────────────────────────────────────────────────────────────────────────────
class CharacterSplitter(BaseSplitter):
    """
    Splits text every N characters.
    Simple but can break mid-sentence.
    Use when: documents are already well-structured or very short.

    Example (chunk_size=20, overlap=5):
      "Hello world, I am here."
      → ["Hello world, I am h", "m hereI am here."]
                                  ^^^^^ overlap
    """

    def split(self, text: str, metadata: dict = None) -> list[Chunk]:
        meta = metadata or {}
        pieces = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            pieces.append(text[start:end])
            start += self.chunk_size - self.overlap
        return self._attach_metadata(pieces, meta)


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTER 2: Recursive Character Splitter (LangChain default, smarter)
# ─────────────────────────────────────────────────────────────────────────────
class RecursiveSplitter(BaseSplitter):
    """
    Tries separators in order: paragraph → sentence → word → character.
    This preserves natural language boundaries as much as possible.

    HOW IT WORKS:
      1. Try splitting by "\n\n" (paragraphs). If chunks are small enough → done.
      2. If a paragraph is still too big, split it by "\n" (lines).
      3. Still too big? Split by ". " (sentences).
      4. Still too big? Split by " " (words).
      5. Last resort: split by character.

    This is what LangChain's RecursiveCharacterTextSplitter does internally.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def split(self, text: str, metadata: dict = None) -> list[Chunk]:
        meta = metadata or {}
        pieces = self._recursive_split(text)
        # Apply overlap
        overlapped = self._add_overlap(pieces)
        return self._attach_metadata(overlapped, meta)

    def _recursive_split(self, text: str) -> list[str]:
        """Recursively split using the separator hierarchy."""
        if len(text) <= self.chunk_size:
            return [text]

        for sep in self.SEPARATORS:
            if sep == "":
                # Last resort: character split
                return [text[i:i+self.chunk_size]
                        for i in range(0, len(text), self.chunk_size)]
            if sep in text:
                parts = text.split(sep)
                result = []
                current = ""
                for part in parts:
                    candidate = current + (sep if current else "") + part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            result.extend(self._recursive_split(current))
                        current = part
                if current:
                    result.extend(self._recursive_split(current))
                return result
        return [text]


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTER 3: Markdown Splitter (respects document structure)
# ─────────────────────────────────────────────────────────────────────────────
class MarkdownSplitter(BaseSplitter):
    """
    Splits at Markdown header boundaries (# ## ###).
    Each section becomes a chunk, with the header path as metadata.

    Example document:
      # ML Guide
      ## Chapter 1: Transformers
      Attention is all you need...
      ## Chapter 2: RAG
      RAG combines retrieval...

    Produces chunks:
      Chunk(text="Attention is all...", metadata={"header": "ML Guide > Chapter 1: Transformers"})
      Chunk(text="RAG combines...",    metadata={"header": "ML Guide > Chapter 2: RAG"})

    WHY: You can then filter by header when retrieving:
      filter = {"header": {"$contains": "Chapter 2"}}
    """

    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def split(self, text: str, metadata: dict = None) -> list[Chunk]:
        meta = metadata or {}
        chunks = []
        headers = list(self.HEADER_PATTERN.finditer(text))

        if not headers:
            # No markdown headers → fall back to recursive splitter
            return RecursiveSplitter(self.chunk_size, self.overlap).split(text, meta)

        # Build sections between headers
        sections = []
        for i, match in enumerate(headers):
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            level = len(match.group(1))
            title = match.group(2).strip()
            content = text[start:end].strip()
            sections.append({"level": level, "title": title, "content": content})

        # Build breadcrumb path for each section
        header_stack = []
        for sec in sections:
            # Pop stack until we're at the right level
            header_stack = [h for h in header_stack if h["level"] < sec["level"]]
            header_stack.append(sec)
            path = " > ".join(h["title"] for h in header_stack)

            # If section is too long, sub-split it
            if len(sec["content"]) > self.chunk_size:
                sub = RecursiveSplitter(self.chunk_size, self.overlap)
                sub_chunks = sub.split(sec["content"], {**meta, "header_path": path})
                chunks.extend(sub_chunks)
            elif sec["content"]:
                chunk_meta = {**meta, "header_path": path, "header_level": sec["level"]}
                chunks.append(Chunk(text=sec["content"], metadata=chunk_meta))

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTER 4: Sentence Splitter (clean sentence boundaries)
# ─────────────────────────────────────────────────────────────────────────────
class SentenceSplitter(BaseSplitter):
    """
    Groups complete sentences into chunks up to chunk_size.
    Never breaks mid-sentence — better embedding quality than char split.

    HOW IT WORKS:
      1. Split text into individual sentences using regex.
      2. Greedily accumulate sentences until chunk_size is reached.
      3. Start a new chunk. Apply overlap by repeating last sentence.
    """

    SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

    def split(self, text: str, metadata: dict = None) -> list[Chunk]:
        meta = metadata or {}
        sentences = self.SENTENCE_END.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks_text = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.chunk_size:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks_text.append(current)
                # Overlap: start new chunk with last sentence of previous
                if self.overlap > 0 and chunks_text:
                    last_sents = chunks_text[-1].split(". ")
                    overlap_text = last_sents[-1] if last_sents else ""
                    current = (overlap_text + " " + sent).strip()
                else:
                    current = sent

        if current:
            chunks_text.append(current)

        return self._attach_metadata(chunks_text, meta)


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTER 5: Semantic Splitter (topic-based grouping)
# ─────────────────────────────────────────────────────────────────────────────
class SemanticSplitter(BaseSplitter):
    """
    Groups sentences by semantic similarity — sentences on the same
    topic stay in the same chunk.

    HOW IT WORKS:
      1. Split into sentences.
      2. Embed each sentence.
      3. Compute cosine similarity between consecutive sentences.
      4. When similarity drops below threshold → topic changed → new chunk.

    This produces the most semantically coherent chunks but is slower
    because it calls embed() for every sentence.

    WHY IT'S POWERFUL:
      Instead of arbitrary char boundaries, each chunk covers ONE topic.
      The embedding of the chunk is more focused → better retrieval.
    """

    def __init__(self, llm, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                 breakpoint_threshold=0.3):
        super().__init__(chunk_size, overlap)
        self.llm = llm
        self.threshold = breakpoint_threshold  # drop below this → new chunk

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        return dot / (mag_a * mag_b + 1e-9)

    def split(self, text: str, metadata: dict = None) -> list[Chunk]:
        meta = metadata or {}
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= 1:
            return [Chunk(text=text.strip(), metadata=meta)]

        # Embed all sentences
        embeddings = [self.llm.embed(s) for s in sentences]

        # Find breakpoints where topic changes
        groups = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                groups.append([sentences[i]])   # new topic → new chunk
            else:
                groups[-1].append(sentences[i]) # same topic → same chunk

        # Join groups into chunk texts
        chunks_text = [" ".join(g) for g in groups]

        # If any chunk is still too long, sub-split recursively
        final = []
        for ct in chunks_text:
            if len(ct) > self.chunk_size:
                sub = RecursiveSplitter(self.chunk_size, self.overlap)
                final.extend(sub.split(ct, meta))
            else:
                final.append(Chunk(text=ct, metadata=meta))

        return final


# ─────────────────────────────────────────────────────────────────────────────
# Factory — choose splitter by name
# ─────────────────────────────────────────────────────────────────────────────
def get_splitter(name: str = "recursive", llm=None, **kwargs) -> BaseSplitter:
    """
    Usage:
        splitter = get_splitter("recursive")
        splitter = get_splitter("markdown")
        splitter = get_splitter("semantic", llm=my_llm)
        chunks = splitter.split(text, metadata={"source": "doc.pdf"})
    """
    name = name.lower()
    if name == "character":
        return CharacterSplitter(**kwargs)
    elif name == "recursive":
        return RecursiveSplitter(**kwargs)
    elif name == "markdown":
        return MarkdownSplitter(**kwargs)
    elif name == "sentence":
        return SentenceSplitter(**kwargs)
    elif name == "semantic":
        if llm is None:
            raise ValueError("SemanticSplitter requires llm= argument")
        return SemanticSplitter(llm=llm, **kwargs)
    else:
        raise ValueError(f"Unknown splitter '{name}'. Options: character|recursive|markdown|sentence|semantic")
