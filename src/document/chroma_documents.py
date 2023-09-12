from dataclasses import dataclass


@dataclass
class ChromaDocuments:
    document_contents: []
    metadata: []
    ids: []


@dataclass
class DocumentMetadata:
    source: str
