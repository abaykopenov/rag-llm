from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UploadRequest(_message.Message):
    __slots__ = ("file_content", "filename", "collection")
    FILE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    file_content: bytes
    filename: str
    collection: str
    def __init__(self, file_content: _Optional[bytes] = ..., filename: _Optional[str] = ..., collection: _Optional[str] = ...) -> None: ...

class UploadResponse(_message.Message):
    __slots__ = ("document_id", "filename", "collection", "pages_count", "chunks_count", "status", "processing_time_ms", "summary")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    PAGES_COUNT_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_COUNT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    filename: str
    collection: str
    pages_count: int
    chunks_count: int
    status: str
    processing_time_ms: float
    summary: str
    def __init__(self, document_id: _Optional[str] = ..., filename: _Optional[str] = ..., collection: _Optional[str] = ..., pages_count: _Optional[int] = ..., chunks_count: _Optional[int] = ..., status: _Optional[str] = ..., processing_time_ms: _Optional[float] = ..., summary: _Optional[str] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("question", "collection", "top_k", "temperature", "max_tokens", "document_id", "element_type", "section", "keywords")
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SECTION_FIELD_NUMBER: _ClassVar[int]
    KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    question: str
    collection: str
    top_k: int
    temperature: float
    max_tokens: int
    document_id: str
    element_type: str
    section: str
    keywords: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, question: _Optional[str] = ..., collection: _Optional[str] = ..., top_k: _Optional[int] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ..., document_id: _Optional[str] = ..., element_type: _Optional[str] = ..., section: _Optional[str] = ..., keywords: _Optional[_Iterable[str]] = ...) -> None: ...

class ChunkInfo(_message.Message):
    __slots__ = ("id", "text", "score", "page", "section")
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    SECTION_FIELD_NUMBER: _ClassVar[int]
    id: str
    text: str
    score: float
    page: int
    section: str
    def __init__(self, id: _Optional[str] = ..., text: _Optional[str] = ..., score: _Optional[float] = ..., page: _Optional[int] = ..., section: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("answer", "chunks_used", "prompt", "model", "tokens_used", "timing")
    class TimingEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_USED_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TOKENS_USED_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    answer: str
    chunks_used: _containers.RepeatedCompositeFieldContainer[ChunkInfo]
    prompt: str
    model: str
    tokens_used: int
    timing: _containers.ScalarMap[str, float]
    def __init__(self, answer: _Optional[str] = ..., chunks_used: _Optional[_Iterable[_Union[ChunkInfo, _Mapping]]] = ..., prompt: _Optional[str] = ..., model: _Optional[str] = ..., tokens_used: _Optional[int] = ..., timing: _Optional[_Mapping[str, float]] = ...) -> None: ...

class QueryChunk(_message.Message):
    __slots__ = ("text_delta", "is_final", "chunks_used", "model")
    TEXT_DELTA_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_USED_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    text_delta: str
    is_final: bool
    chunks_used: _containers.RepeatedCompositeFieldContainer[ChunkInfo]
    model: str
    def __init__(self, text_delta: _Optional[str] = ..., is_final: bool = ..., chunks_used: _Optional[_Iterable[_Union[ChunkInfo, _Mapping]]] = ..., model: _Optional[str] = ...) -> None: ...

class EvalSample(_message.Message):
    __slots__ = ("question", "ground_truth")
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    GROUND_TRUTH_FIELD_NUMBER: _ClassVar[int]
    question: str
    ground_truth: str
    def __init__(self, question: _Optional[str] = ..., ground_truth: _Optional[str] = ...) -> None: ...

class EvalRequest(_message.Message):
    __slots__ = ("samples", "collection", "top_k")
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    samples: _containers.RepeatedCompositeFieldContainer[EvalSample]
    collection: str
    top_k: int
    def __init__(self, samples: _Optional[_Iterable[_Union[EvalSample, _Mapping]]] = ..., collection: _Optional[str] = ..., top_k: _Optional[int] = ...) -> None: ...

class EvalMetric(_message.Message):
    __slots__ = ("question", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall", "answer_preview")
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    FAITHFULNESS_FIELD_NUMBER: _ClassVar[int]
    ANSWER_RELEVANCY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_PRECISION_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_RECALL_FIELD_NUMBER: _ClassVar[int]
    OVERALL_FIELD_NUMBER: _ClassVar[int]
    ANSWER_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall: float
    answer_preview: str
    def __init__(self, question: _Optional[str] = ..., faithfulness: _Optional[float] = ..., answer_relevancy: _Optional[float] = ..., context_precision: _Optional[float] = ..., context_recall: _Optional[float] = ..., overall: _Optional[float] = ..., answer_preview: _Optional[str] = ...) -> None: ...

class EvalResponse(_message.Message):
    __slots__ = ("samples_count", "avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall", "avg_overall", "grade", "results", "eval_time_ms")
    SAMPLES_COUNT_FIELD_NUMBER: _ClassVar[int]
    AVG_FAITHFULNESS_FIELD_NUMBER: _ClassVar[int]
    AVG_ANSWER_RELEVANCY_FIELD_NUMBER: _ClassVar[int]
    AVG_CONTEXT_PRECISION_FIELD_NUMBER: _ClassVar[int]
    AVG_CONTEXT_RECALL_FIELD_NUMBER: _ClassVar[int]
    AVG_OVERALL_FIELD_NUMBER: _ClassVar[int]
    GRADE_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    EVAL_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    samples_count: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall: float
    grade: str
    results: _containers.RepeatedCompositeFieldContainer[EvalMetric]
    eval_time_ms: float
    def __init__(self, samples_count: _Optional[int] = ..., avg_faithfulness: _Optional[float] = ..., avg_answer_relevancy: _Optional[float] = ..., avg_context_precision: _Optional[float] = ..., avg_context_recall: _Optional[float] = ..., avg_overall: _Optional[float] = ..., grade: _Optional[str] = ..., results: _Optional[_Iterable[_Union[EvalMetric, _Mapping]]] = ..., eval_time_ms: _Optional[float] = ...) -> None: ...

class SummaryRequest(_message.Message):
    __slots__ = ("document_id",)
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: _Optional[str] = ...) -> None: ...

class SummaryResponse(_message.Message):
    __slots__ = ("document_id", "summary", "cached")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    CACHED_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    summary: str
    cached: bool
    def __init__(self, document_id: _Optional[str] = ..., summary: _Optional[str] = ..., cached: bool = ...) -> None: ...

class CollectionInfo(_message.Message):
    __slots__ = ("name", "chunks_count")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    chunks_count: int
    def __init__(self, name: _Optional[str] = ..., chunks_count: _Optional[int] = ...) -> None: ...

class CollectionList(_message.Message):
    __slots__ = ("collections",)
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedCompositeFieldContainer[CollectionInfo]
    def __init__(self, collections: _Optional[_Iterable[_Union[CollectionInfo, _Mapping]]] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "version", "llm_provider", "embedding_provider", "reranker_enabled")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LLM_PROVIDER_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_PROVIDER_FIELD_NUMBER: _ClassVar[int]
    RERANKER_ENABLED_FIELD_NUMBER: _ClassVar[int]
    status: str
    version: str
    llm_provider: str
    embedding_provider: str
    reranker_enabled: bool
    def __init__(self, status: _Optional[str] = ..., version: _Optional[str] = ..., llm_provider: _Optional[str] = ..., embedding_provider: _Optional[str] = ..., reranker_enabled: bool = ...) -> None: ...
