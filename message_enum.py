from enum import Enum


class QueryAnswerType(Enum):
    Query = "Q"
    Answer = "A"


class SentenceType(Enum):
    TEXT = "text"
    HIHT = "hint"
    ACTION = "action"
    ERROR = "error"


class MessageStatus(Enum):
    COMPLETE = "complete"
