import uuid
from datetime import datetime
from typing import Any, Optional, Union

from metagpt.actions.action import Action
from metagpt.actions.action_output import ActionOutput
from pydantic import BaseModel, Field, field_validator

from message_enum import SentenceType


class SentenceValue(BaseModel):
    answer: str


class Sentence(BaseModel):
    type: str
    id: Optional[str] = None
    value: SentenceValue
    is_finished: Optional[bool] = None

    @field_validator("id", mode="before")
    @classmethod
    def validate_credits(cls, v):
        if isinstance(v, str):
            return v
        return str(v)


class Sentences(BaseModel):
    id: Optional[str] = None
    action: Optional[str] = None
    role: Optional[str] = None
    skill: Optional[str] = None
    description: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"))
    status: str
    contents: list[dict]


class NewMsg(BaseModel):
    """Chat with MetaGPT"""

    query: str = Field(description="Problem description")
    config: dict[str, Any] = Field(description="Configuration information")


class LLMAPIkeyTest(BaseModel):
    """APIkey"""

    api_key: str = Field(description="API Key")
    llm_type: str = Field(description="Model Type")


class ErrorInfo(BaseModel):
    error: str = None
    traceback: str = None


class ThinkActStep(BaseModel):
    id: str
    status: str
    title: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"))
    description: str
    content: Sentence = None

    @field_validator("id", mode="before")
    @classmethod
    def validate_credits(cls, v):
        if isinstance(v, str):
            return v
        return str(v)


class ThinkActPrompt(BaseModel):
    message_id: int = None
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"))
    step: ThinkActStep = None
    skill: Optional[str] = None
    role: Optional[str] = None

    def update_think(self, tc_id, action: Action):
        self.step = ThinkActStep(
            id=str(tc_id),
            status="running",
            title=action.desc,
            description=action.desc,
        )

    def update_act(self, message: Union[ActionOutput, str], is_finished: bool = True):
        if is_finished:
            self.step.status = "finish"
        self.step.content = Sentence(
            type=SentenceType.TEXT.value,
            id=str(1),
            value=SentenceValue(answer=message.content if is_finished else message),
            is_finished=is_finished,
        )

    @staticmethod
    def guid32():
        return str(uuid.uuid4()).replace("-", "")[0:32]

    @property
    def prompt(self):
        return self.json(exclude_unset=True)


class MessageJsonModel(BaseModel):
    steps: list[Sentences]
    qa_type: str
    created_at: datetime = Field(default_factory=datetime.now)
    query_time: datetime = Field(default_factory=datetime.now)
    answer_time: datetime = Field(default_factory=datetime.now)
    score: Optional[int] = None
    feedback: Optional[str] = None

    def add_think_act(self, think_act_prompt: ThinkActPrompt):
        s = Sentences(
            action=think_act_prompt.step.title,
            skill=think_act_prompt.skill,
            description=think_act_prompt.step.description,
            timestamp=think_act_prompt.timestamp,
            status=think_act_prompt.step.status,
            contents=[think_act_prompt.step.content.dict()],
        )
        self.steps.append(s)

    @property
    def prompt(self):
        return self.json(exclude_unset=True)
