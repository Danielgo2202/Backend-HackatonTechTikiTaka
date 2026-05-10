"""WebSocket and API payload shapes (contract with frontend)."""

from typing import Any

from pydantic import BaseModel, Field


class BattlecardData(BaseModel):
    key_differentiator: str
    suggested_response: str
    recommended_question: str
    weaknesses: list[str] = Field(default_factory=list)


class ClientContext(BaseModel):
    name: str | None = None
    industry: str | None = None
    deal_size: str | None = None
    pain_points: list[str] | None = None


class BattlecardEvent(BaseModel):
    type: str = "battlecard"
    competitor: str
    confidence: float
    data: BattlecardData
    client_context: ClientContext | None = None

    def model_dump_json_ws(self) -> str:
        return self.model_dump_json()


class TranscriptEvent(BaseModel):
    type: str = "transcript"
    text: str
    is_final: bool = False


class ErrorEvent(BaseModel):
    type: str = "error"
    message: str
    detail: str | None = None


def battlecard_from_dict(
    competitor: str,
    confidence: float,
    raw: dict[str, Any],
    client_context: dict[str, Any] | None = None,
) -> BattlecardEvent:
    data = BattlecardData(
        key_differentiator=str(raw.get("key_differentiator", "")),
        suggested_response=str(raw.get("suggested_response", "")),
        recommended_question=str(raw.get("recommended_question", "")),
        weaknesses=list(raw.get("weaknesses") or []),
    )
    if client_context:
        ctx = ClientContext(
            name=client_context.get("name"),
            industry=client_context.get("industry"),
            deal_size=client_context.get("deal_size"),
            pain_points=client_context.get("pain_points"),
        )
    else:
        ctx = ClientContext()
    return BattlecardEvent(
        competitor=competitor,
        confidence=confidence,
        data=data,
        client_context=ctx,
    )
