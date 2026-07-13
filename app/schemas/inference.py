"""Pydantic schemas for stable, backend-friendly JSON responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Confidence(BaseModel):
    action: float = Field(ge=0.0, le=1.0)
    subAction: float = Field(ge=0.0, le=1.0)
    issue: float = Field(ge=0.0, le=1.0)


class Segment(BaseModel):
    index: int = Field(ge=0)
    startMs: int = Field(ge=0)
    endMs: int = Field(ge=0)
    action: str
    subAction: str
    issue: str
    confidence: Confidence
    clipPath: str | None = None


class InferenceTiming(BaseModel):
    modelLoadMs: int = Field(default=0, ge=0)
    poseExtractionMs: int = Field(default=0, ge=0)
    forwardMs: int = Field(default=0, ge=0)
    clipExportMs: int = Field(default=0, ge=0)


class InferenceResponse(BaseModel):
    requestId: str
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    mode: Literal["VIDEO_ONLY", "VIDEO_WITH_IMU"]
    syntheticImu: bool
    modelVersion: str
    device: str
    processingTimeMs: int = Field(ge=0)
    peakGpuMemoryMb: float | None = Field(default=None, ge=0.0)
    timing: InferenceTiming
    segments: list[Segment]
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    status: Literal["FAILED"] = "FAILED"
    errorCode: str
    message: str
    requestId: str | None = None


class HealthResponse(BaseModel):
    status: Literal["UP", "DEGRADED"]
    cudaAvailable: bool
    gpuName: str | None
    modelLoaded: bool
    modelVersion: str


class ModelInfoResponse(BaseModel):
    modelVersion: str
    supportedInputModes: list[str]
    actionClasses: list[str]
    subActionClasses: list[str]
    issueClasses: list[str]
    inputShapes: dict[str, Any]
    device: str
