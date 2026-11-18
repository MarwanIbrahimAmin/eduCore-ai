
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_title: str
    subject: str
    grade_level: str

    student_id: Optional[str] = None
    student_name: Optional[str] = None

    include_recommendations: bool = True


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    content: str


class StudentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_title: str
    subject: str
    grade_level: str

    student_id: str
    student_name: str

    average_percentage: float = Field(ge=0, le=100)
    performance_status: str

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    sections: list[ReportSection] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ClassReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_title: str
    subject: str
    grade_level: str

    student_count: int = Field(ge=1)
    class_average: float = Field(ge=0, le=100)
    pass_rate: float = Field(ge=0, le=100)

    sections: list[ReportSection] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CommunicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient_type: str
    recipient_name: str

    student_name: str
    subject: str
    grade_level: str

    purpose: str
    tone: str = "professional"

    additional_context: Optional[str] = None


class CommunicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient_type: str
    recipient_name: str

    subject_line: str
    message: str

    tone: str
    language: str = "English"
