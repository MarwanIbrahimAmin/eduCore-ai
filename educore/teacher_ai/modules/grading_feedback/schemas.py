
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class GradingQuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PROBLEM_SOLVING = "problem_solving"
    ESSAY = "essay"


class GradingLanguage(str, Enum):
    ARABIC = "Arabic"
    ENGLISH = "English"


class MarkingCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    max_marks: float = Field(gt=0)


class GradingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    question_type: GradingQuestionType
    max_marks: float = Field(gt=0)
    expected_answer: Optional[str] = None
    marking_criteria: list[MarkingCriterion] = Field(default_factory=list)


class StudentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer: str


class GradingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    grade_level: str
    language: GradingLanguage = GradingLanguage.ENGLISH
    teacher_instructions: Optional[str] = None
    course_material: Optional[str] = Field(default=None, max_length=12000)
    questions: list[GradingQuestion] = Field(min_length=1)
    student_answers: list[StudentAnswer] = Field(min_length=1)


class CriterionGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    awarded_marks: float = Field(ge=0)
    feedback: str


class QuestionGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    awarded_marks: float = Field(ge=0)
    max_marks: float = Field(gt=0)
    is_fully_correct: bool
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    criterion_grades: list[CriterionGrade] = Field(default_factory=list)


class GradingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_awarded_marks: float = Field(ge=0)
    total_max_marks: float = Field(gt=0)
    percentage: float = Field(ge=0, le=100)
    overall_feedback: str
    question_grades: list[QuestionGrade]
