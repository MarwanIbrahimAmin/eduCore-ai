
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AssessmentType(str, Enum):
    QUIZ = "quiz"
    HOMEWORK = "homework"
    EXAM = "exam"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"


class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PROBLEM_SOLVING = "problem_solving"
    ESSAY = "essay"


class Language(str, Enum):
    ARABIC = "Arabic"
    ENGLISH = "English"


class AssessmentRequest(BaseModel):
    subject: str = Field(min_length=1)
    grade: str = Field(min_length=1)
    topic: str = Field(min_length=1)

    assessment_type: AssessmentType
    difficulty: Difficulty

    question_types: list[QuestionType] = Field(min_length=1)
    question_count: int = Field(ge=1, le=100)
    total_marks: float = Field(gt=0, le=1000)
    duration_minutes: int = Field(ge=1, le=300)

    language: Language = Language.ENGLISH
    additional_instructions: str = ""


class AnswerOption(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MarkingPoint(BaseModel):
    description: str = Field(min_length=1)
    marks: float = Field(gt=0)


class AssessmentQuestion(BaseModel):
    question_number: int = Field(ge=1)
    question_type: QuestionType
    difficulty: Difficulty

    question_text: str = Field(min_length=1)
    marks: float = Field(gt=0)

    options: list[AnswerOption] = Field(default_factory=list)
    correct_answer: str | None = None
    explanation: str = ""

    marking_scheme: list[MarkingPoint] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    title: str = Field(min_length=1)
    instructions: list[str] = Field(default_factory=list)

    questions: list[AssessmentQuestion] = Field(min_length=1)

    total_marks: float = Field(gt=0)
    duration_minutes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_assessment(self):
        if len(self.questions) == 0:
            raise ValueError("Assessment must contain questions.")

        actual_marks = sum(q.marks for q in self.questions)

        if abs(actual_marks - self.total_marks) > 0.001:
            raise ValueError(
                f"Question marks sum ({actual_marks}) "
                f"does not match total marks ({self.total_marks})."
            )

        for question in self.questions:
            if question.question_type == QuestionType.MCQ:
                if len(question.options) < 2:
                    raise ValueError(
                        f"Question {question.question_number}: "
                        "MCQ must have at least two options."
                    )

                option_ids = [option.id for option in question.options]

                if len(option_ids) != len(set(option_ids)):
                    raise ValueError(
                        f"Question {question.question_number}: "
                        "Duplicate option IDs."
                    )

                if (question.correct_answer is not None and question.correct_answer not in option_ids):
                    raise ValueError(
                        f"Question {question.question_number}: "
                        "Correct answer must match one option ID."
                    )

        return self


class GeneratedAssessment(BaseModel):
    student_version: AssessmentResult
    teacher_version: AssessmentResult

    @model_validator(mode="after")
    def validate_versions(self):
        student = self.student_version
        teacher = self.teacher_version

        if len(student.questions) != len(teacher.questions):
            raise ValueError("Student and teacher versions must match.")

        if abs(student.total_marks - teacher.total_marks) > 0.001:
            raise ValueError("Assessment versions must have equal total marks.")

        return self
