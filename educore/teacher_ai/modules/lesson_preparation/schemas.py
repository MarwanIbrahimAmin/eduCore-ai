"""
Schemas for the Lesson Preparation module (Pydantic v2).

* LessonRequest        - what the teacher asks for (unchanged field names/limits).
* GeneratedLesson      - the structure the AI must return, with nested models.
* parse_generated_lesson(data) -> dict
                         validates structure and returns a normalised plain dict
                         (strings stripped, numbers as ints, optional gaps filled).
* check_lesson_against_request(lesson, request) -> RequestCheck
                         checks that need the request: practice-question count,
                         lesson-flow timing, preparation-mode consistency.

Validation philosophy
---------------------
Core fields that a teacher cannot work without (titles, questions, answers,
solution steps, durations) are required and non-empty. Secondary descriptive
fields default to "" so one missing sentence does not discard an otherwise useful
lesson. Structure is validated here; educational correctness is NOT - that is the
job of the verification and review stages.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

# ---------------------------------------------------------------------------
# 1. Request (public interface, unchanged)
# ---------------------------------------------------------------------------


def _none_to_empty(value):
    return "" if value is None else value


class LessonRequest(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    # Basic lesson information
    subject: str = Field(min_length=2, max_length=100)
    grade: str = Field(min_length=1, max_length=50)
    topic: str = Field(min_length=3, max_length=200)

    # Lesson preparation preferences
    preparation_mode: Literal[
        "Prepare my lesson",
        "Understand and prepare"
    ] = "Prepare my lesson"

    language: Literal[
        "English",
        "Arabic",
        "Egyptian Arabic"
    ] = "English"

    duration_minutes: int = Field(default=45, ge=15, le=180)

    # Classroom context
    student_level: Literal[
        "Beginner",
        "Intermediate",
        "Advanced",
        "Mixed"
    ] = "Mixed"

    teaching_style: Literal[
        "Interactive",
        "Visual",
        "Practical",
        "Traditional",
        "Mixed"
    ] = "Interactive"

    # Assessment preferences
    question_count: int = Field(default=5, ge=1, le=20)

    # Optional teacher instructions (treated as untrusted text by the prompts)
    extra_instructions: Annotated[
        str, BeforeValidator(_none_to_empty), Field(max_length=1000)
    ] = ""


# ---------------------------------------------------------------------------
# 2. Reusable field types
# ---------------------------------------------------------------------------

DIFFICULTY_LABELS = ("Basic", "Intermediate", "Advanced")


def _scalar_text(value):
    """Accept text; accept plain numbers as text (answers are often numeric)."""
    if isinstance(value, bool) or value is None:
        raise ValueError("must be text")
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        raise ValueError(f"must be text, not {type(value).__name__}")
    return value.strip()


def _optional_text(value):
    return "" if value is None else _scalar_text(value)


def _text_list(value):
    """A list of text items. A lone string becomes a one-item list; blank items are dropped."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"must be a list of text items, not {type(value).__name__}")
    items = []
    for index, item in enumerate(value):
        if item is None:
            continue
        if isinstance(item, (dict, list, tuple, bool)):
            raise ValueError(f"item {index + 1} must be text, not {type(item).__name__}")
        text = _scalar_text(item)
        if text:
            items.append(text)
    return items


def _minutes(value):
    """Whole, positive minutes. Booleans, strings and fractional values are rejected."""
    if isinstance(value, bool):
        raise ValueError("must be a whole number of minutes, not true/false")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("must be a whole number of minutes")
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"must be a whole number of minutes, not {type(value).__name__}")
    if value <= 0:
        raise ValueError("must be greater than zero")
    return value


def _difficulty(value):
    if isinstance(value, str):
        cleaned = value.strip().capitalize()
        if cleaned in DIFFICULTY_LABELS:
            return cleaned
    raise ValueError("must be exactly one of: Basic, Intermediate, Advanced")


def _strict_bool(value):
    if isinstance(value, bool):
        return value
    raise ValueError("must be true or false")


Text = Annotated[str, BeforeValidator(_scalar_text), Field(min_length=1)]
OptText = Annotated[str, BeforeValidator(_optional_text)]
TextList = Annotated[list[Text], BeforeValidator(_text_list)]
RequiredTextList = Annotated[list[Text], BeforeValidator(_text_list), Field(min_length=1)]
Minutes = Annotated[int, BeforeValidator(_minutes)]
Difficulty = Annotated[Literal["Basic", "Intermediate", "Advanced"], BeforeValidator(_difficulty)]
Flag = Annotated[bool, BeforeValidator(_strict_bool)]


class _Model(BaseModel):
    # Unknown keys are kept (not silently dropped) so nothing the AI wrote disappears.
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# 3. Generated-lesson models
# ---------------------------------------------------------------------------

class LessonOverview(_Model):
    central_idea: Text
    why_it_matters: OptText = ""
    big_picture: OptText = ""
    expected_outcome: Text


class LearningObjective(_Model):
    objective: Text
    success_criteria: OptText = ""


class Prerequisite(_Model):
    knowledge: Text
    why_needed: OptText = ""
    quick_check: OptText = ""


class Foundation(_Model):
    concept: Text
    deep_explanation: Text
    why_it_works: OptText = ""
    important_connections: OptText = ""


class TeacherExample(_Model):
    problem_or_example: Text
    solution: Text
    reasoning: OptText = ""


class TeacherLearning(_Model):
    include: Flag
    foundations: list[Foundation] = Field(default_factory=list)
    teacher_examples: list[TeacherExample] = Field(default_factory=list)
    knowledge_warnings: TextList = Field(default_factory=list)


class TeachingStrategy(_Model):
    recommended_approach: Text
    pedagogical_reason: OptText = ""
    lesson_narrative: OptText = ""
    engagement_strategy: OptText = ""


class TeacherScriptEntry(_Model):
    say: Text
    ask_students: OptText = ""
    expected_response: OptText = ""
    teacher_follow_up: OptText = ""

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data):
        return {"say": data} if isinstance(data, str) else data


class LessonStage(_Model):
    stage: Text
    duration_minutes: Minutes
    purpose: OptText = ""
    teacher_actions: TextList = Field(default_factory=list)
    teacher_script: list[TeacherScriptEntry] = Field(default_factory=list)
    student_actions: TextList = Field(default_factory=list)
    transition: OptText = ""


class StepExplanation(_Model):
    step: Text
    teacher_explanation: Text
    reasoning: OptText = ""


class StudentCheck(_Model):
    question: Text
    expected_answer: Text
    what_wrong_answers_reveal: OptText = ""


class ConceptExplanation(_Model):
    concept: Text
    teaching_goal: OptText = ""
    introduction: OptText = ""
    step_by_step_explanation: list[StepExplanation] = Field(default_factory=list)
    teacher_script: OptText = ""
    board_work: TextList = Field(default_factory=list)
    student_check: Optional[StudentCheck] = None

    @model_validator(mode="after")
    def _needs_content(self):
        if not self.teacher_script and not self.step_by_step_explanation:
            raise ValueError("needs a teacher_script or at least one step_by_step_explanation")
        return self


class SolutionStep(_Model):
    step: OptText = ""
    work: Text
    teacher_explanation: OptText = ""
    reasoning: OptText = ""

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data):
        return {"work": data} if isinstance(data, str) else data


class WorkedExample(_Model):
    title: Text
    difficulty: Difficulty
    problem: Text
    teaching_purpose: OptText = ""
    solution_steps: Annotated[list[SolutionStep], Field(min_length=1)]
    final_answer: Text
    common_student_error: OptText = ""
    how_to_address_it: OptText = ""


class InteractionScenario(_Model):
    situation: Text
    student_response: Text
    teacher_response: Text
    teaching_purpose: OptText = ""


class ClassroomActivity(_Model):
    activity_name: Text
    learning_goal: OptText = ""
    duration_minutes: Minutes
    instructions: TextList = Field(default_factory=list)
    student_task: OptText = ""
    expected_result: OptText = ""
    teacher_guidance: OptText = ""

    @model_validator(mode="after")
    def _needs_content(self):
        if not self.instructions and not self.student_task:
            raise ValueError("needs instructions or a student_task")
        return self


class Misconception(_Model):
    misconception: Text
    why_it_happens: OptText = ""
    how_to_detect: OptText = ""
    how_to_correct: Text


class FormativeCheckpoint(_Model):
    checkpoint: OptText = ""
    question_or_task: Text
    correct_response: Text
    interpretation: OptText = ""
    next_teacher_action: OptText = ""


class PracticeQuestion(_Model):
    question: Text
    difficulty: Difficulty
    step_by_step_solution: RequiredTextList
    correct_answer: Text
    explanation: Text
    common_mistake: OptText = ""
    feedback: OptText = ""


class LessonSummary(_Model):
    key_takeaways: RequiredTextList
    teacher_closing_script: OptText = ""
    final_check_question: OptText = ""
    expected_answer: OptText = ""


class HomeworkTask(_Model):
    question: Text
    purpose: OptText = ""
    difficulty: Difficulty
    answer_or_guidance: Text


class Differentiation(_Model):
    students_needing_support: TextList = Field(default_factory=list)
    students_at_expected_level: TextList = Field(default_factory=list)
    advanced_students: TextList = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_content(self):
        if not (self.students_needing_support or self.students_at_expected_level
                or self.advanced_students):
            raise ValueError("needs at least one adaptation")
        return self


class GeneratedLesson(_Model):
    """All 19 top-level sections are required; the ones a lesson cannot work without are non-empty."""

    lesson_title: Text
    lesson_overview: LessonOverview
    learning_objectives: Annotated[list[LearningObjective], Field(min_length=1)]
    prerequisites: list[Prerequisite]
    teacher_learning: TeacherLearning
    teaching_strategy: TeachingStrategy
    lesson_flow: Annotated[list[LessonStage], Field(min_length=1)]
    concept_explanations: Annotated[list[ConceptExplanation], Field(min_length=1)]
    worked_examples: Annotated[list[WorkedExample], Field(min_length=1)]
    classroom_interaction: list[InteractionScenario]
    classroom_activities: list[ClassroomActivity]
    common_misconceptions: list[Misconception]
    formative_assessment: list[FormativeCheckpoint]
    practice_questions: Annotated[list[PracticeQuestion], Field(min_length=1)]
    lesson_summary: LessonSummary
    homework: list[HomeworkTask]
    differentiation: Differentiation
    teacher_notes: TextList
    post_lesson_reflection: TextList


REQUIRED_SECTIONS = tuple(GeneratedLesson.model_fields.keys())


# ---------------------------------------------------------------------------
# 4. Error reporting
# ---------------------------------------------------------------------------

def path_to_text(path) -> str:
    """('practice_questions', 2, 'correct_answer') -> 'practice_questions[2].correct_answer'"""
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text


@dataclass
class SchemaProblem:
    path: tuple
    message: str

    @property
    def section(self) -> str:
        return str(self.path[0]) if self.path else ""

    def __str__(self) -> str:
        where = path_to_text(self.path)
        return f"{where}: {self.message}" if where else self.message


class LessonSchemaError(ValueError):
    """The generated lesson does not match the required structure."""

    def __init__(self, problems):
        self.problems = list(problems)
        shown = "; ".join(str(p) for p in self.problems[:8])
        extra = f" (+{len(self.problems) - 8} more)" if len(self.problems) > 8 else ""
        super().__init__(f"Invalid lesson structure: {shown}{extra}")

    @property
    def sections(self) -> list:
        seen = []
        for problem in self.problems:
            if problem.section and problem.section not in seen:
                seen.append(problem.section)
        return seen


_FRIENDLY = {
    "missing": "is required but missing",
    "string_too_short": "must not be empty",
    "too_short": "must contain at least one item",
    "list_type": "must be a list",
    "dict_type": "must be an object",
    "model_type": "must be an object",
}


def format_validation_error(exc: ValidationError) -> list:
    problems = []
    for error in exc.errors():
        loc = tuple(error.get("loc", ()))
        kind = error.get("type", "")
        message = _FRIENDLY.get(kind) or str(error.get("msg", "is invalid"))
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]
        problems.append(SchemaProblem(loc, message))
    return problems


def parse_generated_lesson(data: Any) -> dict:
    """Validate ``data`` and return the normalised lesson as a plain dict."""
    if not isinstance(data, dict):
        raise LessonSchemaError([SchemaProblem((), "The generated lesson must be a JSON object.")])
    try:
        model = GeneratedLesson.model_validate(data)
    except ValidationError as exc:
        raise LessonSchemaError(format_validation_error(exc)) from None
    return model.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# 5. Checks that depend on the request
# ---------------------------------------------------------------------------

@dataclass
class RequestIssue:
    section: str
    message: str

    def __str__(self) -> str:
        return f"{self.section}: {self.message}"


@dataclass
class RequestCheck:
    errors: list = field(default_factory=list)    # block the lesson
    warnings: list = field(default_factory=list)  # teacher attention, non-blocking


def check_lesson_against_request(lesson: dict, request) -> RequestCheck:
    """Request-dependent consistency checks on an already schema-valid lesson."""
    result = RequestCheck()
    err, warn = result.errors, result.warnings

    # Practice-question count
    count = len(lesson.get("practice_questions", []))
    if count != request.question_count:
        err.append(RequestIssue(
            "practice_questions",
            f"expected exactly {request.question_count} practice questions but received {count}."))

    # Lesson-flow timing
    stages = lesson.get("lesson_flow", [])
    total = sum(stage["duration_minutes"] for stage in stages)
    if total != request.duration_minutes:
        err.append(RequestIssue(
            "lesson_flow",
            f"stage durations add up to {total} minutes but the lesson is "
            f"{request.duration_minutes} minutes."))

    # Preparation-mode consistency
    learning = lesson.get("teacher_learning", {})
    deep = request.preparation_mode == "Understand and prepare"
    if learning.get("include") is not deep:
        err.append(RequestIssue(
            "teacher_learning",
            f"include is {learning.get('include')} but the selected mode "
            f"'{request.preparation_mode}' requires {deep}."))
    elif deep and not learning.get("foundations"):
        err.append(RequestIssue(
            "teacher_learning",
            "'Understand and prepare' requires at least one foundation concept."))
    elif not deep and (learning.get("foundations") or learning.get("teacher_examples")
                       or learning.get("knowledge_warnings")):
        warn.append(RequestIssue(
            "teacher_learning",
            "'Prepare my lesson' should leave teacher_learning content empty; it will not be shown."))

    # Classroom feasibility (warnings)
    activities = lesson.get("classroom_activities", [])
    activity_total = sum(a["duration_minutes"] for a in activities)
    if activity_total > request.duration_minutes:
        warn.append(RequestIssue(
            "classroom_activities",
            f"activities total {activity_total} minutes, more than the {request.duration_minutes}-minute lesson."))
    longest_stage = max((s["duration_minutes"] for s in stages), default=0)
    for index, activity in enumerate(activities, start=1):
        if activity["duration_minutes"] > longest_stage:
            warn.append(RequestIssue(
                "classroom_activities",
                f"activity {index} ('{activity['activity_name']}') lasts {activity['duration_minutes']} "
                f"minutes, longer than any lesson stage ({longest_stage})."))
    for index, stage in enumerate(stages, start=1):
        if not stage.get("teacher_script"):
            warn.append(RequestIssue(
                "lesson_flow", f"stage {index} ('{stage['stage']}') has no teacher script."))
    for index, concept in enumerate(lesson.get("concept_explanations", []), start=1):
        if not concept.get("student_check"):
            warn.append(RequestIssue(
                "concept_explanations",
                f"concept {index} ('{concept['concept']}') has no check for understanding."))
    if len(lesson.get("worked_examples", [])) == 1:
        warn.append(RequestIssue("worked_examples", "only one worked example was provided."))
    for key in ("prerequisites", "classroom_interaction", "classroom_activities",
                "common_misconceptions", "formative_assessment", "homework",
                "teacher_notes", "post_lesson_reflection"):
        if not lesson.get(key):
            warn.append(RequestIssue(key, "this section is empty."))
    return result
