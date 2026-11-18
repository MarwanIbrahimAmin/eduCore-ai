
import json

from educore.teacher_ai.core.llm_client import llm_client

from .schemas import CommunicationRequest, CommunicationResult


SYSTEM_PROMPT = """
You are an experienced school communication assistant.

Your task is to write clear, respectful, professional messages
for teachers communicating with parents, students, or school staff.

Rules:
- Use only the information provided by the user.
- Never invent grades, achievements, incidents, or student details.
- Keep the message constructive, specific, and supportive.
- Adapt the tone to the requested tone.
- Avoid harsh, judgmental, or embarrassing language.
- Return valid JSON only.
- Do not include Markdown code fences.

Required JSON structure:
{
    "subject_line": "Message subject",
    "message": "Complete message",
    "tone": "professional",
    "language": "English"
}
"""


def generate_communication(
    request: CommunicationRequest,
) -> CommunicationResult:

    if not isinstance(request, CommunicationRequest):
        request = CommunicationRequest.model_validate(request)

    user_prompt = f"""
Recipient type: {request.recipient_type}
Recipient name: {request.recipient_name}
Student name: {request.student_name}
Subject: {request.subject}
Grade level: {request.grade_level}
Purpose: {request.purpose}
Requested tone: {request.tone}
Additional context: {request.additional_context or "None"}

Write an appropriate message for this situation.

Return JSON only.
"""

    response = llm_client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=2000,
    )

    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(
            "The AI returned invalid JSON for the communication message."
        )

    return CommunicationResult(
        recipient_type=request.recipient_type,
        recipient_name=request.recipient_name,
        subject_line=data["subject_line"],
        message=data["message"],
        tone=data.get("tone", request.tone),
        language=data.get("language", "English"),
    )
