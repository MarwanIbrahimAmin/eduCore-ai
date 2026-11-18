
from collections import defaultdict

from .schemas import (
    AnalyticsRequest,
    AnalyticsResult,
    StudentInsight,
    TopicInsight,
)


def _percentage(awarded: float, maximum: float) -> float:
    return round((awarded / maximum) * 100, 2)


def analyze_student_performance(
    request: AnalyticsRequest,
) -> AnalyticsResult:
    if not isinstance(request, AnalyticsRequest):
        request = AnalyticsRequest.model_validate(request)

    student_records = defaultdict(list)
    topic_records = defaultdict(list)

    for record in request.records:
        if record.awarded_marks > record.max_marks:
            raise ValueError(
                f"Student {record.student_id}: awarded marks "
                "cannot exceed maximum marks."
            )

        percentage = _percentage(
            record.awarded_marks,
            record.max_marks,
        )

        student_records[record.student_id].append(
            {
                "name": record.student_name,
                "percentage": percentage,
            }
        )

        for topic in record.topic_performance:
            if topic.awarded_marks > topic.max_marks:
                raise ValueError(
                    f"Student {record.student_id}, topic {topic.topic}: "
                    "awarded marks cannot exceed maximum marks."
                )

            topic_records[topic.topic].append(
                _percentage(topic.awarded_marks, topic.max_marks)
            )

    student_insights = []

    for student_id, records in student_records.items():
        percentages = [item["percentage"] for item in records]

        average = round(sum(percentages) / len(percentages), 2)
        latest = percentages[-1]

        if average >= 85:
            status = "Excellent"
        elif average >= request.pass_mark:
            status = "Satisfactory"
        else:
            status = "Needs Support"

        student_insights.append(
            StudentInsight(
                student_id=student_id,
                student_name=records[-1]["name"],
                assessment_count=len(records),
                average_percentage=average,
                latest_percentage=latest,
                performance_status=status,
            )
        )

    topic_insights = []

    for topic_name, percentages in topic_records.items():
        average = round(sum(percentages) / len(percentages), 2)

        if average >= 85:
            level = "Strong"
        elif average >= 60:
            level = "Moderate"
        else:
            level = "Needs Improvement"

        topic_insights.append(
            TopicInsight(
                topic=topic_name,
                average_percentage=average,
                students_count=len(percentages),
                performance_level=level,
            )
        )

    student_averages = [
        student.average_percentage
        for student in student_insights
    ]

    total_awarded = sum(
        record.awarded_marks for record in request.records
    )
    total_maximum = sum(
        record.max_marks for record in request.records
    )

    class_average = round(
        sum(student_averages) / len(student_averages),
        2,
    )

    pass_count = sum(
        student.average_percentage >= request.pass_mark
        for student in student_insights
    )

    pass_rate = round(
        (pass_count / len(student_insights)) * 100,
        2,
    )

    return AnalyticsResult(
        subject=request.subject,
        grade_level=request.grade_level,
        student_count=len(student_insights),
        assessment_count=len(request.records),
        class_average=class_average,
        pass_rate=pass_rate,
        highest_average=max(student_averages),
        lowest_average=min(student_averages),
        students=student_insights,
        topics=topic_insights,
    )
