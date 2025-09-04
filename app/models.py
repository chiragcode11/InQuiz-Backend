from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Annotated, Union
from datetime import datetime
from bson import ObjectId
from enum import Enum
from pydantic import AfterValidator, PlainSerializer, WithJsonSchema

def validate_object_id(v: Any) -> ObjectId:
    if isinstance(v, ObjectId):
        return v
    if ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")

PyObjectId = Annotated[
    Union[str, ObjectId],
    AfterValidator(validate_object_id),
    PlainSerializer(lambda x: str(x), return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]

class DifficultyLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"

class QuestionType(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    EXPERIENCE = "experience"
    SITUATIONAL = "situational"

class Resume(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    filename: str
    content: str
    parsed_data: Dict[str, Any]
    skills: List[str] = []
    experience: List[str] = []
    education: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Question(BaseModel):
    id: str
    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    expected_answer_points: List[str] = []
    follow_up_questions: List[str] = []

class InterviewResponse(BaseModel):
    question_id: str
    question_text: str
    user_response: str
    response_time: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class InterviewSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    resume_id: str
    difficulty: DifficultyLevel
    questions: List[Question] = []
    responses: List[InterviewResponse] = []
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class InterviewFeedback(BaseModel):
    interview_id: str
    overall_score: float
    technical_score: float
    behavioral_score: float
    communication_score: float
    strengths: List[str] = []
    improvements: List[str] = []
    detailed_feedback: str
    transcription: List[Dict[str, str]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class InterviewConfig(BaseModel):
    difficulty: DifficultyLevel
    duration_minutes: int = 20
    question_types: List[QuestionType] = [QuestionType.TECHNICAL, QuestionType.BEHAVIORAL]
    num_questions: int = 5
