from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import PyPDF2
import io
from datetime import datetime
from bson import ObjectId

from .database import get_database
from .models import *
from .ai_service import ai_generator, response_analyzer
from .voice_service import voice_manager

router = APIRouter()
db = get_database()

@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
        
        parsed_data = {
            "raw_text": text_content,
            "page_count": len(pdf_reader.pages),
            "file_size": len(contents)
        }
        
        skills = extract_skills(text_content)
        experience = extract_experience(text_content)
        education = extract_education(text_content)
        
        resume = Resume(
            filename=file.filename,
            content=text_content,
            parsed_data=parsed_data,
            skills=skills,
            experience=experience,
            education=education
        )
        
        result = db.resumes.insert_one(resume.dict(by_alias=True, exclude={'id'}))
        resume.id = result.inserted_id
        
        return JSONResponse(content={
            "message": "Resume uploaded successfully",
            "resume_id": str(resume.id),
            "skills": skills,
            "experience": experience,
            "education": education,
            "preview": text_content[:200] + "..."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@router.get("/resume/{resume_id}")
async def get_resume(resume_id: str):
    try:
        resume = db.resumes.find_one({"_id": ObjectId(resume_id)})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        resume['_id'] = str(resume['_id'])
        return resume
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-questions/{resume_id}")
async def generate_questions(resume_id: str, config: InterviewConfig):
    try:
        resume = db.resumes.find_one({"_id": ObjectId(resume_id)})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        resume_data = {
            "skills": resume.get("skills", []),
            "experience": resume.get("experience", []),
            "education": resume.get("education", []),
            "content": resume.get("content", "")
        }
        
        config_dict = {
            "difficulty": config.difficulty,
            "question_types": config.question_types,
            "num_questions": config.num_questions,
            "duration_minutes": config.duration_minutes
        }
        
        questions = await ai_generator.generate_questions(resume_data, config_dict)
        
        interview_session = InterviewSession(
            resume_id=resume_id,
            difficulty=config.difficulty,
            questions=questions,
            status="ready"
        )
        
        result = db.interviews.insert_one(interview_session.dict(by_alias=True, exclude={'id'}))
        interview_session.id = result.inserted_id
        
        return JSONResponse(content={
            "interview_id": str(interview_session.id),
            "questions": [q.dict() for q in questions],
            "total_questions": len(questions),
            "estimated_duration": config.duration_minutes
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interview/{interview_id}")
async def get_interview(interview_id: str):
    try:
        interview = db.interviews.find_one({"_id": ObjectId(interview_id)})
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        interview['_id'] = str(interview['_id'])
        return interview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interview/{interview_id}/question/{question_index}")
async def get_current_question(interview_id: str, question_index: int):
    try:
        interview = db.interviews.find_one({"_id": ObjectId(interview_id)})
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        questions = interview.get("questions", [])
        if question_index >= len(questions):
            return JSONResponse(content={"message": "No more questions", "completed": True})
        
        current_question = questions[question_index]
        
        return JSONResponse(content={
            "question": current_question,
            "question_number": question_index + 1,
            "total_questions": len(questions),
            "is_last": question_index == len(questions) - 1
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview/{interview_id}/start")
async def start_interview(interview_id: str):
    try:
        result = db.interviews.update_one(
            {"_id": ObjectId(interview_id)},
            {"$set": {"status": "in_progress", "started_at": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return JSONResponse(content={"message": "Interview started", "status": "in_progress"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview/{interview_id}/response")
async def submit_response(interview_id: str, response: InterviewResponse):
    try:
        interview = db.interviews.find_one({"_id": ObjectId(interview_id)})
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        question = next((q for q in interview["questions"] if q["id"] == response.question_id), None)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        analysis = await response_analyzer.analyze_response(
            question["question_text"],
            response.user_response,
            question.get("expected_answer_points", [])
        )
        
        response_with_analysis = response.dict()
        response_with_analysis["analysis"] = analysis
        
        result = db.interviews.update_one(
            {"_id": ObjectId(interview_id)},
            {"$push": {"responses": response_with_analysis}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return JSONResponse(content={
            "message": "Response recorded",
            "analysis": analysis,
            "follow_up_needed": analysis.get("follow_up_needed", False),
            "suggested_follow_up": analysis.get("suggested_follow_up", "")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview/{interview_id}/complete")
async def complete_interview(interview_id: str):
    try:
        result = db.interviews.update_one(
            {"_id": ObjectId(interview_id)},
            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return JSONResponse(content={"message": "Interview completed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview/{interview_id}/start-voice")
async def start_voice_interview(interview_id: str):
    try:
        print(f"Starting voice interview: {interview_id}")
        result = await voice_manager.start_voice_interview(interview_id)
        if not result:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        db.interviews.update_one(
            {"_id": ObjectId(interview_id)},
            {"$set": {"status": "in_progress", "started_at": datetime.utcnow()}}
        )
        
        print(f"Voice interview started successfully: {interview_id}")
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error starting voice interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview/{interview_id}/voice-response")
async def process_voice_response(interview_id: str, response_data: dict):
    try:
        user_response = response_data.get("response", "")
        response_time = response_data.get("response_time", 0)
        
        print(f"Received voice response for interview {interview_id}")
        print(f"Response: {user_response[:100]}{'...' if len(user_response) > 100 else ''}")
        
        if not user_response.strip():
            raise HTTPException(status_code=400, detail="Empty response")
        
        result = await voice_manager.process_voice_response(
            interview_id, user_response, response_time
        )
        
        if "error" in result:
            print(f"Error in processing: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        has_follow_up = result.get('has_follow_up', False)
        has_next = result.get('next_question') is not None
        is_completed = result.get('interview_completed', False)
        
        print(f"Processing result - Follow-up: {has_follow_up}, Next: {has_next}, Completed: {is_completed}")
        
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error processing voice response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interview/{interview_id}/conversation")
async def get_conversation(interview_id: str):
    try:
        conversation = voice_manager.get_conversation(interview_id)
        return JSONResponse(content={"conversation": conversation})
    except Exception as e:
        print(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview/{interview_id}/complete-voice")
async def complete_voice_interview(interview_id: str):
    try:
        result = await voice_manager.complete_interview(interview_id)
        if result:
            return JSONResponse(content={"message": "Voice interview completed successfully"})
        else:
            raise HTTPException(status_code=404, detail="Interview not found")
    except Exception as e:
        print(f"Error completing voice interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def extract_skills(text: str) -> List[str]:
    programming_languages = ["python", "javascript", "java", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust", "typescript"]
    frameworks = ["react", "angular", "vue", "django", "flask", "fastapi", "express", "spring", "laravel", "rails"]
    databases = ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "cassandra"]
    cloud_tools = ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "gitlab", "github"]
    concepts = ["machine learning", "data science", "blockchain", "api", "microservices", "devops", "agile", "scrum"]
    
    all_skills = programming_languages + frameworks + databases + cloud_tools + concepts
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in all_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill.title())
    
    return list(set(found_skills))

def extract_experience(text: str) -> List[str]:
    lines = text.split('\n')
    experience = []
    
    keywords = ['worked', 'developed', 'managed', 'led', 'created', 'designed', 'implemented', 'built', 'maintained', 'deployed']
    
    for line in lines:
        line_clean = line.strip()
        if len(line_clean) > 30 and any(keyword in line_clean.lower() for keyword in keywords):
            if not any(edu_word in line_clean.lower() for edu_word in ['university', 'college', 'school', 'education']):
                experience.append(line_clean)
    
    return experience[:5]

def extract_education(text: str) -> List[str]:
    lines = text.split('\n')
    education = []
    
    edu_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma', 'certification', 'institute']
    
    for line in lines:
        line_clean = line.strip()
        if len(line_clean) > 15 and any(keyword in line_clean.lower() for keyword in edu_keywords):
            education.append(line_clean)
    
    return education[:3]
