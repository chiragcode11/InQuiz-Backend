from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import PyPDF2
import io
from datetime import datetime
from bson import ObjectId
import os
import json
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

from .database import get_database
from .models import *
from .ai_service import ai_generator, response_analyzer, conversation_manager
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

@router.get("/elevenlabs/signed-url")
async def get_elevenlabs_signed_url():
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not agent_id or not api_key:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_AGENT_ID and ELEVENLABS_API_KEY must be configured in environment variables"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id={agent_id}",
                headers={"xi-api-key": api_key},
                timeout=10.0
            )
            
            if response.status_code != 200:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get signed URL from ElevenLabs: {response.text}"
                )
                
            data = response.json()
            return JSONResponse(content={"signed_url": data.get("signed_url")})
            
    except httpx.RequestError as e:
        print(f"HTTP request error fetching ElevenLabs signed URL: {e}")
        raise HTTPException(status_code=503, detail="Service Unavailable: Error connecting to ElevenLabs API")
    except Exception as e:
        print(f"Unexpected error fetching ElevenLabs signed URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def event_generator(response_text: str):
    words = response_text.split(" ")
    for i, word in enumerate(words):
        space = " " if i > 0 else ""
        chunk = {
            "id": "chatcmpl-elevenlabs",
            "object": "chat.completion.chunk",
            "created": int(datetime.utcnow().timestamp()),
            "model": "elevenlabs-custom-llm",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": space + word
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.02)
        
    done_chunk = {
        "id": "chatcmpl-elevenlabs",
        "object": "chat.completion.chunk",
        "created": int(datetime.utcnow().timestamp()),
        "model": "elevenlabs-custom-llm",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/elevenlabs/chat/completions")
async def elevenlabs_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    messages = body.get("messages", [])
    
    interview_id = request.query_params.get("interview_id")
    if not interview_id:
        extra_body = body.get("customLlmExtraBody") or body.get("custom_llm_extra_body") or {}
        if isinstance(extra_body, dict):
            interview_id = extra_body.get("interview_id")
            
    if not interview_id:
        raise HTTPException(status_code=400, detail="interview_id is required either as a query parameter or in customLlmExtraBody")
        
    try:
        interview = db.interviews.find_one({"_id": ObjectId(interview_id)})
        if not interview:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        # Update the conversation transcript log in MongoDB
        db_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                db_messages.append({"type": "user_response", "text": content, "timestamp": datetime.utcnow().timestamp()})
            elif role == "assistant":
                db_messages.append({"type": "ai_question", "text": content, "timestamp": datetime.utcnow().timestamp()})
        
        update_data = {"conversation": db_messages}
        if not interview.get("started_at"):
            update_data["started_at"] = datetime.utcnow()
            update_data["status"] = "in_progress"

        db.interviews.update_one(
            {"_id": ObjectId(interview_id)},
            {"$set": update_data}
        )

        responses = interview.get("responses", [])
        current_question_index = len(responses)
        questions = interview.get("questions", [])
        
        if current_question_index >= len(questions):
            response_text = "The interview has already been completed. Thank you!"
            return StreamingResponse(
                event_generator(response_text),
                media_type="text/event-stream"
            )
            
        current_question = questions[current_question_index]
        current_question_text = current_question["question_text"]
        
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        last_user_message = user_messages[-1].get("content", "").strip() if user_messages else ""
        
        if not assistant_messages:
            db.interviews.update_one(
                {"_id": ObjectId(interview_id)},
                {"$set": {"status": "in_progress", "started_at": datetime.utcnow()}}
            )
            response_text = f"Hello! Welcome to your interview. Let's begin. Here is your first question: {current_question_text}"
            return StreamingResponse(
                event_generator(response_text),
                media_type="text/event-stream"
            )
            
        manager_history = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                manager_history.append({"type": "user_response", "text": content})
            elif role == "assistant":
                manager_history.append({"type": "ai_response", "text": content})
                
        conversation_result = await conversation_manager.process_user_input(
            last_user_message,
            current_question_text,
            manager_history,
            {
                "question_type": current_question.get("question_type"),
                "difficulty": current_question.get("difficulty"),
                "expected_answer_points": current_question.get("expected_answer_points", [])
            }
        )
        
        action = conversation_result.get("action", "continue")
        response_quality = conversation_result.get("response_quality", "fair")
        
        if action in ["repeat_question", "clarify_question", "provide_example", "adjust_pace", "confirm_end_interview", "continue_after_declining_end", "provide_hint", "redirect_off_topic"] or response_quality == "off_topic" or (
            action in ["encourage_elaboration", "encourage_more", "follow_up"] and conversation_result.get("continue_listening", False)
        ):
            response_text = conversation_result["response"]
            return StreamingResponse(
                event_generator(response_text),
                media_type="text/event-stream"
            )
            
        elif action == "end_interview_confirmed":
            db.interviews.update_one(
                {"_id": ObjectId(interview_id)},
                {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
            )
            response_text = conversation_result["response"]
            return StreamingResponse(
                event_generator(response_text),
                media_type="text/event-stream"
            )

        elif action == "skip_question":
            db.interviews.update_one(
                {"_id": ObjectId(interview_id)},
                {"$push": {"responses": {
                    "question_id": current_question["id"],
                    "question_text": current_question_text,
                    "user_response": "[SKIPPED]",
                    "response_time": 0,
                    "analysis": {
                        "completeness_score": 0,
                        "accuracy_score": 0,
                        "overall_feedback": "Question skipped by candidate."
                    },
                    "conversation_quality": "poor",
                    "timestamp": datetime.utcnow()
                }}}
            )
            
            next_index = current_question_index + 1
            if next_index < len(questions):
                resume = db.resumes.find_one({"_id": ObjectId(interview.get("resume_id"))})
                resume_data = {"skills": resume.get("skills", []) if resume else []}
                next_question = questions[next_index]
                adapted_res = await ai_generator.adapt_next_question(
                    next_question,
                    "[SKIPPED]",
                    db_messages,
                    resume_data
                )
                adapted_question_text = adapted_res["question_text"]
                db.interviews.update_one(
                    {"_id": ObjectId(interview_id), "questions.id": next_question["id"]},
                    {"$set": {
                        "questions.$.question_text": adapted_question_text,
                        "questions.$.expected_answer_points": adapted_res["expected_answer_points"]
                    }}
                )
                response_text = f"No problem, we can skip that. Let's move to the next question: {adapted_question_text}"
            else:
                response_text = "No problem. You've completed all the questions in your interview. Thank you for your time, and you'll receive detailed feedback shortly."
                db.interviews.update_one(
                    {"_id": ObjectId(interview_id)},
                    {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
                )
                
            return StreamingResponse(
                event_generator(response_text),
                media_type="text/event-stream"
            )
            
        else:
            next_index = current_question_index + 1
            if next_index < len(questions):
                resume = db.resumes.find_one({"_id": ObjectId(interview.get("resume_id"))})
                resume_data = {"skills": resume.get("skills", []) if resume else []}
                next_question = questions[next_index]
                
                # Filter adaptation input to avoid contaminating future questions
                adaptation_input = last_user_message if response_quality not in ["off_topic", "poor", "wrong"] else "[OFF-TOPIC]"
                
                adapted_res = await ai_generator.adapt_next_question(
                    next_question,
                    adaptation_input,
                    db_messages,
                    resume_data
                )
                adapted_question_text = adapted_res["question_text"]
                db.interviews.update_one(
                    {"_id": ObjectId(interview_id), "questions.id": next_question["id"]},
                    {"$set": {
                        "questions.$.question_text": adapted_question_text,
                        "questions.$.expected_answer_points": adapted_res["expected_answer_points"]
                    }}
                )
                response_text = f"Got it. {conversation_result['response']} Now, let's move to the next question: {adapted_question_text}"
            else:
                response_text = f"Excellent! {conversation_result['response']} You've completed all the questions in your interview. Thank you for your time, and you'll receive detailed feedback shortly."
            
            async def run_analysis_and_save(interview_id, current_question, user_input, conversation_quality):
                try:
                    analysis = await response_analyzer.analyze_response(
                        current_question["question_text"],
                        user_input,
                        current_question.get("expected_answer_points", [])
                    )
                    db.interviews.update_one(
                        {"_id": ObjectId(interview_id)},
                        {"$push": {"responses": {
                            "question_id": current_question["id"],
                            "question_text": current_question["question_text"],
                            "user_response": user_input,
                            "response_time": 0,
                            "analysis": analysis,
                            "conversation_quality": conversation_quality,
                            "timestamp": datetime.utcnow()
                        }}}
                    )
                    if next_index >= len(questions):
                        db.interviews.update_one(
                            {"_id": ObjectId(interview_id)},
                            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
                        )
                except Exception as e:
                    print(f"Error in background response analysis: {e}")
                    
            asyncio.create_task(run_analysis_and_save(
                interview_id, 
                current_question, 
                last_user_message, 
                conversation_result.get("response_quality", "fair")
            ))
            
            return StreamingResponse(
                event_generator(response_text),
                media_type="text/event-stream"
            )
            
    except Exception as e:
        print(f"Error in elevenlabs completions: {e}")
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
