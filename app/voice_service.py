import asyncio
from typing import Dict, List
from ai_service import ai_generator, response_analyzer, conversation_manager
from database import get_database
from bson import ObjectId

db = get_database()

class VoiceInterviewManager:
    def __init__(self):
        self.active_interviews: Dict[str, Dict] = {}
    
    async def start_voice_interview(self, interview_id: str):
        try:
            interview = db.interviews.find_one({"_id": ObjectId(interview_id)})
            if not interview:
                return None
            
            print(f"Starting voice interview {interview_id}")
            
            self.active_interviews[interview_id] = {
                "current_question_index": 0,
                "questions": interview["questions"],
                "conversation": [],
                "status": "active",
                "follow_up_count": 0,
                "current_question_attempts": 0
            }
            
            first_question = interview["questions"][0]
            
            self.active_interviews[interview_id]["conversation"].append({
                "type": "ai_question",
                "text": first_question["question_text"],
                "timestamp": asyncio.get_event_loop().time(),
                "question_id": first_question["id"]
            })
            
            print(f"First question: {first_question['question_text']}")
            
            return {
                "message": "Voice interview started",
                "current_question": first_question,
                "question_index": 0,
                "total_questions": len(interview["questions"])
            }
        except Exception as e:
            print(f"Error starting voice interview: {e}")
            return None
    
    async def process_voice_response(self, interview_id: str, user_response: str, response_time: int):
        try:
            if interview_id not in self.active_interviews:
                print(f"Interview {interview_id} not found in active interviews")
                return {"error": "Interview not found or not active"}
            
            interview_data = self.active_interviews[interview_id]
            current_index = interview_data["current_question_index"]
            
            if current_index >= len(interview_data["questions"]):
                print(f"No more questions available for interview {interview_id}")
                return {"error": "No more questions available"}
            
            current_question = interview_data["questions"][current_index]
            
            print(f"Processing response for interview {interview_id}, question {current_index + 1}")
            print(f"User response: {user_response}")

            interview_data["conversation"].append({
                "type": "user_response",
                "text": user_response,
                "timestamp": asyncio.get_event_loop().time(),
                "question_id": current_question["id"]
            })

            conversation_result = await conversation_manager.process_user_input(
                user_response,
                current_question["question_text"],
                interview_data["conversation"],
                {
                    "question_type": current_question.get("question_type"),
                    "difficulty": current_question.get("difficulty"),
                    "expected_answer_points": current_question.get("expected_answer_points", [])
                }
            )
            
            print(f"Conversation analysis: {conversation_result.get('action', 'unknown')}")

            action = conversation_result.get("action", "continue")
            
            response_data = {
                "conversation": interview_data["conversation"],
                "ai_response": conversation_result["response"]
            }

            if action not in ["repeat_question", "clarify_question", "provide_example", "adjust_pace"]:
                interview_data["conversation"].append({
                    "type": "ai_response",
                    "text": conversation_result["response"],
                    "timestamp": asyncio.get_event_loop().time(),
                    "question_id": current_question["id"]
                })
            
            if action in ["repeat_question", "clarify_question", "provide_example", "adjust_pace"]:

                message_type = "ai_repeat" if action == "repeat_question" else "ai_clarification"
                interview_data["conversation"].append({
                    "type": message_type,
                    "text": conversation_result["response"],
                    "timestamp": asyncio.get_event_loop().time(),
                    "question_id": current_question["id"]
                })
                
                response_data["continue_same_question"] = True
                response_data["has_follow_up"] = True
                response_data["follow_up_question"] = conversation_result["response"]
                
            elif action in ["encourage_elaboration", "encourage_more", "follow_up"] and conversation_result.get("continue_listening", False):
                interview_data["follow_up_count"] += 1
                response_data["has_follow_up"] = True
                response_data["follow_up_question"] = conversation_result["response"]

                if interview_data["follow_up_count"] >= 2:
                    await self._move_to_next_question(interview_data, response_data, interview_id)
                    
            elif action == "skip_question":
                await self._move_to_next_question(interview_data, response_data, interview_id)
                
            elif action in ["continue", "move_next"] or not conversation_result.get("continue_listening", True):
                await self._save_response_analysis(interview_id, current_question, user_response, response_time, conversation_result)
                await self._move_to_next_question(interview_data, response_data, interview_id)
                
            else:
                response_data["has_follow_up"] = True
                response_data["follow_up_question"] = conversation_result["response"]
            
            return response_data
            
        except Exception as e:
            print(f"Error processing voice response: {e}")
            return {"error": f"Error processing response: {str(e)}"}
    
    async def _move_to_next_question(self, interview_data: Dict, response_data: Dict, interview_id: str):
        """Move to the next question in the interview"""
        interview_data["current_question_index"] += 1
        interview_data["follow_up_count"] = 0  
        next_index = interview_data["current_question_index"]
        
        print(f"Moving to question {next_index + 1}")
        
        if next_index < len(interview_data["questions"]):
            next_question = interview_data["questions"][next_index]

            transition_message = "Great! Now let's move on to the next question."
            interview_data["conversation"].append({
                "type": "ai_transition",
                "text": transition_message,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            interview_data["conversation"].append({
                "type": "ai_question",
                "text": next_question["question_text"],
                "timestamp": asyncio.get_event_loop().time(),
                "question_id": next_question["id"]
            })
            
            response_data["next_question"] = next_question
            response_data["question_index"] = next_index
            response_data["has_follow_up"] = False
            response_data["transition_message"] = transition_message
            
        else:
            interview_data["status"] = "completed"
            completion_message = "Excellent! You've completed all the questions in your interview. You provided thoughtful responses and demonstrated your skills well. Thank you for your time, and you'll receive detailed feedback shortly."
            
            interview_data["conversation"].append({
                "type": "ai_completion",
                "text": completion_message,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            response_data["interview_completed"] = True
            response_data["completion_message"] = completion_message
            
            print("Interview completed")
            await self.complete_interview(interview_id)
    
    async def _save_response_analysis(self, interview_id: str, current_question: Dict, user_response: str, response_time: int, conversation_result: Dict):
        """Save the response with analysis to database"""
        try:
            analysis = await response_analyzer.analyze_response(
                current_question["question_text"],
                user_response,
                current_question.get("expected_answer_points", [])
            )
            
            db.interviews.update_one(
                {"_id": ObjectId(interview_id)},
                {"$push": {"responses": {
                    "question_id": current_question["id"],
                    "question_text": current_question["question_text"],
                    "user_response": user_response,
                    "response_time": response_time,
                    "analysis": analysis,
                    "conversation_quality": conversation_result.get("response_quality", "fair"),
                    "timestamp": asyncio.get_event_loop().time()
                }}}
            )
        except Exception as e:
            print(f"Error saving response analysis: {e}")
    
    async def complete_interview(self, interview_id: str):
        try:
            if interview_id in self.active_interviews:
                conversation = self.active_interviews[interview_id]["conversation"]
                
                db.interviews.update_one(
                    {"_id": ObjectId(interview_id)},
                    {"$set": {
                        "status": "completed",
                        "conversation": conversation,
                        "completed_at": asyncio.get_event_loop().time()
                    }}
                )
                
                del self.active_interviews[interview_id]
                print(f"Interview {interview_id} completed and cleaned up")
                return True
            return False
        except Exception as e:
            print(f"Error completing interview: {e}")
            return False
    
    def get_conversation(self, interview_id: str):
        if interview_id in self.active_interviews:
            return self.active_interviews[interview_id]["conversation"]
        
        try:
            interview = db.interviews.find_one({"_id": ObjectId(interview_id)})
            if interview and "conversation" in interview:
                return interview["conversation"]
        except Exception as e:
            print(f"Error getting conversation from database: {e}")
        
        return []

voice_manager = VoiceInterviewManager()
