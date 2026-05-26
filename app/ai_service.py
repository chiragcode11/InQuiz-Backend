import google.generativeai as genai
import os
from typing import List, Dict, Any
from .models import Question, QuestionType, DifficultyLevel, Resume
import json
import re

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class AIQuestionGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate_questions(self, resume_data: Dict, config: Dict) -> List[Question]:
        questions = []
        
        if QuestionType.TECHNICAL in config.get('question_types', []):
            tech_questions = await self._generate_technical_questions(
                resume_data.get('skills', []), 
                config['difficulty']
            )
            questions.extend(tech_questions)
        
        if QuestionType.BEHAVIORAL in config.get('question_types', []):
            behavioral_questions = await self._generate_behavioral_questions(
                config['difficulty']
            )
            questions.extend(behavioral_questions)
        
        if QuestionType.EXPERIENCE in config.get('question_types', []):
            exp_questions = await self._generate_experience_questions(
                resume_data.get('experience', []), 
                config['difficulty']
            )
            questions.extend(exp_questions)
        
        return questions[:config.get('num_questions', 5)]
    
    async def _generate_technical_questions(self, skills: List[str], difficulty: DifficultyLevel) -> List[Question]:
        if not skills:
            return []
        
        prompt = f"""
        Generate technical interview questions for a {difficulty} level candidate with skills: {', '.join(skills[:5])}
        
        For each skill, create 1-2 questions that test:
        - Practical knowledge and application
        - Problem-solving abilities
        - Real-world usage scenarios
        
        Difficulty levels:
        - entry: Basic concepts, definitions, simple usage
        - mid: Implementation details, best practices, debugging
        - senior: Architecture decisions, optimization, leadership in technical choices
        
        Return response as JSON array with format:
        [{{
            "question": "question text",
            "skill": "related skill",
            "expected_points": ["point1", "point2", "point3"],
            "follow_up": "follow up question"
        }}]
        """
        
        try:
            response = self.model.generate_content(prompt)
            questions_data = self._parse_json_response(response.text)
            
            questions = []
            for i, q_data in enumerate(questions_data[:3]):
                question = Question(
                    id=f"tech_{i}",
                    question_text=q_data.get('question', ''),
                    question_type=QuestionType.TECHNICAL,
                    difficulty=difficulty,
                    expected_answer_points=q_data.get('expected_points', []),
                    follow_up_questions=[q_data.get('follow_up', '')]
                )
                questions.append(question)
            
            return questions
        except Exception as e:
            return self._fallback_technical_questions(skills, difficulty)
    
    async def _generate_behavioral_questions(self, difficulty: DifficultyLevel) -> List[Question]:
        prompt = f"""
        Generate behavioral interview questions for a {difficulty} level candidate using STAR method.
        
        Focus areas based on difficulty:
        - entry: Learning ability, teamwork, basic problem-solving
        - mid: Leadership potential, conflict resolution, project management
        - senior: Strategic thinking, mentoring, organizational impact
        
        Return response as JSON array with format:
        [{{
            "question": "question text",
            "focus_area": "area being tested",
            "expected_points": ["point1", "point2", "point3"],
            "follow_up": "follow up question"
        }}]
        
        Generate 3 questions.
        """
        
        try:
            response = self.model.generate_content(prompt)
            questions_data = self._parse_json_response(response.text)
            
            questions = []
            for i, q_data in enumerate(questions_data[:3]):
                question = Question(
                    id=f"behavioral_{i}",
                    question_text=q_data.get('question', ''),
                    question_type=QuestionType.BEHAVIORAL,
                    difficulty=difficulty,
                    expected_answer_points=q_data.get('expected_points', []),
                    follow_up_questions=[q_data.get('follow_up', '')]
                )
                questions.append(question)
            
            return questions
        except Exception as e:
            return self._fallback_behavioral_questions(difficulty)
    
    async def _generate_experience_questions(self, experiences: List[str], difficulty: DifficultyLevel) -> List[Question]:
        if not experiences:
            return []
        
        prompt = f"""
        Based on these work experiences, generate specific questions for a {difficulty} level interview:
        
        Experiences:
        {chr(10).join(experiences[:3])}
        
        Create questions that:
        - Dive deep into specific projects and responsibilities
        - Test problem-solving and decision-making
        - Explore achievements and challenges
        
        Difficulty considerations:
        - entry: Focus on learning and contribution
        - mid: Focus on independence and problem-solving
        - senior: Focus on leadership and strategic impact
        
        Return response as JSON array with format:
        [{{
            "question": "question text",
            "experience_focus": "which experience this targets",
            "expected_points": ["point1", "point2", "point3"],
            "follow_up": "follow up question"
        }}]
        """
        
        try:
            response = self.model.generate_content(prompt)
            questions_data = self._parse_json_response(response.text)
            
            questions = []
            for i, q_data in enumerate(questions_data[:2]):
                question = Question(
                    id=f"exp_{i}",
                    question_text=q_data.get('question', ''),
                    question_type=QuestionType.EXPERIENCE,
                    difficulty=difficulty,
                    expected_answer_points=q_data.get('expected_points', []),
                    follow_up_questions=[q_data.get('follow_up', '')]
                )
                questions.append(question)
            
            return questions
        except Exception as e:
            return self._fallback_experience_questions(experiences, difficulty)

    async def adapt_next_question(self, next_question: Dict, last_user_response: str, conversation_history: List[Dict], resume_data: Dict) -> Dict:
        """
        Rewrite the next question template to blend in a smooth conversational transition
        based on the candidate's last answer, simulating real-world interviews.
        """
        recent_context = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        context_text = "\n".join([f"{msg.get('type', 'message')}: {msg.get('text', '')}" for msg in recent_context])
        
        prompt = f"""
        You are an elite corporate technical and behavioral interviewer. You are transitioning the candidate to the next question.
        
        Last Candidate Response: "{last_user_response}"
        Next Question Template: "{next_question.get('question_text', '')}"
        Next Question Type: "{next_question.get('question_type', '')}"
        Next Question Difficulty: "{next_question.get('difficulty', '')}"
        Candidate Resume Skills: {', '.join(resume_data.get('skills', []))}
        
        Recent Dialogue History:
        {context_text}
        
        Your Goal:
        Rewrite the "Next Question Template" to create a highly professional, adaptive next question.
        1. Start with a natural, brief conversational transition acknowledging the candidate's last answer/points. (Do not say "Excellent job" or sound robotic; make it organic like a human interviewer. e.g. "That makes sense, especially regarding how you handle concurrency. Pivoting a bit...")
        2. Introduce the core topic of the Next Question Template. You may slightly tailor the context of the question to match the candidate's resume skills if it makes the question more relevant.
        3. Do NOT change the target difficulty or the core concept of the original question.
        4. Keep the question concise and clear (maximum 2-3 sentences).
        5. CRITICAL CONSTRAINT: If the candidate's last response was off-topic, unrelated, or skipped, do NOT attempt to connect it to the next question. Instead, use a standard transition like "Moving on to the next question..." and ask the next question exactly as templated, without any tailoring to the off-topic subject.
        6. CRITICAL CONSTRAINT: The rewritten question MUST be a professional interview question focusing entirely on the technical or behavioral topic of the "Next Question Template". Never ask about non-professional, general knowledge, or off-topic subjects.
        
        Return response in this JSON format:
        {{
            "question_text": "The rewritten question text with transition",
            "expected_answer_points": ["point1", "point2", "point3"]
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            data = self._parse_single_json_response(response.text)
            if data and data.get("question_text"):
                return {
                    "question_text": data["question_text"],
                    "expected_answer_points": data.get("expected_answer_points", next_question.get("expected_answer_points", []))
                }
        except Exception as e:
            print(f"Error adapting next question: {e}")
        
        return {
            "question_text": next_question.get("question_text", ""),
            "expected_answer_points": next_question.get("expected_answer_points", [])
        }

    def _parse_single_json_response(self, response_text: str) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            return {}
        except:
            return {}
    
    def _parse_json_response(self, response_text: str) -> List[Dict]:
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return []
        except:
            return []
    
    def _fallback_technical_questions(self, skills: List[str], difficulty: DifficultyLevel) -> List[Question]:
        fallback_questions = {
            DifficultyLevel.ENTRY: [
                "What is {skill} and how have you used it?",
                "Can you explain a simple project you built with {skill}?"
            ],
            DifficultyLevel.MID: [
                "Describe a challenging problem you solved using {skill}",
                "How do you ensure code quality when working with {skill}?"
            ],
            DifficultyLevel.SENIOR: [
                "How would you architect a large-scale system using {skill}?",
                "What are the performance considerations when using {skill}?"
            ]
        }
        
        questions = []
        templates = fallback_questions.get(difficulty, fallback_questions[DifficultyLevel.ENTRY])
        
        for i, skill in enumerate(skills[:2]):
            if i < len(templates):
                question = Question(
                    id=f"tech_fallback_{i}",
                    question_text=templates[i].format(skill=skill),
                    question_type=QuestionType.TECHNICAL,
                    difficulty=difficulty,
                    expected_answer_points=[f"Understanding of {skill}", "Practical experience", "Problem-solving approach"],
                    follow_up_questions=[f"What challenges did you face with {skill}?"]
                )
                questions.append(question)
        
        return questions
    
    def _fallback_behavioral_questions(self, difficulty: DifficultyLevel) -> List[Question]:
        fallback_questions = {
            DifficultyLevel.ENTRY: [
                "Tell me about a time you learned something new quickly",
                "Describe a situation where you worked effectively in a team",
                "How do you handle feedback and criticism?"
            ],
            DifficultyLevel.MID: [
                "Tell me about a time you had to lead a project or initiative",
                "Describe a situation where you had to resolve a conflict",
                "How do you prioritize tasks when facing multiple deadlines?"
            ],
            DifficultyLevel.SENIOR: [
                "Tell me about a time you made a strategic decision that impacted your team",
                "Describe how you mentor and develop junior team members",
                "How do you handle situations where you disagree with senior management?"
            ]
        }
        
        questions = []
        templates = fallback_questions.get(difficulty, fallback_questions[DifficultyLevel.ENTRY])
        
        for i, template in enumerate(templates):
            question = Question(
                id=f"behavioral_fallback_{i}",
                question_text=template,
                question_type=QuestionType.BEHAVIORAL,
                difficulty=difficulty,
                expected_answer_points=["Situation", "Task", "Action", "Result"],
                follow_up_questions=["What would you do differently next time?"]
            )
            questions.append(question)
        
        return questions
    
    def _fallback_experience_questions(self, experiences: List[str], difficulty: DifficultyLevel) -> List[Question]:
        questions = []
        for i, exp in enumerate(experiences[:2]):
            question_text = f"Can you walk me through your role and responsibilities in: {exp[:100]}..."
            
            question = Question(
                id=f"exp_fallback_{i}",
                question_text=question_text,
                question_type=QuestionType.EXPERIENCE,
                difficulty=difficulty,
                expected_answer_points=["Role description", "Key responsibilities", "Main achievements"],
                follow_up_questions=["What was the most challenging aspect of this role?"]
            )
            questions.append(question)
        
        return questions

class ConversationManager:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def process_user_input(self, user_input: str, current_question: str, conversation_history: List[Dict], question_context: Dict = None) -> Dict:
        """
        Process user input and determine appropriate AI response with full conversational intelligence
        """
        user_input_lower = user_input.lower().strip()
        
        # 1. Handle confirmation of early termination first
        is_confirming_end = False
        if conversation_history:
            for msg in reversed(conversation_history):
                msg_type = msg.get("type", "").lower()
                msg_text = msg.get("text", "").lower()
                if ("ai" in msg_type or "assistant" in msg_type) and "are you sure you want to end the interview early" in msg_text:
                    is_confirming_end = True
                    break
        
        if is_confirming_end:
            if any(phrase in user_input_lower for phrase in ['yes', 'yep', 'yeah', 'sure', 'confirm', 'end', 'stop', 'correct', 'do it']):
                return {
                    "action": "end_interview_confirmed",
                    "response": "Alright, I understand. I will end our session now and save your progress. Thank you for your time, and you'll receive your feedback shortly.",
                    "continue_listening": False,
                    "needs_follow_up": False
                }
            else:
                return {
                    "action": "continue_after_declining_end",
                    "response": "No problem! Let's continue with our interview. To recap, here was the question: " + current_question,
                    "continue_listening": True,
                    "needs_follow_up": False
                }
        
        # 2. Check for early termination trigger
        if any(phrase in user_input_lower for phrase in ['end the interview', 'stop the interview', 'terminate', 'quit', 'finish early', 'leave the interview', 'stop now', 'can we end', 'i want to end', 'exit interview']):
            return {
                "action": "confirm_end_interview",
                "response": "Are you sure you want to end the interview early? If you end now, I will save your responses and generate feedback based on what we've completed so far. Please say 'Yes, end the interview' to confirm, or 'No' to continue.",
                "continue_listening": True,
                "needs_follow_up": False
            }

        # 3. Check for hint request
        if any(phrase in user_input_lower for phrase in ['hint', 'give me a hint', 'clue', 'help me with this', 'stuck', 'give me a clue', 'any tips']):
            hint = await self._provide_hint(current_question, question_context)
            return {
                "action": "provide_hint",
                "response": hint,
                "continue_listening": True,
                "needs_follow_up": False
            }

        # 4. Handle other conversational helpers - HIGHEST PRIORITY
        if any(phrase in user_input_lower for phrase in ['repeat', 'again', 'say that again', 'repeat question', 'what was the question', 'can you repeat']):
            return {
                "action": "repeat_question",
                "response": f"Of course! Let me repeat the question: {current_question}",
                "continue_listening": True,
                "needs_follow_up": False
            }
        
        if any(phrase in user_input_lower for phrase in ['clarify', 'explain', 'what do you mean', 'unclear', 'confused', 'don\'t understand', 'rephrase']):
            clarification = await self._clarify_question(current_question, question_context)
            return {
                "action": "clarify_question", 
                "response": clarification,
                "continue_listening": True,
                "needs_follow_up": False
            }
            
        if any(phrase in user_input_lower for phrase in ['skip', 'next question', 'pass', 'i don\'t know', 'no idea', 'not sure']):
            return {
                "action": "skip_question",
                "response": "I understand. That's perfectly fine. Let me give you a moment to think, or we can move on to the next question. What would you prefer?",
                "continue_listening": True,
                "needs_follow_up": False
            }
        
        if any(phrase in user_input_lower for phrase in ['slow down', 'too fast', 'speak slower']):
            return {
                "action": "adjust_pace",
                "response": "I'll speak more slowly. Let me repeat the question at a comfortable pace: " + current_question,
                "continue_listening": True,
                "needs_follow_up": False
            }
        
        if any(phrase in user_input_lower for phrase in ['example', 'give me an example', 'for instance', 'what do you mean by']):
            example_response = await self._provide_example(current_question, question_context)
            return {
                "action": "provide_example",
                "response": example_response,
                "continue_listening": True,
                "needs_follow_up": False
            }
        
        if len(user_input.split()) < 3:
            return {
                "action": "encourage_elaboration",
                "response": "I'd love to hear more about that. Could you elaborate and give me more details about your thoughts or experience?",
                "continue_listening": True,
                "needs_follow_up": True
            }
        
        return await self._analyze_answer_intelligently(user_input, current_question, conversation_history, question_context)
    
    async def _provide_hint(self, question: str, question_context: Dict = None) -> str:
        expected_info = ""
        if question_context and question_context.get("expected_answer_points"):
            expected_info = f"Expected Points: {', '.join(question_context.get('expected_answer_points', []))}"
            
        prompt = f"""
        A candidate is stuck on this interview question: "{question}"
        {expected_info}
        
        Provide a helpful, subtle hint that:
        1. Points the candidate in the right direction without giving away the complete answer.
        2. Encourages them to explain their thinking.
        3. Is concise (1-2 sentences).
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return "Try to think about the core concepts involved, or walk me through your initial thoughts and we can build from there."

    async def _clarify_question(self, question: str, question_context: Dict = None) -> str:
        context_info = ""
        if question_context:
            context_info = f"Question Type: {question_context.get('question_type', '')}\nExpected Areas: {', '.join(question_context.get('expected_answer_points', []))}"
        
        prompt = f"""
        A candidate is confused about this interview question: "{question}"
        
        {context_info}
        
        Provide a helpful clarification that:
        1. Rephrases the question in simpler, clearer terms
        2. Explains what kind of answer or information is expected
        3. Gives a brief example or scenario to illustrate
        4. Is encouraging and supportive
        5. Maintains professional interview tone
        
        Keep the clarification concise (2-3 sentences) and help the candidate understand what you're looking for.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Let me clarify that question for you. {question} - I'm looking for your personal experience, thoughts, or approach to this topic. Take your time and share what comes to mind."
    
    async def _provide_example(self, question: str, question_context: Dict = None) -> str:
        prompt = f"""
        A candidate has asked for an example to better understand this interview question: "{question}"
        
        Provide a helpful example that:
        1. Illustrates what kind of response is expected
        2. Gives a brief scenario or sample answer structure
        3. Encourages the candidate to share their own experience
        4. Is relevant to the question type
        
        Keep it concise and helpful.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"For example, if I were answering this question, I might talk about a specific situation, the actions I took, and the outcome. Now, I'd like to hear about your own experience with this topic."
    
    async def _analyze_answer_intelligently(self, user_input: str, current_question: str, conversation_history: List[Dict], question_context: Dict = None) -> Dict:
        # Get recent conversation context
        recent_context = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        context_text = "\n".join([f"{msg['type']}: {msg['text']}" for msg in recent_context])
        
        # Build context information
        question_info = ""
        if question_context:
            question_info = f"""
            Question Type: {question_context.get('question_type', '')}
            Difficulty Level: {question_context.get('difficulty', '')}
            Expected Points: {', '.join(question_context.get('expected_answer_points', []))}
            """
        
        prompt = f"""
        You are an experienced, adaptive AI interviewer analyzing a candidate's response. Be conversational, supportive, but honest in your analysis.

        Current Question: {current_question}
        {question_info}
        
        Candidate's Answer: {user_input}
        
        Recent Conversation Context:
        {context_text}

        Analyze this response and determine the most appropriate next action. Return JSON in this exact format:
        {{
            "action": "continue|follow_up|provide_feedback|encourage_more|correct_misunderstanding|move_next",
            "response_quality": "excellent|good|fair|poor|off_topic|wrong",
            "is_relevant": true/false,
            "completeness_score": 1-10,
            "accuracy_score": 1-10,
            "needs_follow_up": true/false,
            "ai_response": "What you should say next - be conversational and adaptive",
            "follow_up_question": "Specific follow-up question if needed",
            "feedback": "Brief encouraging feedback",
            "next_action": "continue_listening|move_to_next_question"
        }}

        AI Response Guidelines & Strict Context Constraints:
        - CRITICAL CONSTRAINT: Under no circumstances should you engage in chit-chat, talk about unrelated topics (such as food/cooking, recipes, sports, hobbies, movies, weather, etc.), or answer unrelated questions.
        - OFF-TOPIC DEFINITION: Any response that does not attempt to answer the current interview question or talks about unrelated concepts (e.g., food/cooking, sports, hobbies, personal chit-chat, asking the interviewer personal/philosophical questions) is strictly OFF-TOPIC.
        - If the candidate's answer is OFF-TOPIC: You must set "action" to "redirect_off_topic", set "response_quality" to "off_topic", set "needs_follow_up" to true, and set "ai_response" to a polite but firm redirection back to the active interview question. e.g., "I see. Let's make sure we stay focused on our interview topic. To get back on track, let me repeat the question: [repeat question]"
        - Keep all follow-up questions strictly scoped to the active interview question's technical/behavioral domain. Do not deviate to any unrelated topics.
        - If answer is WRONG or POOR: Provide gentle correction, explain the right approach, then ask a clarifying question
        - If answer is INCOMPLETE: Ask for more details or examples
        - If answer is GOOD/EXCELLENT: Acknowledge strengths and move forward
        - Be conversational and natural, like a real interviewer
        - Show you're listening by referencing their specific points
        - For wrong answers, say something like: "I appreciate your effort, but let me help clarify this concept..." or "That's not quite right, let me explain..."
        - For good answers: "Excellent point about...", "That's a solid approach..."
        - Adapt your language to match the candidate's communication style
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            
            if not result:
                return self._fallback_intelligent_analysis(user_input, current_question)
            
            return {
                "action": result.get("action", "continue"),
                "response": result.get("ai_response", "Thank you for that response."),
                "follow_up_question": result.get("follow_up_question", ""),
                "feedback": result.get("feedback", ""),
                "needs_follow_up": result.get("needs_follow_up", False),
                "quality_score": result.get("completeness_score", 7),
                "accuracy_score": result.get("accuracy_score", 7),
                "continue_listening": result.get("next_action", "move_to_next_question") == "continue_listening",
                "response_quality": result.get("response_quality", "fair")
            }
            
        except Exception as e:
            print(f"Intelligence analysis error: {e}")
            return self._fallback_intelligent_analysis(user_input, current_question)    

    def _fallback_intelligent_analysis(self, user_input: str, current_question: str) -> Dict:
        word_count = len(user_input.split())
        
        # Check for common wrong answer indicators
        wrong_indicators = ['i dont know', 'no idea', 'not sure', 'maybe', 'i think', 'probably']
        has_uncertainty = any(indicator in user_input.lower() for indicator in wrong_indicators)
        
        if word_count < 5 or has_uncertainty:
            return {
                "action": "provide_feedback",
                "response": "I can see you're uncertain about this. Let me help clarify the concept and give you another chance to think about it. This is a common area that many candidates find challenging.",
                "continue_listening": True,
                "needs_follow_up": True,
                "response_quality": "poor"
            }
        elif word_count < 15:
            return {
                "action": "encourage_more", 
                "response": "That's a good start! I can see you understand some aspects. Could you elaborate more and perhaps give me a specific example or walk me through your thought process?",
                "continue_listening": True,
                "needs_follow_up": True,
                "response_quality": "fair"
            }
        else:
            return {
                "action": "continue",
                "response": "Thank you for that comprehensive answer. I can see you have good knowledge in this area and you've explained your thinking clearly.",
                "continue_listening": False,
                "needs_follow_up": False,
                "response_quality": "good"
            }

    def _parse_json_response(self, response_text: str) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            return {}
        except:
            return {}

class ResponseAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def analyze_response(self, question: str, user_response: str, expected_points: List[str]) -> Dict:
        prompt = f"""
        Analyze this interview response comprehensively:
        
        Question: {question}
        Response: {user_response}
        Expected Key Points: {', '.join(expected_points)}
        
        Provide analysis in JSON format:
        {{
            "completeness_score": 0-10,
            "accuracy_score": 0-10,
            "clarity_score": 0-10,
            "relevance_score": 0-10,
            "depth_score": 0-10,
            "missing_points": ["point1", "point2"],
            "strengths": ["strength1", "strength2"],
            "areas_for_improvement": ["improvement1", "improvement2"],
            "overall_feedback": "Comprehensive feedback summary",
            "follow_up_needed": true/false,
            "suggested_follow_up": "follow up question if needed"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_analysis_response(response.text)
        except Exception as e:
            return self._fallback_analysis(user_response)
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return self._fallback_analysis("")
        except:
            return self._fallback_analysis("")
    
    def _fallback_analysis(self, user_response: str) -> Dict:
        response_length = len(user_response.split())
        
        if response_length < 10:
            completeness = 3
            clarity = 4
            depth = 2
        elif response_length < 30:
            completeness = 6
            clarity = 7
            depth = 5
        elif response_length < 60:
            completeness = 8
            clarity = 8
            depth = 7
        else:
            completeness = 9
            clarity = 8
            depth = 8
        
        return {
            "completeness_score": completeness,
            "accuracy_score": 7,
            "clarity_score": clarity,
            "relevance_score": 7,
            "depth_score": depth,
            "missing_points": [],
            "strengths": ["Provided thoughtful response", "Good communication"],
            "areas_for_improvement": ["Could provide more specific examples"] if response_length < 30 else [],
            "overall_feedback": "Good response with room for more detail" if response_length < 30 else "Comprehensive and well-structured response",
            "follow_up_needed": response_length < 20,
            "suggested_follow_up": "Could you elaborate on that with a specific example?" if response_length < 20 else ""
        }

ai_generator = AIQuestionGenerator()
conversation_manager = ConversationManager()
response_analyzer = ResponseAnalyzer()
