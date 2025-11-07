# # Modified PreScreeningAgent class with Azure OpenAI Whisper - LangChain 0.3 Compatible
# #jai mata diA
# from langchain_core.tools import tool
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# # from langchain.agents import create_react_agent, AgentExecutor
# from langgraph.prebuilt import create_react_agent
# from langchain.agents import AgentExecutor

# import traceback

# import warnings
# from requests.auth import HTTPBasicAuth
# warnings.filterwarnings("ignore")
# import os
# from twilio.rest import Client

# # Azure OpenAI imports
# from openai import AzureOpenAI
# import tempfile

# from langchain.memory import ConversationBufferMemory
# import requests
# import json
# import time
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# import urllib.parse
# import io
# load_dotenv()

# class PreScreeningAgent:
#     def __init__(self):
#         # Initialize Azure OpenAI clients
#         self.whisper_client = AzureOpenAI(
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#             api_version="2024-02-01",
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#         )
#         self.gpt_client = AzureOpenAI(
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             api_version="2024-02-01",
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")   
#         )
        
#         # Deployment names
#         self.whisper_deployment_id = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper-1")
#         self.chat_deployment_id = "gpt-4.1-2"
        
#         self.memory = ConversationBufferMemory(memory_key="chat_history")
        
#         # Create agent with new LangChain 0.3 pattern
#         self.agent_executor = self._create_agent()
    
#     def _create_agent(self):
#         """Create agent using LangChain 0.3 pattern"""
#         from langchain_core.language_models.llms import LLM
#         from langchain_core.callbacks.manager import CallbackManagerForLLMRun
#         from typing import Optional
#         from pydantic import Field
        
#         # Custom LLM wrapper for Azure OpenAI
#         class AzureOpenAILLM(LLM):
#             client: Any = Field(...)
#             deployment_id: str = Field(...)

#             @property
#             def _llm_type(self) -> str:
#                 return "azure_openai"

#             def _call(
#                 self,
#                 prompt: str,
#                 stop: Optional[List[str]] = None,
#                 run_manager: Optional[CallbackManagerForLLMRun] = None,
#                 **kwargs: Any,
#             ) -> str:
#                 try:
#                     response = self.client.chat.completions.create(
#                         model=self.deployment_id,
#                         messages=[{"role": "user", "content": prompt}],
#                         temperature=0.1,
#                         max_tokens=1000
#                     )
#                     return response.choices[0].message.content
#                 except Exception as e:
#                     print(f"Azure OpenAI API error: {e}")
#                     return "Error: Unable to generate response"

#         # Create tools using @tool decorator
#         tools = self._create_tools()
        
#         # Create LLM
#         azure_llm = AzureOpenAILLM(
#             client=self.gpt_client,
#             deployment_id=self.chat_deployment_id
#         )
        
#         # Create prompt for ReAct agent
#         prompt = PromptTemplate.from_template("""
# Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}
# """)
        
#         # Create ReAct agent
#         agent = create_react_agent(azure_llm, tools)
#         #removing prompt delibartely
        
#         # Create agent executor
#         return AgentExecutor(
#             agent=agent,
#             tools=tools,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_iterations=5
#         )
    
#     def _create_tools(self):
#         """Create tools using LangChain 0.3 @tool decorator"""
        
#         @tool
#         def pre_screen_candidates(input_data: str) -> str:
#             """Conduct AI voice pre-screening for job candidates. 
#             Input should be a JSON string with 'candidates' and 'job_description' keys."""
#             return json.dumps(self.run_pre_screening(input_data))
        
#         @tool
#         def generate_questions(job_description: str) -> str:
#             """Generate job-specific screening questions. 
#             Input should be the job description text."""
#             questions = self.generate_screening_questions(job_description)
#             return json.dumps(questions)
        
#         @tool
#         def evaluate_candidate(input_data: str) -> str:
#             """Evaluate a single candidate's responses. 
#             Input should be a JSON string with 'candidate' and 'responses' keys."""
#             return json.dumps(self.evaluate_single_candidate(input_data))
        
#         return [pre_screen_candidates, generate_questions, evaluate_candidate]
    
#     def generate_screening_questions(self, job_description: str) -> List[str]:
#         """Generate job-specific screening questions using Azure OpenAI"""
        
#         prompt = f"""
# Generate exactly 3 specific screening questions for this job:

# Job Description: {job_description}

# Questions should be:
# 1. Technical/skill-based
# 2. Experience-focused
# 3. Scenario-based

# Format: Return only questions, one per line.
# """
        
#         try:
#             response = self.gpt_client.chat.completions.create(
#                 model=self.chat_deployment_id,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1,
#                 max_tokens=500
#             )
            
#             response_text = response.choices[0].message.content
#             questions = [q.strip() for q in response_text.split('\n') if q.strip()]
#             return questions[:3]
            
#         except Exception as e:
#             print(f"Error generating questions: {e}")
#             return [
#                 "Tell me about your relevant experience for this role.",
#                 "Describe a challenging project you've worked on.",
#                 "How do you stay updated with industry trends?"
#             ]
    
#     def trigger_twilio_call(self, candidate: Dict, questions: List[str], chat_id: str) -> Dict:
#         """Call candidate with questions passed directly in URL parameters."""
        
#         account_sid = os.getenv("TWILIO_ACCOUNT_SID")
#         auth_token = os.getenv("TWILIO_AUTH_TOKEN")
#         client = Client(account_sid, auth_token)
        
#         candidate_id = candidate.get("id")
#         candidate_phone1 = candidate.get("phone")
        
#         if candidate_phone1 and not candidate_phone1.startswith("+91"):
#             candidate_phone1 = "+91" + candidate_phone1
        
#         candidate_phone = candidate_phone1
        
#         if not candidate_id or not candidate_phone:
#             return {"success": False, "error": "Candidate ID or phone missing"}
        
#         try:
#             session_id = f"{chat_id}_{candidate_id}_{int(time.time())}"
#             questions_json = json.dumps(questions)
#             encoded_questions = urllib.parse.quote(questions_json)
            
#             webhook_url = f"https://newaiprescreeningwebhook-dkcxc6d5e9ame4a2.centralindia-01.azurewebsites.net/voice/{session_id}?questions={encoded_questions}&chat_id={chat_id}&candidate_id={candidate_id}"
            
#             call = client.calls.create(
#                 from_="+18508459228",
#                 to=candidate_phone,
#                 url=webhook_url,
#                 timeout=30,
#                 record=True
#             )
            
#             print(f"Call initiated: {call.sid} for session: {session_id}")
            
#             return {
#                 "success": True,
#                 "call_sid": call.sid,
#                 "session_id": session_id,
#                 "chat_id": chat_id,
#                 "candidate_id": candidate_id,
#                 "total_questions": len(questions),
#                 "questions": questions
#             }
            
#         except Exception as e:
#             print(f"Error initiating call: {str(e)}")
#             return {"success": False, "error": f"Failed to initiate call: {str(e)}"}
    
#     def transcribe_audio(self, audio_url: str) -> str:
#         """Transcribe audio using Azure OpenAI Whisper API"""
#         print("Starting audio transcription with Azure OpenAI...")
        
#         try:
#             if not audio_url.startswith("http"):
#                 raise ValueError(f"Invalid audio URL: {audio_url}")

#             print("Downloading audio...")
#             time.sleep(10)
#             audio_response = requests.get(
#                 audio_url,
#                 auth=HTTPBasicAuth(
#                     os.getenv("TWILIO_ACCOUNT_SID"), 
#                     os.getenv("TWILIO_AUTH_TOKEN")
#                 )
#             )
            
#             if audio_response.status_code != 200:
#                 raise ValueError(f"Failed to download audio: {audio_response.status_code}")

#             audio_content = audio_response.content
#             if len(audio_content) < 100:
#                 raise ValueError(f"Audio data too small ({len(audio_content)} bytes)")

#             print(f"Downloaded {len(audio_content)} bytes of audio data")

#             audio_buffer = io.BytesIO(audio_content)
#             audio_buffer.name = "audio.wav"
            
#             try:
#                 print("Transcribing audio with Azure OpenAI...")
#                 result = self.whisper_client.audio.transcriptions.create(
#                     file=audio_buffer,
#                     model=self.whisper_deployment_id,
#                     language="en"
#                 )
                
#                 print("Transcription completed successfully!")
#                 return result.text.strip()
#             except Exception as e:
#                 print(f"Azure OpenAI transcription error: {e}")
#                 return ""
                             
#         except Exception as e:
#             print(f"Transcription error: {e}")
#             return ""

#     def evaluate_answer(self, question: str, answer: str) -> float:
#         """Evaluate answer using Azure OpenAI"""
#         print("inside evaluate answer")
#         if not answer.strip():
#             return 0.0
        
#         prompt = f"""
# Evaluate this interview answer on a scale of 0-10:

# Question: {question}
# Answer: {answer}

# Consider:
# - Relevance to question
# - Technical accuracy
# - Communication clarity
# - Experience depth

# Return only a number between 0-10:
# """
        
#         try:
#             response = self.gpt_client.chat.completions.create(
#                 model=self.chat_deployment_id,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1,
#                 max_tokens=100
#             )
            
#             response_text = response.choices[0].message.content
#             print("response of question and answer", response_text)
            
#             score_str = ''.join(filter(str.isdigit, response_text.split()[0]))
#             score = float(score_str) if score_str else 0.0
#             return min(max(score, 0.0), 10.0)
            
#         except Exception as e:
#             print(f"Error evaluating answer: {e}")
#             return 0.0
    
#     def wait_for_responses(self, session_id: str, num_questions: int, timeout: int = 300, webhook_base_url: str = "https://newaiprescreeningwebhook-dkcxc6d5e9ame4a2.centralindia-01.azurewebsites.net"):
#         """Wait for webhook responses by calling API endpoints."""
#         import requests
#         import time
        
#         start_time = time.time()
#         print(f"Waiting for {num_questions} responses for session {session_id}...")
        
#         while (time.time() - start_time) < timeout:
#             try:
#                 status_response = requests.get(f"{webhook_base_url}/status/{session_id}", timeout=10)
                
#                 if status_response.status_code == 200:
#                     status_data = status_response.json()
                    
#                     if status_data.get("success"):
#                         completed_questions = status_data.get("completed_questions", 0)
#                         status = status_data.get("status", "unknown")
                        
#                         print(f"Progress: {completed_questions}/{num_questions} - Status: {status}")
                        
#                         if status == "completed" or completed_questions >= num_questions:
#                             print(f"Interview completed! Getting responses...")
                            
#                             responses_response = requests.get(f"{webhook_base_url}/responses/{session_id}", timeout=10)
                            
#                             if responses_response.status_code == 200:
#                                 responses_data = responses_response.json()
                                
#                                 if responses_data.get("success"):
#                                     responses = responses_data.get("responses", [])
#                                     print(f"Successfully retrieved {len(responses)} responses!")
                                    
#                                     try:
#                                         requests.delete(f"{webhook_base_url}/session/{session_id}", timeout=5)
#                                     except:
#                                         pass
                                    
#                                     return responses
                    
#             except Exception as e:
#                 print(f"Error checking status: {e}")
            
#             time.sleep(5)
        
#         print(f"Timeout reached. Attempting to get partial responses...")
#         try:
#             responses_response = requests.get(f"{webhook_base_url}/responses/{session_id}", timeout=10)
#             if responses_response.status_code == 200:
#                 responses_data = responses_response.json()
#                 if responses_data.get("success"):
#                     return responses_data.get("responses", [])
#         except:
#             pass
        
#         return []

#     def check_call_status(self, call_sid: str) -> dict:
#         """Check the status of a Twilio call"""
#         try:
#             account_sid = os.getenv("TWILIO_ACCOUNT_SID")
#             auth_token = os.getenv("TWILIO_AUTH_TOKEN")
#             client = Client(account_sid, auth_token)
            
#             call = client.calls(call_sid).fetch()
#             return {
#                 "status": call.status,
#                 "duration": call.duration,
#                 "start_time": call.start_time,
#                 "end_time": call.end_time
#             }
#         except Exception as e:
#             return {"error": str(e)}
    
#     def run_pre_screening(self, input_data: str) -> Dict:
#         """Main pre-screening function"""
#         print("Starting pre-screening process...")
#         try:
#             data = json.loads(input_data) if isinstance(input_data, str) else input_data
#             candidates = data.get("candidates", [])
#             job_description = data.get("job_description", "")
            
#             if not candidates or not job_description:
#                 return {"error": "Missing candidates or job description"}
            
#             print("Generating screening questions...")
#             questions = self.generate_screening_questions(job_description)
#             print(f"Generated {len(questions)} questions: {questions}")
            
#             results = []
            
#             for candidate in candidates:
#                 candidate_name = candidate.get('name', 'Unknown')
#                 candidate_id = candidate.get('id')
                
#                 print(f"\n{'='*50}")
#                 print(f"Starting pre-screening for {candidate_name} (ID: {candidate_id})")
#                 print(f"{'='*50}")
                
#                 call_result = self.trigger_twilio_call(candidate, questions, "1234")
                
#                 if "error" in call_result:
#                     print(f"Call failed: {call_result['error']}")
#                     results.append({
#                         "candidate_id": candidate_id,
#                         "name": candidate_name,
#                         "status": "call_failed",
#                         "error": call_result["error"],
#                         "score": 0.0
#                     })
#                     continue
                
#                 call_sid = call_result.get("call_sid")
#                 print(f"Call initiated successfully. SID: {call_sid}")
                
#                 time.sleep(10)
                
#                 responses = self.wait_for_responses(call_result.get("session_id"), len(questions), timeout=300)
                
#                 if not responses:
#                     results.append({
#                         "candidate_id": candidate_id,
#                         "name": candidate_name,
#                         "status": "no_response",
#                         "score": 0.0,
#                         "call_sid": call_sid
#                     })
#                     continue
                
#                 scores = []
#                 transcripts = []
                
#                 for i, response in enumerate(responses):
#                     if i < len(questions):
#                         audio_url = response.get("audio_url", "")
#                         if audio_url:
#                             transcript = self.transcribe_audio(audio_url)
#                             score = self.evaluate_answer(questions[i], transcript)
                            
#                             scores.append(score)
#                             transcripts.append({
#                                 "question": questions[i],
#                                 "answer": transcript,
#                                 "score": score
#                             })
                
#                 avg_score = sum(scores) / len(scores) if scores else 0.0
#                 qualified = avg_score >= 6.0
                
#                 results.append({
#                     "candidate_id": candidate_id,
#                     "name": candidate_name,
#                     "phone": candidate.get("phone"),
#                     "email": candidate.get("email"),
#                     "status": "completed",
#                     "overall_score": round(avg_score, 2),
#                     "individual_scores": scores,
#                     "responses": transcripts,
#                     "qualified": qualified,
#                     "call_sid": call_sid
#                 })
            
#             qualified_candidates = sorted(
#                 [r for r in results if r.get("qualified", False)],
#                 key=lambda x: x.get("overall_score", 0),
#                 reverse=True
#             )
            
#             return {
#                 "status": "success",
#                 "total_candidates": len(candidates),
#                 "completed_screenings": len([r for r in results if r["status"] == "completed"]),
#                 "qualified_count": len(qualified_candidates),
#                 "qualified_candidates": qualified_candidates,
#                 "all_results": results
#             }
            
#         except Exception as e:
#             print(f"Pre-screening failed: {str(e)}")
#             traceback.print_exc()
#             return {"error": f"Pre-screening failed: {str(e)}"}
    
#     def evaluate_single_candidate(self, input_data: str) -> Dict:
#         """Evaluate a single candidate's performance"""
#         try:
#             data = json.loads(input_data) if isinstance(input_data, str) else input_data
#             candidate = data.get("candidate", {})
#             responses = data.get("responses", [])
            
#             if not responses:
#                 return {"error": "No responses to evaluate"}
            
#             scores = [r.get("score", 0) for r in responses]
#             avg_score = sum(scores) / len(scores) if scores else 0.0
            
#             strengths = []
#             weaknesses = []
            
#             for response in responses:
#                 if response.get("score", 0) >= 7:
#                     strengths.append(response.get("question", ""))
#                 elif response.get("score", 0) <= 4:
#                     weaknesses.append(response.get("question", ""))
            
#             recommendation = "PROCEED" if avg_score >= 6.0 else "REJECT"
            
#             return {
#                 "candidate_name": candidate.get("name"),
#                 "overall_score": round(avg_score, 2),
#                 "recommendation": recommendation,
#                 "strengths": strengths,
#                 "weaknesses": weaknesses,
#                 "detailed_scores": scores
#             }
            
#         except Exception as e:
#             return {"error": f"Evaluation failed: {str(e)}"}


# def create_prescreening_agent():
#     """Factory function to create agent instance"""
#     return PreScreeningAgent()


# if __name__ == "__main__":
#     agent = create_prescreening_agent()
    
#     test_input = {
#         "candidates": [
#             {"id": 1, "name": "John Doe", "phone": "8887596182"},
#         ],
#         "job_description": "Senior Python Developer with Django and PostgreSQL experience"
#     }
    
#     result = agent.run_pre_screening(json.dumps(test_input))
#     print(json.dumps(result, indent=2))

