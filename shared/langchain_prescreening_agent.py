


# Modified PreScreeningAgent class with Azure OpenAI Whisper

# from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
# from langchain_core.prompts import StringPromptTemplate
# import traceback
# from langchain.chains import LLMChain
# from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain_core.prompts import StringPromptTemplate
import traceback
from langchain.chains import LLMChain
# from langchain_core.agents import AgentOutputParser
from langchain_core.outputs import LLMResult

from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser


import warnings
from requests.auth import HTTPBasicAuth
warnings.filterwarnings("ignore")
import os
from twilio.rest import Client

# Azure OpenAI imports
from openai import AzureOpenAI
import tempfile

from langchain.memory import ConversationBufferMemory
# from langchain_core.memory import ConversationBufferMemory
import requests
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import urllib.parse
import io
load_dotenv()

class PreScreeningAgent:
    def __init__(self):
        # Initialize LLM
       
        
        # Initialize Azure OpenAI client for Whisper
        self.whisper_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.gpt_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")   
        )
        
        # Your Whisper deployment name in Azure OpenAI
        self.whisper_deployment_id = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper-1")
        
        
          # Your chat completion deployment name in Azure OpenAI
        self.chat_deployment_id = "gpt-4.1-2"
        
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Setup tools
        self.tools = [
            Tool(
                name="pre_screen_candidates",
                func=self.run_pre_screening,
                description="Conduct AI voice pre-screening for job candidates. Input: {'candidates': [...], 'job_description': '...'}"
            ),
            Tool(
                name="generate_questions",
                func=self.generate_screening_questions,
                description="Generate job-specific screening questions. Input: job_description"
            ),
            Tool(
                name="evaluate_candidate",
                func=self.evaluate_single_candidate,
                description="Evaluate a single candidate's responses. Input: {'candidate': {...}, 'responses': [...]}"
            )
        ]
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
     from langchain_core.language_models.llms import LLM
     from langchain_core.callbacks.manager import CallbackManagerForLLMRun
     from typing import Optional, List, Any
    
     prompt_template = """
     You are an AI recruitment assistant specializing in candidate pre-screening.
    
     Available tools: {tools}
     Tool names: {tool_names}
    
     Current conversation:
     {chat_history}
    
     Human: {input}
    
     Think step by step:
     Thought: I need to understand what the human wants
     Action: [tool_name]
     Action Input: [input_to_tool]
     Observation: [result_from_tool]
     ... (repeat Thought/Action/Action Input/Observation as needed)
     Thought: I now know the final answer
     Final Answer: [final_response]
    
     {agent_scratchpad}
     """
    
     class CustomPromptTemplate(StringPromptTemplate):
        template: str
        tools: List[Tool]
        
        def format(self, **kwargs) -> str:
            kwargs['tools'] = '\n'.join([f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs['tool_names'] = ', '.join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)
    
     # Create a proper LangChain LLM wrapper for Azure OpenAI
     from pydantic import Field

     class AzureOpenAILLM(LLM):
      client: Any = Field(...)
      deployment_id: str = Field(...)

      @property
      def _llm_type(self) -> str:
        return "azure_openai"

      def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
      ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Azure OpenAI API error: {e}")
            return "Error: Unable to generate response"

     prompt = CustomPromptTemplate(
        template=prompt_template,
        tools=self.tools,
        input_variables=["input", "chat_history", "agent_scratchpad"]
     )
    
    #  azure_llm = AzureOpenAILLM(self.azure_openai_client, self.chat_deployment_id)
     azure_llm = AzureOpenAILLM(
     client=self.gpt_client,
     deployment_id="gpt-4.1-2"  # Use your actual chat deployment ID here
     )
     llm_chain = LLMChain(llm=azure_llm, prompt=prompt)
    
     return LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=ReActSingleInputOutputParser(),
        prompt=prompt,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in self.tools]
    )
    def generate_screening_questions(self, job_description: str) -> List[str]:
     """Generate job-specific screening questions using Azure OpenAI"""
    
     prompt = f"""
     Generate exactly   strictly 5 specific screening questions for this job:
    
     Job Description: {job_description}
    
     Questions should be:
     1. Technical/skill-based
     2. Experience-focused
     3. Scenario-based
    
     Format: Return only questions, one per line.
     """
    
     try:
        response = self.gpt_client.chat.completions.create(
            model=self.chat_deployment_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        questions = [q.strip() for q in response_text.split('\n') if q.strip()]
        return questions # Ensure exactly 3 questions
        
     except Exception as e:
        print(f"Error generating questions: {e}")
        return [
            "Tell me about your relevant experience for this role.",
            "Describe a challenging project you've worked on.",
            "How do you stay updated with industry trends?"
        ]

    
    def trigger_twilio_call(self, candidate: Dict, questions: List[str], chat_id: str) -> Dict:
      """Call candidate with questions passed directly in URL parameters."""
    
      account_sid = os.getenv("TWILIO_ACCOUNT_SID")
      auth_token = os.getenv("TWILIO_AUTH_TOKEN")
      client = Client(account_sid, auth_token)
    
      candidate_id = candidate.get("id")
      candidate_phone1 = candidate.get("phone")
    #   candidate_phone1="8887596182"
      if candidate_phone1 and not candidate_phone1.startswith("+91"):
        candidate_phone1="+91"+candidate_phone1
      
    #   candidate_phone = "8887596182"
      candidate_phone = candidate_phone1
      if not candidate_id or not candidate_phone:
        return {"success": False, "error": "Candidate ID or phone missing"}
    
      try:
        # Create session identifier
        session_id = f"{chat_id}_{candidate_id}_{int(time.time())}"
        
        # Encode questions as JSON and URL encode it
        questions_json = json.dumps(questions)
        encoded_questions = urllib.parse.quote(questions_json)
        
        # Build webhook URL with questions data
        webhook_url = f"https://newaiprescreeningwebhook-dkcxc6d5e9ame4a2.centralindia-01.azurewebsites.net/voice/{session_id}?questions={encoded_questions}&chat_id={chat_id}&candidate_id={candidate_id}"
        
        # Initiate Twilio call
        call = client.calls.create(
            from_="+18508459228",
            to=candidate_phone,
            url=webhook_url,
            timeout=30,
            record=True
        )
        
        print(f"Call initiated: {call.sid} for session: {session_id}")
        print(f"Webhook URL: {webhook_url}")
        
        return {
            "success": True,
            "call_sid": call.sid,
            "session_id": session_id,
            "chat_id": chat_id,
            "candidate_id": candidate_id,
            "total_questions": len(questions),
            "questions": questions
        }
        
      except Exception as e:
        print(f"Error initiating call: {str(e)}")
        return {"success": False, "error": f"Failed to initiate call: {str(e)}"}
    
    def transcribe_audio(self, audio_url: str) -> str:
        """Transcribe audio using Azure OpenAI Whisper API"""
        print("Starting audio transcription with Azure OpenAI...")
        
        try:
            # Validate URL
            if not audio_url.startswith("http"):
                raise ValueError(f"Invalid audio URL: {audio_url}")

            # Download audio data
            print("Downloading audio...")
            time.sleep(10)
            audio_response = requests.get(
                audio_url,
                auth=HTTPBasicAuth(
                    os.getenv("TWILIO_ACCOUNT_SID"), 
                    os.getenv("TWILIO_AUTH_TOKEN")
                )
            )
            
            if audio_response.status_code != 200:
                raise ValueError(f"Failed to download audio: {audio_response.status_code}")

            # Check if we got valid audio data
            audio_content = audio_response.content
            if len(audio_content) < 100:
                raise ValueError(f"Audio data too small ({len(audio_content)} bytes) - likely corrupted or empty")

            print(f"Downloaded {len(audio_content)} bytes of audio data")

          
            audio_buffer = io.BytesIO(audio_content)
            audio_buffer.name = "audio.wav"  # Required for OpenAI API
            

            try:
                # Transcribe using Azure OpenAI Whisper
                print("Transcribing audio with Azure OpenAI...")
                
                # with open(temp_audio_file_path, "rb") as audio_file:
                result = self.whisper_client.audio.transcriptions.create(
                        file=audio_buffer,
                        model=self.whisper_deployment_id,
                        language="en"  # Optional: specify language
                    )
                
                print("Transcription completed successfully!")
                return result.text.strip()
            except Exception as e:
                print(f"Azure OpenAI transcription error: {e}")
                print(f"Error type: {type(e).__name__}")
                return ""
                             
        except Exception as e:
            print(f"Transcription error: {e}")
            print(f"Error type: {type(e).__name__}")
            return ""

    def evaluate_answer(self, question: str, answer: str) -> float:
     """Evaluate answer using Azure OpenAI"""
     print("inside evaluate answer")
     if not answer.strip():
        return 0.0
    
     prompt = f"""
     Evaluate this interview answer on a scale of 0-10:
    
     Question: {question}
     Answer: {answer}
    
     Consider:
     - Relevance to question
     - Technical accuracy
     - Communication clarity
     - Experience depth
     - If you find something in answer , treat it as 1 mark
    
     Return only a number between 0-10:
     """
    
     try:
        response = self.gpt_client.chat.completions.create(
            model=self.chat_deployment_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        response_text = response.choices[0].message.content
        print("response of question and answer", response_text)
        
        # Extract number from response
        score_str = ''.join(filter(str.isdigit, response_text.split()[0]))
        score = float(score_str) if score_str else 0.0
        return min(max(score, 0.0), 10.0)  # Clamp between 0-10
        
     except Exception as e:
        print(f"Error evaluating answer: {e}")
        return 0.0
       
       
    def wait_for_responses(
        self,
        session_id: str,
        num_questions: int,
        webhook_base_url: str = "https://newaiprescreeningwebhook-dkcxc6d5e9ame4a2.centralindia-01.azurewebsites.net",
        call_sid: str | None = None,
    ):
        import time, requests
        from twilio.rest import Client

        # Tunable values
        HARD_TIMEOUT = 41                   # seconds before considering "no response" (increased)
        SOFT_TIMEOUT_PER_Q = 40            # inactivity allowance per answered question
        MAX_TIMEOUT = max(300, num_questions * SOFT_TIMEOUT_PER_Q)  # absolute cap (at least 5min)

        start_time = time.time()
        last_progress_time = time.time()
        last_completed = 0

        print(f"Waiting for {num_questions} responses for session {session_id}...")

        while True:
            elapsed = time.time() - start_time

            # If no progress yet and elapsed exceeds HARD_TIMEOUT, check Twilio before quitting
            if elapsed > HARD_TIMEOUT and last_completed == 0:
                # If we have a call_sid, fetch Twilio status and only declare no-answer when Twilio is terminal
                if call_sid:
                    try:
                        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                        client = Client(account_sid, auth_token)
                        call = client.calls(call_sid).fetch()
                        tw_status = str(getattr(call, "status", "")).lower()
                        print(f"Twilio call status (hard-timeout check): {tw_status}")
                        if tw_status in ("failed", "no-answer", "busy", "canceled"):
                            print("Twilio reports terminal status — treating as no_response.")
                            return []
                        else:
                            # call still ringing/in-progress — give it more time
                            print("Call still active according to Twilio — extending wait.")
                            # bump last_progress_time so soft timeout doesn't trigger immediately
                            last_progress_time = time.time()
                            # keep waiting until MAX_TIMEOUT
                    except Exception as e:
                        print(f"Failed to fetch Twilio call status: {e}. Proceeding to treat as no_answer.")
                        return []
                else:
                    # No call_sid to check -> be conservative and wait longer (extend hard timeout by doubling once)
                    if elapsed < HARD_TIMEOUT * 2:
                        print("No call_sid provided. Extending wait briefly before concluding no_response.")
                        time.sleep(3)
                        continue
                    else:
                        print("No call_sid and extended wait expired. Treating as no_response.")
                        return []

            # Soft timeout after some progress
            if last_completed > 0 and (time.time() - last_progress_time) > SOFT_TIMEOUT_PER_Q:
                print("Soft timeout — no new responses. Returning partial responses.")
                break

            # Absolute safety cap
            if elapsed > MAX_TIMEOUT:
                print("Max timeout reached, returning partial responses.")
                break

            # Poll webhook status
            try:
                status_resp = requests.get(f"{webhook_base_url}/status/{session_id}", timeout=8)
                if status_resp.status_code != 200:
                    time.sleep(2)
                    continue

                status_data = status_resp.json()
                if not status_data.get("success"):
                    time.sleep(2)
                    continue

                completed = status_data.get("completed_questions", 0)
                status = status_data.get("status", "unknown")

                print(f"Progress: {completed}/{num_questions} - Status: {status}")

                # Update progress timer
                if completed > last_completed:
                    last_completed = completed
                    last_progress_time = time.time()

                # Interview fully completed
                if status == "completed" or completed >= num_questions:
                    break

            except requests.exceptions.RequestException as e:
                print("Webhook polling error:", e)
            except Exception as e:
                print("Unexpected error while polling webhook:", e)

            time.sleep(2)

        # Fetch responses (if any)
        try:
            resp = requests.get(f"{webhook_base_url}/responses/{session_id}", timeout=15)
            if resp.status_code == 200 and resp.json().get("success"):
                return resp.json().get("responses", [])
        except Exception as e:
            print("Error fetching responses:", e)

        return []

        
    #    
    

# # Also add this helper method to your PreScreeningAgent class
    def check_call_status(self, call_sid: str) -> dict:
       """Check the status of a Twilio call"""
       try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        client = Client(account_sid, auth_token)
        
        call = client.calls(call_sid).fetch()
        return {
            "status": call.status,
            "duration": call.duration,
            "start_time": call.start_time,
            "end_time": call.end_time
        }
       except Exception as e:
        return {"error": str(e)}
    
    def run_pre_screening(self, input_data: str) -> Dict:
      """Main pre-screening function with better error handling"""
      print("Starting pre-screening process...")
      try:
        # Parse input
        data = json.loads(input_data) if isinstance(input_data, str) else input_data
        candidates = data.get("candidates", [])
        job_description = data.get("job_description", "")
        
        if not candidates or not job_description:
            return {"error": "Missing candidates or job description"}
        
        # Generate questions
        print("Generating screening questions...")
        questions = self.generate_screening_questions(job_description)
        print(f"Generated {len(questions)} questions: {questions}")
        
        results = []
        
        for candidate in candidates:
            candidate_name = candidate.get('name', 'Unknown')
            candidate_id = candidate.get('id')
            
            print(f"\n{'='*50}")
            print(f"Starting pre-screening for {candidate_name} (ID: {candidate_id})")
            print(f"{'='*50}")
            
            # Trigger call
            print("Initiating call...")
            call_result = self.trigger_twilio_call(candidate, questions,"1234")
            
            if "error" in call_result:
                print(f"Call failed: {call_result['error']}")
                results.append({
                    "candidate_id": candidate_id,
                    "name": candidate_name,
                    "status": "call_failed",
                    "error": call_result["error"],
                    "score": 0.0
                })
                continue
            
            call_sid = call_result.get("call_sid")
            print(f"Call initiated successfully. SID: {call_sid}")
            
            # Wait a bit for call to connect
            print("Waiting for call to connect...")
            time.sleep(10)
            
            # Check call status
            if call_sid:
                call_status = self.check_call_status(call_sid)
                print(f"Call status: {call_status}")

                # If check_call_status returned an error object, treat as failed
                if call_status.get("error"):
                    print(f"Call fetch error: {call_status['error']}")
                    results.append({
                        "candidate_id": candidate_id,
                        "name": candidate_name,
                        "status": "call_failed",
                        "error": call_status["error"],
                        # "score": 0.0,
                         "resume_url": candidate.get("resume_url", "jmd"),
                    "overall_score": 0.0,
                        "call_sid": call_sid,
                         "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                    })
                    continue

                # Normalize status string (twilio statuses: queued, ringing, in-progress, completed, busy, failed, no-answer, canceled)
                status = str(call_status.get("status", "")).lower()
                if status in ("failed", "no-answer", "busy", "canceled"):
                    print(f"Call ended with status '{status}'. Marking as call_failed/no_response.")
                    results.append({
                        "candidate_id": candidate_id,
                        "name": candidate_name,
                        "status": "call_failed" if status == "failed" else "no_response",
                        "error": f"call_status_{status}",
                         "resume_url": candidate.get("resume_url", "jmd"),
                    "overall_score": 0.0,
                     "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                    })
                    continue

            
            # Wait for responses with longer timeout
            print(f"Waiting for {len(questions)} responses...")
            num_questions=len(questions)
            # print(f"Number of questions: {num_questions}--------------------hejbfejhrbf----------")
            # responses = self.wait_for_responses(call_result.get("session_id"), num_questions, )  # 5 minutes
            responses = self.wait_for_responses(call_result.get("session_id"), num_questions, call_sid=call_sid)

            
            if not responses:
                print(f"No responses received for {candidate_name}")
                results.append({
                    "candidate_id": candidate_id,
                    "name": candidate_name,
                    "status": "no_response",
                    "score": 0.0,
                    "call_sid": call_sid,
                    "resume_url": candidate.get("resume_url", "jmd"),
                    "overall_score": 0.0,
                     "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                })
                continue
            
            print(f"Received {len(responses)} responses, processing...")
            
            # Process responses
            scores = []
            transcripts = []
            
            for i, response in enumerate(responses):
                if i < len(questions):
                    audio_url = response.get("audio_url", "")
                    if audio_url:
                        print(f"Transcribing audio for question {i+1}...")
                        transcript = self.transcribe_audio(audio_url)
                        print(f"Transcript: {transcript[:100]}...")
                        
                        score = self.evaluate_answer(questions[i], transcript)
                        print(f"Score for question {i+1}: {score}/10")
                        
                        scores.append(score)
                        transcripts.append({
                            "question": questions[i],
                            "answer": transcript,
                            "score": score
                        })
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            qualified = avg_score >= 2.0
            
            print(f"Final Results for {candidate_name}:")
            print(f"   Average Score: {avg_score:.2f}/10")
            print(f"   Qualified: {'YES' if qualified else 'NO'}")
            
            results.append({
                "candidate_id": candidate_id,
                "name": candidate_name,
                "phone": candidate.get("phone"),
                "email": candidate.get("email"),
                "status": "completed",
                "overall_score": round(avg_score, 2),
                "individual_scores": scores,
                "responses": transcripts,
                "qualified": qualified,
                "call_sid": call_sid,
                "resume_url": candidate.get("resume_url", ""),
                "job_description_url": data.get("job_description_url", "")
            })
        
        # Sort by score (highest first)
        qualified_candidates = sorted(
            [r for r in results if r.get("qualified", False)],
            key=lambda x: x.get("overall_score", 0),
            reverse=True
        )
        
        print(f"\n{'='*50}")
        print("FINAL SCREENING RESULTS")
        print(f"{'='*50}")
        print(f"Total Candidates: {len(candidates)}")
        print(f"Completed Screenings: {len([r for r in results if r['status'] == 'completed'])}")
        print(f"Qualified Candidates: {len(qualified_candidates)}")
        
        return {
            "status": "success",
            "total_candidates": len(candidates),
            "completed_screenings": len([r for r in results if r["status"] == "completed"]),
            "qualified_count": len(qualified_candidates),
            "qualified_candidates": qualified_candidates,
            "all_results": results
        }
        
      except Exception as e:
        print(f"Pre-screening failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Pre-screening failed: {str(e)}"}
    
    
    def evaluate_single_candidate(self, input_data: str) -> Dict:
        """Evaluate a single candidate's performance"""
        try:
            data = json.loads(input_data) if isinstance(input_data, str) else input_data
            candidate = data.get("candidate", {})
            responses = data.get("responses", [])
            
            if not responses:
                return {"error": "No responses to evaluate"}
            
            scores = [r.get("score", 0) for r in responses]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Detailed analysis
            strengths = []
            weaknesses = []
            
            for response in responses:
                if response.get("score", 0) >= 7:
                    strengths.append(response.get("question", ""))
                elif response.get("score", 0) <= 4:
                    weaknesses.append(response.get("question", ""))
            
            recommendation = "PROCEED" if avg_score >= 6.0 else "REJECT"
            
            return {
                "candidate_name": candidate.get("name"),
                "overall_score": round(avg_score, 2),
                "recommendation": recommendation,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "detailed_scores": scores
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

# Main execution function for external calls
def create_prescreening_agent():
    """Factory function to create agent instance"""
    return PreScreeningAgent()

# Example usage
if __name__ == "__main__":
    agent = create_prescreening_agent()
    
    # Test data
    test_input = {
        "candidates": [
            {"id": 1, "name": "John Doe", "phone": "8887596182"},
           
        ],
        "job_description": "Senior Python Developer with Django and PostgreSQL experience"
    }
    
    result = agent.run_pre_screening(json.dumps(test_input))
    print(json.dumps(result, indent=2))