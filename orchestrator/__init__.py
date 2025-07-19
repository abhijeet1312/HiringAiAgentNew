import json
import logging
import azure.durable_functions as df
from datetime import datetime
from typing import List, Dict, Any

def orchestrator_function(context: df.DurableOrchestrationContext):
    """
    Fixed orchestrator that properly handles activity results
    """
    logging.info("Starting orchestrator function")
    
    try:
        # Get input data
        input_data = context.get_input()
        chat_id = input_data.get("chat_id")
        resume_urls = input_data.get("resume_urls", [])
        job_description_urls = input_data.get("job_description_urls", [])
        notification_email = input_data.get("notification_email", "")
        voice_interview_threshold = input_data.get("voice_interview_threshold", 3.0)
        
        logging.info(f"Processing hiring workflow for chat_id: {chat_id}")
        logging.info(f"Total resumes to process: {len(resume_urls)}")
        logging.info(f"Total job descriptions: {len(job_description_urls)}")
        
        # PHASE 1: INPUT VALIDATION
        if not resume_urls or not job_description_urls:
            error_msg = "Invalid input: resume_urls and job_description_urls are required"
            logging.error(error_msg)
            yield context.call_activity("log_error_activity", {
                "chat_id": chat_id,
                "error": error_msg,
                "step": "input_validation"
            })
            return {"status": "failed", "error": error_msg, "step": "input_validation"}
        
        # PHASE 2: RESUME SCREENING
        logging.info("Starting resume screening")
        
        # Create screening tasks for each resume
        screening_tasks = []
        for i, resume_url in enumerate(resume_urls):
            task_input = {
                "resume_url": resume_url,
                "job_description_urls": job_description_urls,
                "chat_id": chat_id,
                "resume_index": i
            }
            logging.info(f"Creating screening task {i} for resume: {resume_url}")
            task = context.call_activity("screen_single_resume_activity", task_input)
            screening_tasks.append(task)
        
        if not context.is_replaying:
          logging.info(f"🚀 Executing {len(screening_tasks)} screening tasks in parallel")
        logging.info(f"Executing {len(screening_tasks)} screening tasks in parallel")
        try:
        #  result1 = yield context.call_activity("jaimatadi", "Seattle")
        #  # Option 1 (safe and recommended):
         logging.info(f"-------jai mata di:---Screenings Starts")
        #  screening_results = yield context.task_all(screening_tasks)
         screening_results={
            "resume_url": "https://aurjobsaiagentsstorage.blob.core.windows.net/aurjobs123/resumes/17527406413345d279b8eAbhijeet_Srivastava_resume_2025.pdf",
            "resume_index": 0,
            "filename": "17527406413345d279b8eAbhijeet_Srivastava_resume_2025.pdf",
            "chat_id": "ssntssns11",
            "job_description_url": "https://aurjobsaiagentsstorage.blob.core.windows.net/aurjobs123/job_descriptions/17528279065454baa842eJob_Description.pdf",
            "candidate_name": "Abhijeet Srivastava",
            "recommendation": "Potential Match",
            "candidate_email": "shivamsrivastava2189@gmail.com",
            "candidate_phone": "8887596182",
            "status": "PASS"}
                    
         logging.info(f"{screening_results}------ screening results")
         
         
        # Logging: After resuming from task_all
         if not context.is_replaying:
          logging.info("✅ All screening tasks completed")
          logging.info(f"📦 Results: {screening_results}")
          logging.info("✅ Resumed after task_all")
          logging.info(f"🧪 Type of screening_results: {type(screening_results)}")
          logging.info(f"📦 Raw screening_results: {screening_results}")
        except Exception as e:
            logging.error(f"❌ Error executing screening tasks: {str(e)}")
            return {"status": "failed", "error": str(e), "step": "screening"}
       
       
        
      
        # Process results
        valid_screenings = []
        failed_screenings = []
        
        for i, result in enumerate(screening_results):
            if result is None:
                logging.warning(f"Result {i} is None")
                failed_screenings.append({"error": "Result is None", "index": i})
                continue
                
            if not isinstance(result, dict):
                logging.warning(f"Result {i} is not a dict: {type(result)}")
                failed_screenings.append({"error": f"Result is {type(result)}", "index": i})
                continue
                
            if result.get("error"):
                logging.warning(f"Result {i} has error: {result.get('error')}")
                failed_screenings.append(result)
                continue
                
            # Valid result
            valid_screenings.append(result)
            logging.info(f"✅ Valid result {i}: {result.get('candidate_name', 'Unknown')}")
        
        logging.info(f"Processing complete: {len(valid_screenings)} valid, {len(failed_screenings)} failed")
        
        # Check if we have any valid results
        if not valid_screenings:
            logging.error("No valid screening results")
            return {
                "status": "failed",
                "error": "No valid screening results",
                "step": "screening",
                "total_resumes": len(resume_urls),
                "failed_screenings": failed_screenings
            }
        
        # Filter qualified candidates
        qualified_candidates = []
        for result in valid_screenings:
            if (result.get("status") == "PASS" and 
                result.get("qualifies_for_voice_interview", False)):
                qualified_candidates.append(result)
        
        logging.info(f"Qualified candidates: {len(qualified_candidates)}/{len(valid_screenings)}")
        
        if not qualified_candidates:
            logging.warning("No qualified candidates found")
            return {
                "status": "completed",
                "chat_id": chat_id,
                "total_candidates": len(resume_urls),
                "resume_qualified": 0,
                "voice_qualified": 0,
                "final_hired": 0,
                "message": "No qualified candidates found after screening",
                "screening_results": valid_screenings,
                "failed_screenings": failed_screenings
            }
        
        # PHASE 2A: SEND EMAILS TO SCREENED CANDIDATES
        logging.info("Sending emails to screened candidates")
        
        try:
            screening_email_input = {
                "qualified_candidates": qualified_candidates,
                "job_description_urls": job_description_urls,
                "chat_id": chat_id,
                "current_stage": "Resume Screening",
                "next_stage": "Voice Interview"
            }
            
            screening_email_result = yield context.call_activity("send_final_emails_activity", screening_email_input)
            logging.info(f"Screening emails sent: {screening_email_result.get('emails_sent', 0)}")
            
        except Exception as e:
            logging.error(f"Error sending screening emails: {str(e)}")
            screening_email_result = {"error": str(e), "emails_sent": 0}
        
        # PHASE 3: VOICE INTERVIEWS
        logging.info("Starting voice interviews")
        
        voice_interview_tasks = []
        for i, candidate in enumerate(qualified_candidates):
            task_input = {
                "candidate": candidate,
                "job_description_urls": job_description_urls,
                "chat_id": chat_id,
                "voice_interview_threshold": voice_interview_threshold
            }
            logging.info(f"Creating voice interview task {i}")
            task = context.call_activity("trigger_single_voice_interview_activity", task_input)
            voice_interview_tasks.append(task)
        
        # Execute voice interviews
        try:
            voice_results = yield context.task_all(voice_interview_tasks)
            logging.info(f"Voice interviews completed: {len(voice_results) if voice_results else 0}")
            
            if voice_results is None:
                voice_results = []
                
        except Exception as e:
            logging.error(f"Error in voice interviews: {str(e)}")
            voice_results = []
        
        # Filter successful voice interviews
        successful_voice_results = []
        for result in voice_results:
            if result and isinstance(result, dict) and not result.get("error"):
                successful_voice_results.append(result)
        
        logging.info(f"Successful voice interviews: {len(successful_voice_results)}")
        
        # PHASE 4: ANALYSIS
        logging.info("Starting analysis")
        
        try:
            analysis_input = {
                "screening_results": valid_screenings,
                "voice_results": successful_voice_results,
                "job_description_urls": job_description_urls,
                "chat_id": chat_id,
                "voice_interview_threshold": voice_interview_threshold
            }
            
            analysis_result = yield context.call_activity("analyze_results_activity", analysis_input)
            
            if analysis_result.get("error"):
                logging.error(f"Analysis failed: {analysis_result.get('error')}")
                return {"status": "failed", "error": analysis_result.get('error'), "step": "analysis"}
                
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            return {"status": "failed", "error": str(e), "step": "analysis"}
        
        # PHASE 5: FINAL EMAILS
        logging.info("Sending final emails")
        
        try:
            email_input = {
                "final_recommendations": analysis_result.get("final_recommendations", []),
                "job_description_urls": job_description_urls,
                "chat_id": chat_id,
                "notification_email": notification_email,
                "analysis_summary": analysis_result.get("analysis_summary", "")
            }
            
            email_result = yield context.call_activity("send_final_emails_activity", email_input)
            logging.info(f"Final emails sent: {email_result.get('emails_sent', 0)}")
            
        except Exception as e:
            logging.error(f"Error sending final emails: {str(e)}")
            email_result = {"error": str(e), "emails_sent": 0}
        
        # FINAL RESPONSE
        final_response = {
            "status": "completed",
            "chat_id": chat_id,
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(resume_urls),
            "resume_qualified": len(qualified_candidates),
            "voice_qualified": len(successful_voice_results),
            "final_hired": len(analysis_result.get("top_candidates", [])),
            "screening_emails_sent": screening_email_result.get("emails_sent", 0),
            "final_emails_sent": email_result.get("emails_sent", 0),
            "top_candidates": analysis_result.get("top_candidates", []),
            "analysis_summary": analysis_result.get("analysis_summary", ""),
            "screening_results": valid_screenings,
            "voice_results": successful_voice_results,
            "final_recommendations": analysis_result.get("final_recommendations", []),
            "failed_screenings": failed_screenings
        }
        
        logging.info(f"Workflow completed successfully for chat_id: {chat_id}")
        logging.info(f"Final stats: {len(valid_screenings)} screened, {len(qualified_candidates)} qualified, {len(successful_voice_results)} interviewed")
        
        return final_response
        
    except Exception as e:
        error_msg = f"Orchestrator error: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Exception type: {type(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to log the error
        try:
            yield context.call_activity("log_error_activity", {
                "chat_id": input_data.get("chat_id", "unknown") if 'input_data' in locals() else "unknown",
                "error": error_msg,
                "step": "orchestrator"
            })
        except:
            pass
        
        return {"status": "failed", "error": error_msg, "step": "orchestrator"}

# Create the orchestrator
main = df.Orchestrator.create(orchestrator_function)