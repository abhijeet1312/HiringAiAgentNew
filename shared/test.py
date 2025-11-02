# import json
# import logging
# import azure.durable_functions as df
# from datetime import datetime
# from typing import List, Dict, Any
# from supabase_client import supabase
# from shared.azurestorage import (
#     azure_config,
#     extract_text_from_azure,
#     parse_azure_url_to_container_blob_path
# )


# def orchestrator_function(context: df.DurableOrchestrationContext):
#     """
#     Fixed orchestrator following proper Durable Functions patterns - Phase 1 & 2 only
#     """
#     logging.info("🎯 Starting hiring workflow orchestration")
    
#     # Get input data
#     input_data = context.get_input()
#     chat_id = input_data.get("chat_id")
#     resume_urls = input_data.get("resume_urls", [])
#     job_description_urls = input_data.get("job_description_urls", [])
#     notification_email = input_data.get("notification_email", "")
#     voice_interview_threshold = input_data.get("voice_interview_threshold", 3.0)
    
#     logging.info(f"Processing hiring workflow for chat_id: {chat_id}")
#     logging.info(f"Total resumes to process: {len(resume_urls)}")
#     logging.info(f"Total job descriptions: {len(job_description_urls)}")
#     job_desc_url = job_description_urls[0]  # Use first job description URL
#     job_desc_blob_info = parse_azure_url_to_container_blob_path(job_desc_url)
#     job_description_text = extract_text_from_azure(job_desc_blob_info['blob_path'])
    
#     # PHASE 1: INPUT VALIDATION
#     if not resume_urls or not job_description_urls:
#         error_msg = "Invalid input: resume_urls and job_description_urls are required"
#         logging.error(error_msg)
        
#         # Log error activity (yield required for activity calls)
#         yield context.call_activity("log_error_activity", {
#             "chat_id": chat_id,
#             "error": error_msg,
#             "step": "input_validation"
#         })
#         return {"status": "failed", "error": error_msg, "step": "input_validation"}
    
#     # PHASE 2: RESUME SCREENING - Fan-Out/Fan-In Pattern
#     logging.info("🚀 Starting resume screening phase")
    
#     # Collect screening tasks (no yield here - just collecting)
#     screening_tasks = []
#     for i, resume_url in enumerate(resume_urls):
#         task_input = {
#             "resume_url": resume_url,
#             "job_description_urls": job_description_urls,
#             "chat_id": chat_id,
#             "resume_index": i
#         }
#         logging.info(f"Creating screening task {i} for resume: {resume_url}")
#         # Append task without yielding
#         screening_tasks.append(context.call_activity("resume_screening_activity", task_input))
    
#     # Fan-In: Execute all screening tasks in parallel (single yield)
#     logging.info(f"Executing {len(screening_tasks)} screening tasks in parallel")
#     screening_results = yield context.task_all(screening_tasks)
#     logging.info("Resume screening phase completed, ====================", screening_results)
#     # logging.info(f"Screening tasks completed: {screening_results}results received")
#     # Extract only resume_url and overall_fit_score
#     screening_result_resume = [
#         {
#             "resume_url": item["resume_url"],
#             "score": item["overall_fit_score"]
#         }
#         for item in screening_results
#     ]
    
#     response = (
#     supabase.table("screening")
#     .update({"screening_result_resumes": screening_result_resume})
#     .eq("chat_id", chat_id)
#     .execute()
# )
#     logging.info(f"Screening results updated in Supabase: {response}")
    
#     # return screening_results
#     logging.info(" All screening tasks completed")
#     selected=[]
#     not_selected=[]
#     score_threshold=4
#     idx=0
#     for candidate in screening_results:
#         candidate_info = {
#             "name": candidate["candidate_name"],
#             "email": candidate["candidate_email"],
#             "phone": candidate["candidate_phone"],
#             "screening_score": candidate["overall_fit_score"],
#             "status": candidate["status"],
#             "resume_url": candidate["resume_url"],
#             "job_description_url": job_desc_url,
#             "id":idx+1
#         }
        
#         if candidate["overall_fit_score"] > score_threshold:
#             selected.append(candidate_info)
#         else:
#             not_selected.append(candidate_info)
        
#     #phase 3 sending emails
#     screening_email_input = {
#                 "qualified_candidates": selected,
#                 "job_description": job_description_text,
#                 "chat_id": chat_id,
#                 "current_stage": "Resume Screening",
#                 "next_stage": "Voice Interview"
#             }
            
#     screening_email_result = yield context.call_activity("send_final_emails_activity", screening_email_input)
#     # logging.info(f"Screening emails sent: {screening_email_result.get('emails_sent', 0)}")
           
        
    
#     #phase 4 starting voice interview
    
#     # PHASE 4: VOICE INTERVIEWS
#     logging.info("🎯 Starting voice interviews")

#     voice_interview_tasks = []
#     for i, candidate in enumerate(selected):  # 'selected' from resume screening phase
#         task_input = {
#             "candidate": candidate,
#             "job_description_text": job_description_text,  # Assuming this is a string of JD text; rename if it's URL
#             "chat_id": chat_id,
#             "voice_interview_threshold": voice_interview_threshold  # Define this above
#         }
#         logging.info(f"🔊 Creating voice interview task {i+1} for {candidate.get('name')}")
#         task = context.call_activity("trigger_single_voice_interview_activity", task_input)
#         voice_interview_tasks.append(task)

#     # Execute all voice interview tasks in parallel
#     try:
#         voice_results = yield context.task_all(voice_interview_tasks)
#         logging.info(f"✅ Voice interviews completed for {len(voice_results)} candidates")
        
#         if voice_results is None:
#             voice_results = []

#     except Exception as e:
#         logging.error(f"❌ Error in voice interviews: {str(e)}")
#         voice_results = []
#     voice_result_score = [
#         {
#             "resume_url": item["resume_url"],
#             "score": item["overall_score"]
#         }
#         for item in voice_results
#     ]
#     response = (
#     supabase.table("screening")
#     .update({"prescreening_result_resumes": voice_result_score})
#     .eq("chat_id", chat_id)
#     .execute()
# )
#     logging.info(f"PreScreening results updated in Supabase: {response}")
#     status=(
#         supabase.table("screening")
#         .update({"status": "completed"})
#         .eq("chat_id", chat_id)
#         .execute()
#     )
    
#     # return voice_results
#     # Filter only successful voice interview results
#     successful_voice_results = []
#     for result in voice_results:
#         if result and isinstance(result, dict) and not result.get("error"):
#             successful_voice_results.append(result)
   
#     logging.info(f"🏆 Successful voice interviews: {len(successful_voice_results)}")

    
   
#     logging.info("🎯 Orchestrator completed successfully")
#     if screening_results[0]["overall_fit_score"]>2:
#         logging.info("jai mata di")
#     # Return screening results
#     final_result = {
#         "status": "success",
#         "chat_id": chat_id,
#         "summary": {
#             "total_resumes_processed": len(resume_urls),
#             "processing_timestamp": datetime.now().isoformat()
#         },
#         "screening_results": screening_results,
#         "voice_results":voice_results
#     }
    
        
#     logging.info(final_result)
#     return final_result

# # Create the orchestrator
# main = df.Orchestrator.create(orchestrator_function)