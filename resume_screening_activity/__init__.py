import logging
import tempfile
import requests
import os
import json
import azure.functions as func
from shared.screening import CandidateScreeningAgent  # update path if needed
import hashlib
from urllib.parse import urlparse, unquote
import sys
from pathlib import Path

# Add shared folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from shared.azurestorage import (
    azure_config,
    extract_text_from_azure,
    parse_azure_url_to_container_blob_path
)

def extract_phone_number(text: str) -> str:
    """Extract phone number from text using regex pattern"""
    import re
    phone_pattern = r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    phone_matches = re.findall(phone_pattern, text)
    return phone_matches[0] if phone_matches else None

def get_score_category(score: int) -> str:
    """Categorize score into performance bands"""
    if score >= 8:
        return "Excellent"
    elif score >= 6:
        return "Good"
    elif score >= 4:
        return "Average"
    else:
        return "Below Average"

def main(context):
    """
    Screens a single resume using CandidateScreeningAgent with batch processing

    Args:
        context: Input data (dict or JSON string):
        {
            "resume_url": "https://aurjobsaiagentsstorage.blob.core.windows.net/aurjobs123/resumes/...",
            "job_description_urls": [
                "https://aurjobsaiagentsstorage.blob.core.windows.net/aurjobs123/job_descriptions/..."
            ],
            "chat_id": "unique_chat_123",
            "resume_index": 0
        }

    Returns:
        str: JSON string with result
    """
    logging.info("Starting screen_single_resume_activity")
    
    try:
        # Handle both dict and string inputs
        if isinstance(context, dict):
            data = context
        else:
            data = json.loads(context)
        
        resume_url = data.get("resume_url")
        job_description_urls = data.get("job_description_urls", [])
        chat_id = data.get("chat_id")
        resume_index = data.get("resume_index", 0)
        
        logging.info(f"Screening resume {resume_index} for chat_id: {chat_id}")
        
        if not resume_url or not job_description_urls:
            result = {
                "error": "Missing required fields: resume_url or job_description_urls",
                "resume_url": resume_url,
                "resume_index": resume_index,
                "chat_id": chat_id
            }
            return result
        
        # Parse and extract job description from first URL
        try:
            job_desc_url = job_description_urls[0]  # Use first job description URL
            job_desc_blob_info = parse_azure_url_to_container_blob_path(job_desc_url)
            job_description_text = extract_text_from_azure(job_desc_blob_info['blob_path'])
            
            if not job_description_text or len(job_description_text.strip()) < 10:
                result = {
                    "error": "Job description text extraction failed or text too short",
                    "job_description_url": job_desc_url,
                    "resume_url": resume_url,
                    "resume_index": resume_index,
                    "chat_id": chat_id
                }
                return result
        except Exception as e:
            logging.error(f"Error extracting job description from {job_desc_url}: {str(e)}")
            result = {
                "error": f"Job description extraction failed: {str(e)}",
                "job_description_url": job_desc_url,
                "resume_url": resume_url,
                "resume_index": resume_index,
                "chat_id": chat_id
            }
            return result
        
        # Parse and extract resume text from Azure blob
        try:
            resume_blob_info = parse_azure_url_to_container_blob_path(resume_url)
            resume_text = extract_text_from_azure(resume_blob_info['blob_path'])
            
            if not resume_text or len(resume_text.strip()) < 10:
                result = {
                    "error": "Resume text extraction failed or text too short",
                    "resume_url": resume_url,
                    "resume_index": resume_index,
                    "chat_id": chat_id
                }
                return result
        except Exception as e:
            logging.error(f"Error extracting text from {resume_url}: {str(e)}")
            result = {
                "error": f"Resume text extraction failed: {str(e)}",
                "resume_url": resume_url,
                "resume_index": resume_index,
                "chat_id": chat_id
            }
            return result
        
        # Create screening agent with extracted job description
        try:
            logging.info(f"job_description_text: {job_description_text}...")  # Log first 100 chars for debugging
            agent = CandidateScreeningAgent(job_description=job_description_text)
        except Exception as e:
            logging.error(f"Error creating screening agent: {str(e)}")
            result = {
                "error": f"Agent creation failed: {str(e)}",
                "resume_url": resume_url,
                "resume_index": resume_index,
                "chat_id": chat_id
            }
            return result
        
        # Set voice interview threshold (consistent with batch processing)
        voice_interview_threshold = 3.0
        
        # Save to temporary file and use batch processing
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(resume_text)
                temp_file_path = temp_file.name
            
            # Screen the candidate using batch logic (but with one file)
            assessments = agent.batch_screen_candidates([temp_file_path])
            assessment = assessments[0] if assessments else None
            
            if not assessment:
                result = {
                    "error": "Screening assessment failed - no assessment returned",
                    "resume_url": resume_url,
                    "resume_index": resume_index,
                    "chat_id": chat_id
                }
                return result
            
            # Convert assessment to dict if needed
            if hasattr(assessment, 'dict'):
                assessment = assessment.dict()
            elif not isinstance(assessment, dict):
                assessment = {
                    "error": f"Unexpected assessment type: {type(assessment)}",
                    "raw_assessment": str(assessment),
                    "resume_url": resume_url,
                    "resume_index": resume_index,
                    "chat_id": chat_id
                }
                return assessment
            
            # Extract filename from URL
            filename = resume_url.split('/')[-1]
            
            # Extract phone number from resume text
            phone_number = extract_phone_number(resume_text)
            
            # Determine status based on threshold (similar to KMeans logic in batch)
            overall_fit_score = assessment.get("overall_fit_score", 0)
            status = "PASS" if overall_fit_score >= voice_interview_threshold else "FAIL"
            qualifies_for_voice = overall_fit_score >= voice_interview_threshold
            
            # IMPROVED: Standardize recommendation based on score and assessment
            original_recommendation = assessment.get("recommendation", "Not Recommended")
            
            #
            result = {
                "resume_url": resume_url,
                "resume_index": resume_index,
                "filename": filename,
                "chat_id": chat_id,
                "job_description_url": job_desc_url,
                "candidate_name": assessment.get("candidate_name", "Unknown"),
                "candidate_email": assessment.get("candidate_email", ""),
                "candidate_phone": phone_number or assessment.get("candidate_phone", ""),
                "overall_fit_score":overall_fit_score,
                
                "status": status,
                
                
               
            }
            
            # # Use logging instead of print for better visibility
            # logging.info(f"Successfully screened resume {resume_index}: {result.get('candidate_name')} - "
            #             f"Recommendation: {result.get('recommendation')} - "
            #             f"Status: {result.get('status')} - "
            #             f"Score: {result.get('overall_fit_score')} - "
            #             f"Qualifies for voice: {result.get('qualifies_for_voice_interview')}")
            # logging.info("going out of screen_single_resume_activity")
            logging.info(f"About to return result: {json.dumps(result, indent=2)}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error during screening process: {str(e)}")
            result = {
                "error": f"Screening process failed: {str(e)}",
                "resume_url": resume_url,
                "resume_index": resume_index,
                "chat_id": chat_id
            }
            return result
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logging.warning(f"Error deleting temp file: {str(e)}")
    
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in screen_single_resume_activity: {e}")
        result = {
            "error": f"Invalid JSON input: {str(e)}",
            "chat_id": "unknown"
        }
        return result
    
    except Exception as e:
        logging.error(f"Unexpected error in screen_single_resume_activity: {str(e)}")
        import traceback
        traceback.print_exc()
        result = {
            "error": f"Unexpected error: {str(e)}",
            "resume_url": data.get("resume_url", "unknown") if 'data' in locals() else "unknown",
            "resume_index": data.get("resume_index", 0) if 'data' in locals() else 0,
            "chat_id": data.get("chat_id", "unknown") if 'data' in locals() else "unknown"
        }
        return result