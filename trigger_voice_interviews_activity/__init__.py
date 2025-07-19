## activity_functions/screen_resumes/__init__.py
import azure.functions as func
import azure.durable_functions as df
import json
import os
import logging
from typing import Dict, List, Any
import tempfile
import requests
from azure.storage.blob import BlobServiceClient
import pandas as pd
from datetime import datetime

# Import your existing classes
from shared.screening import CandidateScreeningAgent
from shared.langchain_prescreening_agent import create_prescreening_agent




def trigger_voice_interviews_activity(voice_data: Dict) -> Dict:
    """
    Activity function to trigger voice interviews for qualified candidates
    
    Args:
        voice_data: {
            "qualified_candidates": [...],
            "job_description": "...",
            "chat_id": "..."
        }
    
    Returns:
        {
            "interview_results": [...],
            "total_completed": int,
            "voice_qualified": [...]
        }
    """
    try:
        logging.info(f"Starting voice interviews for chat_id: {voice_data.get('chat_id')}")
        
        qualified_candidates = voice_data.get("qualified_candidates", [])
        job_description = voice_data.get("job_description", "")
        
        if not qualified_candidates:
            return {"error": "No qualified candidates for voice interview"}
        
        # Initialize voice interview agent
        voice_agent = create_prescreening_agent()
        
        # Prepare input for voice interviewer
        voice_input = {
            "candidates": qualified_candidates,
            "job_description": job_description
        }
        
        # Run voice interviews
        voice_results = voice_agent.run_pre_screening(json.dumps(voice_input))
        
        if "error" in voice_results:
            return {"error": voice_results["error"]}
        
        logging.info(f"Voice interviews completed. {voice_results.get('qualified_count', 0)} candidates passed")
        
        return {
            "interview_results": voice_results.get("all_results", []),
            "total_completed": voice_results.get("completed_screenings", 0),
            "voice_qualified": voice_results.get("qualified_candidates", []),
            "qualified_count": voice_results.get("qualified_count", 0)
        }
        
    except Exception as e:
        logging.error(f"Voice interviews failed: {e}")
        return {"error": f"Voice interviews failed: {str(e)}"}
