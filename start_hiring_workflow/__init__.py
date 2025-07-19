#start_hiring_workflow/__init__.py
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
import uuid

# Import your existing classes
from shared.screening import CandidateScreeningAgent
from shared.langchain_prescreening_agent import create_prescreening_agent

print("Starting hiring workflow HTTP trigger...")



async def start_hiring_workflow(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    """
    HTTP trigger to start the hiring workflow
    
    Expected input:
    {
        "chat_id": "unique_chat_identifier",  # Optional - will be auto-generated if not provided
        "resumes": ["url1", "url2", ...],
        "job_descriptions": ["url1", "url2", ...],
        "notification_email": "hr@company.com",  # Optional
        "voice_interview_threshold": 3.0  # Optional
    }
    """
    logging.info('Starting hiring workflow HTTP trigger')
    
    try:
        client = df.DurableOrchestrationClient(starter)
        
        # Parse request body
        req_body = req.get_json()
        print(f"Received request body: {req_body}")
        
        # Validate request body exists
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract fields with defaults
        chat_id = req_body.get("chat_id")
        resumes = req_body.get("resumes", [])
        job_descriptions = req_body.get("job_descriptions", [])
        notification_email = req_body.get("notification_email", "")
        voice_interview_threshold = req_body.get("voice_interview_threshold", 3.0)
        
        # Generate chat_id if not provided
        if not chat_id:
            chat_id = str(uuid.uuid4())
            logging.info(f"Generated chat_id: {chat_id}")
        
        # Validate required fields
        if not resumes or not isinstance(resumes, list):
            return func.HttpResponse(
                json.dumps({"error": "resumes must be a non-empty list of URLs"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not job_descriptions or not isinstance(job_descriptions, list):
            return func.HttpResponse(
                json.dumps({"error": "job_descriptions must be a non-empty list of URLs"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Create the input payload for orchestrator
        orchestrator_input = {
            "chat_id": chat_id,
            "job_description_urls": job_descriptions,  # Pass URLs directly
            "resume_urls": resumes,  # Pass URLs directly
            "notification_email": notification_email,
            "voice_interview_threshold": voice_interview_threshold
        }
        
        # Start orchestration with unique instance ID
        instance_id = f"hiring_{chat_id}"
        
        # Start the orchestration
        await client.start_new(
            "orchestrator",
            instance_id=instance_id,
            client_input=orchestrator_input
        )
        
        # Return response with management URLs and detailed information
        management_urls = client.get_client_response_links(req, instance_id)
        
        response_data = {
            "message": f"Hiring workflow started successfully",
            "instance_id": instance_id,
            "chat_id": chat_id,
            "status": "started",
            "total_resumes": len(resumes),
            "total_job_descriptions": len(job_descriptions),
            "management_urls": management_urls
        }
        
        logging.info(f"Hiring workflow started successfully: {instance_id}")
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=202,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Failed to start workflow: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Failed to start workflow: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )