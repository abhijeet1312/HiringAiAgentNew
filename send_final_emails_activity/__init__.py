# send_final_emails/__init__.py
import azure.functions as func
import azure.durable_functions as df
import json
import os
import logging
import smtplib
from typing import Dict, List, Any, Optional
import tempfile
import requests
from azure.storage.blob import BlobServiceClient
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Import your existing classes
from shared.screening import CandidateScreeningAgent
from shared.langchain_prescreening_agent import create_prescreening_agent


def send_stage_emails(email_data: Dict) -> Dict:
    """
    Universal email sending function that handles all hiring stages
    
    Args:
        email_data: {
            "candidates": [...],  # List of candidate objects
            "job_description_urls": [...],
            "chat_id": "...",
            "current_stage": "Resume Screening",  # Stage candidates just completed
            "next_stage": "Voice Interview",      # Stage candidates are moving to
            "email_type": "qualification|final"   # Type of email flow
        }
    
    Returns:
        {
            "emails_sent": int,
            "success": bool,
            "details": "...",
            "recipients": [...],
            "hired_count": int (only for final emails),
            "rejected_count": int (only for final emails)
        }
    """
    try:
        logging.info(f"Processing {email_data.get('email_type', 'qualification')} emails for chat_id: {email_data.get('chat_id')}")
        
        # Extract and validate input data
        candidates = email_data.get("candidates", [])
        if not candidates:
            return {"error": "No candidates provided"}
        
        job_description = _extract_job_description(email_data.get("job_description_urls", []))
        current_stage = email_data.get("current_stage", "Current Stage")
        next_stage = email_data.get("next_stage", "Next Stage")
        email_type = email_data.get("email_type", "qualification")
        
        # Handle different email types
        if email_type == "final":
            return _send_final_decision_emails(candidates, job_description, current_stage, next_stage)
        else:
            return _send_qualification_emails(candidates, job_description, current_stage, next_stage)
        
    except Exception as e:
        logging.error(f"Email sending failed: {e}")
        return {"error": f"Email sending failed: {str(e)}"}


def _extract_job_description(job_description_urls: List[str]) -> str:
    """Extract job description from URLs or return placeholder"""
    if job_description_urls:
        return f"Job posting reference: {job_description_urls[0]}"
    return "Position details as discussed"


def _extract_candidate_emails(candidates: List[Dict]) -> List[str]:
    """Extract valid email addresses from candidate objects"""
    emails = []
    for candidate in candidates:
        email = candidate.get("candidate_email") or candidate.get("email")
        if email and email.strip():
            emails.append(email.strip())
    return emails


def _send_qualification_emails(candidates: List[Dict], job_description: str, current_stage: str, next_stage: str) -> Dict:
    """Send emails to candidates who qualified for next stage"""
    candidate_emails = _extract_candidate_emails(candidates)
    
    if not candidate_emails:
        return {"error": "No candidate emails found"}
    
    # Send bulk email
    success = _send_bulk_email(
        candidate_emails,
        job_description,
        current_stage,
        next_stage
    )
    
    if success:
        return {
            "emails_sent": len(candidate_emails),
            "success": True,
            "details": f"Sent {current_stage} qualification emails to {len(candidate_emails)} candidates",
            "recipients": candidate_emails
        }
    else:
        return {"error": f"Failed to send {current_stage} qualification emails"}


def _send_final_decision_emails(candidates: List[Dict], job_description: str, current_stage: str, next_stage: str) -> Dict:
    """Send final decision emails (hire/reject)"""
    # Separate hired and rejected candidates
    hired_candidates = [c for c in candidates if c.get("final_status") == "HIRE"]
    rejected_candidates = [c for c in candidates if c.get("final_status") == "REJECT"]
    
    emails_sent = 0
    
    # Send emails to hired candidates
    if hired_candidates:
        hired_emails = _extract_candidate_emails(hired_candidates)
        if hired_emails:
            try:
                success = _send_bulk_email(
                    hired_emails,
                    job_description,
                    current_stage,
                    "Job Offer - Next Steps"
                )
                if success:
                    emails_sent += len(hired_emails)
                    logging.info(f"Sent congratulations emails to {len(hired_emails)} candidates")
            except Exception as e:
                logging.error(f"Failed to send congratulations emails: {e}")
    
    # Send emails to rejected candidates
    if rejected_candidates:
        rejected_emails = _extract_candidate_emails(rejected_candidates)
        if rejected_emails:
            try:
                success = _send_bulk_email(
                    rejected_emails,
                    job_description,
                    current_stage,
                    "Thank You for Your Interest"
                )
                if success:
                    emails_sent += len(rejected_emails)
                    logging.info(f"Sent rejection emails to {len(rejected_emails)} candidates")
            except Exception as e:
                logging.error(f"Failed to send rejection emails: {e}")
    
    return {
        "emails_sent": emails_sent,
        "success": True,
        "details": f"Sent {emails_sent} final decision emails",
        "hired_count": len(hired_candidates),
        "rejected_count": len(rejected_candidates)
    }


def _send_bulk_email(recipients: List[str], job_description: str, current_stage: str, next_stage: str) -> bool:
    """
    Send bulk emails to multiple recipients using Gmail SMTP with AI-generated content.
    
    Args:
        recipients: List of recipient email addresses
        job_description: Job description text
        current_stage: Current stage in hiring process
        next_stage: Next stage in hiring process
    
    Returns:
        bool: True if successful, False otherwise
    """
    sender_email = "shivamsrivastava2189@gmail.com"
    app_password = os.getenv("Google_app_password")
    API_KEY = os.getenv("GOOGLE_API_KEY")

    # Initialize the AI model
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        api_key=API_KEY,
        temperature=0.7
    )

    # Create the prompt for email content generation
    prompt = f"""
    You are an HR assistant. Craft a professional, polite, and encouraging email to a job applicant.
    Inform them that they have successfully qualified the current stage of the hiring process.

    Job Description:
    {job_description}

    Current Stage:
    {current_stage}

    Next Stage:
    {next_stage}

    Ensure the email includes:
    - A congratulatory tone
    - Reference to the job role and selection stage
    - What the next stage involves
    - Next steps or instructions
    - Encouragement to prepare
    - Professional email format with subject line

    Keep it reusable and concise. Format as a complete email with Subject line.
    """

    try:
        # Generate email content using AI
        response = model.invoke(prompt)
        message = response.content if hasattr(response, "content") else response
        
        # Send email via SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(sender_email, app_password)
            s.sendmail(sender_email, recipients, message)
            logging.info(f"Email sent successfully to: {recipients}")
            return True
            
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return False


# Azure Function entry point - handles all email types
def send_final_emails_activity(email_data: Dict) -> Dict:
    """
    Main Azure Function entry point that handles all email types
    Auto-detects email type based on input data structure
    
    Args:
        email_data: Can contain any of these patterns:
        - qualified_candidates (screening emails)
        - voice_qualified_candidates (voice interview emails)  
        - final_recommendations (final decision emails)
        - OR direct candidates list with email_type specified
    """
    try:
        # Auto-detect email type and map data structure
        if "final_recommendations" in email_data:
            # Final decision emails
            email_data.update({
                "candidates": email_data.get("final_recommendations", []),
                "current_stage": email_data.get("current_stage", "Final Interview Results"),
                "next_stage": email_data.get("next_stage", "Next Steps"),
                "email_type": "final"
            })
        elif "voice_qualified_candidates" in email_data:
            # Voice interview qualification emails
            email_data.update({
                "candidates": email_data.get("voice_qualified_candidates", []),
                "current_stage": email_data.get("current_stage", "Voice Interview"),
                "next_stage": email_data.get("next_stage", "Final Review"),
                "email_type": "qualification"
            })
        elif "qualified_candidates" in email_data:
            # Screening qualification emails
            email_data.update({
                "candidates": email_data.get("qualified_candidates", []),
                "current_stage": email_data.get("current_stage", "Resume Screening"),
                "next_stage": email_data.get("next_stage", "Voice Interview"),
                "email_type": "qualification"
            })
        # If email_type is already specified, use as-is
        
        return send_stage_emails(email_data)
        
    except Exception as e:
        logging.error(f"Azure Function entry point error: {e}")
        return {"error": f"Azure Function entry point error: {str(e)}"}


# Convenience wrapper functions for direct calls (optional)
def send_screening_emails_activity(email_data: Dict) -> Dict:
    """Send emails to candidates who passed screening"""
    email_data.update({
        "candidates": email_data.get("qualified_candidates", []),
        "current_stage": email_data.get("current_stage", "Resume Screening"),
        "next_stage": email_data.get("next_stage", "Voice Interview"),
        "email_type": "qualification"
    })
    return send_stage_emails(email_data)


def send_voice_interview_emails_activity(email_data: Dict) -> Dict:
    """Send emails to candidates who passed voice interviews"""
    email_data.update({
        "candidates": email_data.get("voice_qualified_candidates", []),
        "current_stage": email_data.get("current_stage", "Voice Interview"),
        "next_stage": email_data.get("next_stage", "Final Review"),
        "email_type": "qualification"
    })
    return send_stage_emails(email_data)


# Azure Functions entry points
def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main entry point for HTTP-triggered function"""
    try:
        # This would handle HTTP requests if needed
        return func.HttpResponse("Email service is running", status_code=200)
    except Exception as e:
        logging.error(f"HTTP function error: {e}")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)