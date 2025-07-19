#analyze_results
#init.py
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



def analyze_results_activity(analysis_data: Dict) -> Dict:
    """
    Activity function to analyze combined results from resume screening and voice interviews
    
    Args:
        analysis_data: {
            "screening_results": [...],
            "voice_results": [...],
            "job_description": "...",
            "chat_id": "..."
        }
    
    Returns:
        {
            "final_recommendations": [...],
            "analysis_summary": "...",
            "top_candidates": [...]
        }
    """
    try:
        logging.info(f"Analyzing results for chat_id: {analysis_data.get('chat_id')}")
        
        screening_results = analysis_data.get("screening_results", [])
        voice_results = analysis_data.get("voice_results", [])
        
        # Combine and analyze results
        final_candidates = []
        
        # Create a mapping of candidates from voice results
        voice_map = {}
        for candidate in voice_results:
            if candidate.get("status") == "completed":
                voice_map[candidate.get("candidate_id")] = candidate
        
        # Combine screening and voice results
        for i, screening in enumerate(screening_results):
            if not isinstance(screening, dict) or "error" in screening:
                continue
                
            candidate_id = i + 1
            voice_result = voice_map.get(candidate_id, {})
            
            combined_score = 0
            if screening.get("overall_fit_score"):
                resume_score = screening["overall_fit_score"]
                voice_score = voice_result.get("overall_score", 0)
                
                # Weighted combination: 40% resume, 60% voice interview
                combined_score = (resume_score * 0.4) + (voice_score * 0.6)
            
            final_candidate = {
                "candidate_name": screening.get("candidate_name", "Unknown"),
                "candidate_email": screening.get("candidate_email", ""),
                "resume_score": screening.get("overall_fit_score", 0),
                "voice_score": voice_result.get("overall_score", 0),
                "combined_score": round(combined_score, 2),
                "resume_recommendation": screening.get("recommendation", ""),
                "voice_qualified": voice_result.get("qualified", False),
                "strengths": screening.get("strengths", []),
                "weaknesses": screening.get("weaknesses", []),
                "final_status": "HIRE" if combined_score >= 6.0 and voice_result.get("qualified", False) else "REJECT"
            }
            
            final_candidates.append(final_candidate)
        
        # Sort by combined score
        final_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Generate analysis summary
        hire_count = len([c for c in final_candidates if c["final_status"] == "HIRE"])
        analysis_summary = f"""
        Analysis Summary:
        - Total candidates processed: {len(final_candidates)}
        - Candidates recommended for hire: {hire_count}
        - Average combined score: {sum(c['combined_score'] for c in final_candidates) / len(final_candidates):.2f}
        - Top candidate: {final_candidates[0]['candidate_name'] if final_candidates else 'None'} (Score: {final_candidates[0]['combined_score'] if final_candidates else 0})
        """
        
        logging.info(f"Analysis completed. {hire_count} candidates recommended for hire")
        
        return {
            "final_recommendations": final_candidates,
            "analysis_summary": analysis_summary.strip(),
            "top_candidates": final_candidates[:5],  # Top 5 candidates
            "hire_count": hire_count
        }
        
    except Exception as e:
        logging.error(f"Results analysis failed: {e}")
        return {"error": f"Results analysis failed: {str(e)}"}