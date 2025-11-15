#triggertrigger_single_voice_interview_activity
import logging
import json
import azure.functions as func
from shared.langchain_prescreening_agent import create_prescreening_agent  # Adjust import as needed
def main(input_data: dict) -> str:
    """
    Triggers voice interview for a single candidate

    Args:
        context: Azure Functions context containing input data:
        {
            "candidate": { ... },  # one candidate dict
            "job_description": "...",
            "chat_id": "..."
        }

    Returns:
        str: JSON string with result
    """
    try:
        # Get the activity input
        candidate = input_data["candidate"]
        job_description = input_data["job_description_text"]
        chat_id = input_data.get("chat_id", "unknown")
        logging.info(candidate)
        logging.info(f"Running voice interview for candidate {candidate.get('name')}")

        agent = create_prescreening_agent()

        # Wrap the input as expected by your chain
        input_payload = {
            "candidates": [candidate],  # Single candidate
            "job_description": job_description
        }

        results = agent.run_pre_screening(input_payload)

        if "error" in results:
         return {"error": results["error"]}

        all_results = results.get("all_results", [])
        result = all_results[0] if all_results else {"error": "No result returned"}
        # return json.dumps(result)
        return result

    except Exception as e:
        logging.error(f"trigger_single_voice_interview_activity failed: {e}")
        return {
            "error": f"trigger_single_voice_interview_activity failed: {str(e)}"
        }
