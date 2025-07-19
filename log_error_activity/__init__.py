import azure.functions as func
import logging
import json

def main(context: func.Context) -> str:
    """
    Simple activity to log errors
    """
    try:
        # Get the activity input
        data = context.get_input()
        msg = data.get("msg", "Unknown error")
        logging.error(f"Orchestration error: {msg}")
        result = {"status": "logged", "message": msg}
        return json.dumps(result)
    except Exception as e:
        logging.error(f"log_error_activity failed: {e}")
        result = {"status": "failed", "error": str(e)}
        return json.dumps(result)