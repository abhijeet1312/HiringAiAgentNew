import logging
import json
import uuid
from azure.functions import HttpRequest, HttpResponse
from azure.durable_functions import DurableOrchestrationClient

async def main(req: HttpRequest, starter: str) -> HttpResponse:
    """
    Generic HTTP trigger to start any orchestration with enhanced features
    
    Expected input format:
    {
        "orchestrator_name": "orchestrator",  # Required - name of orchestrator to start
        "chat_id": "unique_chat_identifier",  # Optional - will be auto-generated if not provided
        "resume_urls": ["url1", "url2", ...], # Your data fields go directly here
        "job_description_urls": ["url1", ...],
        "notification_email": "hr@company.com",
        "voice_interview_threshold": 3.0,
        "validation_rules": {                 # Optional - validation rules for your data
            "required_fields": ["resume_urls", "job_description_urls"],
            "field_types": {
                "resume_urls": "list",
                "job_description_urls": "list",
                "voice_interview_threshold": "float"
            }
        }
    }
    """
    logging.info('Starting generic orchestration HTTP trigger')
    
    try:
        client = DurableOrchestrationClient(starter)
        
        # Parse request body
        try:
            req_body = req.get_json()
        except ValueError:
            return HttpResponse(
                json.dumps({"error": "Invalid JSON input"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate request body exists
        if not req_body:
            return HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract main fields with defaults
        orchestrator_name = req_body.get("orchestrator_name", "orchestrator")
        chat_id = req_body.get("chat_id")
        validation_rules = req_body.get("validation_rules", {})
        
        # Pass entire request body to orchestrator
        input_data = req_body
        
        # Generate chat_id if not provided
        if not chat_id:
            chat_id = str(uuid.uuid4())
            logging.info(f"Generated chat_id: {chat_id}")
        
        # Perform validation if rules are provided
        if validation_rules:
            validation_error = validate_input_data(req_body, validation_rules)
            if validation_error:
                return HttpResponse(
                    json.dumps({"error": validation_error}),
                    status_code=400,
                    mimetype="application/json"
                )
        
        # Add chat_id to request body if not already present
        if not chat_id:
            req_body["chat_id"] = chat_id
        
        # Create unique instance ID
        instance_id = f"{orchestrator_name}_{chat_id}"
        
        # Start the orchestration
        started_instance_id = await client.start_new(
            orchestrator_name,
            instance_id=instance_id,
            client_input=input_data
        )
        
        # Get management URLs
        management_urls = client.get_client_response_links(req, instance_id)
        
        # Prepare detailed response
        response_data = {
            "message": f"Orchestration '{orchestrator_name}' started successfully",
            "instance_id": instance_id,
            "started_instance_id": started_instance_id,
            "chat_id": chat_id,
            "orchestrator_name": orchestrator_name,
            "status": "started",
            "input_data_keys": list(input_data.keys()) if input_data else [],
            "management_urls": management_urls
        }
        
        logging.info(f"Orchestration started successfully: {instance_id}")
        
        return HttpResponse(
            json.dumps(response_data),
            status_code=202,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Failed to start orchestration: {e}")
        return HttpResponse(
            json.dumps({"error": f"Failed to start orchestration: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


def validate_input_data(input_data: dict, validation_rules: dict) -> str:
    """
    Validate input data based on provided validation rules
    
    Args:
        input_data: The data to validate
        validation_rules: Dictionary containing validation rules
        
    Returns:
        Error message if validation fails, None if validation passes
    """
    # Check required fields
    required_fields = validation_rules.get("required_fields", [])
    for field in required_fields:
        if field not in input_data:
            return f"Missing required field: {field}"
        
        # Check if field is empty for lists
        if isinstance(input_data[field], list) and len(input_data[field]) == 0:
            return f"Required field '{field}' cannot be empty"
    
    # Check field types
    field_types = validation_rules.get("field_types", {})
    for field, expected_type in field_types.items():
        if field in input_data:
            value = input_data[field]
            
            if expected_type == "list" and not isinstance(value, list):
                return f"Field '{field}' must be a list"
            elif expected_type == "string" and not isinstance(value, str):
                return f"Field '{field}' must be a string"
            elif expected_type == "int" and not isinstance(value, int):
                return f"Field '{field}' must be an integer"
            elif expected_type == "float" and not isinstance(value, (int, float)):
                return f"Field '{field}' must be a number"
            elif expected_type == "bool" and not isinstance(value, bool):
                return f"Field '{field}' must be a boolean"
    
    # Custom validation for specific patterns
    custom_validations = validation_rules.get("custom_validations", {})
    for field, validation_type in custom_validations.items():
        if field in input_data:
            value = input_data[field]
            
            if validation_type == "non_empty_list" and (not isinstance(value, list) or len(value) == 0):
                return f"Field '{field}' must be a non-empty list"
            elif validation_type == "email" and isinstance(value, str) and value and "@" not in value:
                return f"Field '{field}' must be a valid email address"
            elif validation_type == "positive_number" and isinstance(value, (int, float)) and value <= 0:
                return f"Field '{field}' must be a positive number"
    
    return None  # No validation errors


# No validation errors
# import logging
# import json
# from azure.functions import HttpRequest, HttpResponse
# from azure.durable_functions import DurableOrchestrationClient


# async def main(req: HttpRequest, starter: str) -> HttpResponse:
#     client = DurableOrchestrationClient(starter)

#     try:
#         input_data = req.get_json()
#     except ValueError:
#         return HttpResponse("Invalid JSON input", status_code=400)

#     # Start the orchestration by name and pass input_data
#     instance_id = await client.start_new("orchestrator", None, input_data)

#     logging.info(f"✅ Started orchestration with ID = '{instance_id}'.")

#     return client.create_check_status_response(req, instance_id)



# # # This function an HTTP starter function for Durable Functions.
# # # Before running this sample, please:
# # # - create a Durable orchestration function
# # # - create a Durable activity function (default name is "Hello")
# # # - add azure-functions-durable to requirements.txt
# # # - run pip install -r requirements.txt
 
# # import logging

# # from azure.functions import HttpRequest, HttpResponse
# # from azure.durable_functions import DurableOrchestrationClient


# # async def main(req: HttpRequest, starter: str) -> HttpResponse:
# #     client = DurableOrchestrationClient(starter)
# #     instance_id = await client.start_new(req.route_params["functionName"], None, None)

# #     logging.info(f"Started orchestration with ID = '{instance_id}'.")

# #     return client.create_check_status_response(req, instance_id)