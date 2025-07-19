## http_triggers/get_workflow_status/__init__.py
import azure.functions as func
import azure.durable_functions as df
import json
import logging

async def get_workflow_status(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    """
    HTTP trigger to get workflow status
    """
    try:
        client = df.DurableOrchestrationClient(starter)
        
        # Get instance ID from route
        instance_id = req.route_params.get('instance_id')
        
        if not instance_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing instance_id parameter"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Get orchestration status
        status = await client.get_status(instance_id)
        
        if status is None:
            return func.HttpResponse(
                json.dumps({"error": "Instance not found"}),
                status_code=404,
                mimetype="application/json"
            )
        
        # Return status information
        return func.HttpResponse(
            json.dumps({
                "instanceId": status.instance_id,
                "runtimeStatus": status.runtime_status,
                "input": status.input_,
                "output": status.output,
                "createdTime": status.created_time.isoformat() if status.created_time else None,
                "lastUpdatedTime": status.last_updated_time.isoformat() if status.last_updated_time else None
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Failed to get workflow status: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Failed to get workflow status: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )
