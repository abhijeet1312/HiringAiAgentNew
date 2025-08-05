
import smtplib
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def send_final_emails_activity(email_data:dict):
    screening_email_input=email_data
    import smtplib
    import os
    from langchain_openai import AzureChatOpenAI
    recipients = screening_email_input["qualified_candidates"]
    job_description = screening_email_input["job_description"]
    current_stage = screening_email_input["current_stage"]
    next_stage = screening_email_input["next_stage"]
    
    recipients_email = [candidate["email"] for candidate in recipients]
        

    # Email configuration
    sender_email = os.getenv("ZOHOMAIL_EMAIL")
    app_password = os.getenv("ZOHOMAIL_PASSWORD")
    
    # Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")  # e.g., "https://your-resource.openai.azure.com/"
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = "gpt-35-turboo"  # Your deployment name
    # azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")  # Default version
    
    
    
    # Initialize Azure OpenAI model
    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version="2024-02-01",
        temperature=0.7
    )

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are an HR assistant. Craft a professional, polite, and encouraging email to a job applicant.
    Inform them that they have successfully qualified the current stage of the hiring process.

    Job Description: {job_description}
    Current Stage: {current_stage}
    Next Stage: {next_stage}

    Ensure the email includes:
    - A congratulatory tone
    - Reference to the job role and selection stage
    - What the next stage involves
    - Next steps or instructions
    - Encouragement to prepare
    - Professional email format with subject and body

    Keep it reusable and concise.
    """)

    # Generate email content
    try:
        chain = prompt_template | model
        response = chain.invoke({
            "job_description": job_description,
            "current_stage": current_stage,
            "next_stage": next_stage
        })
        
        message = response.content if hasattr(response, "content") else str(response)
        
        # Create proper email format
        subject = f"Congratulations! You've Advanced to {next_stage} - {job_description}"
        email_body = f"Subject: {subject}\n\n{message}"
        
        
        
    except Exception as e:
        print(f"Error generating email content: {e}")
        return

    # Send email
    try:
        with smtplib.SMTP('smtp.zoho.in', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            
            server.sendmail(sender_email, recipients_email, message)
            
            
            print("Email sent successfully to:", recipients)
            
    except Exception as e:
        print(f"Error sending email: {e}")

