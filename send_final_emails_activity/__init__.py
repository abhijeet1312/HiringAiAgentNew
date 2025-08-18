import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging
load_dotenv()


def send_final_emails_activity(email_data: dict):
    screening_email_input = email_data
    recipients = screening_email_input["qualified_candidates"]
    job_description = screening_email_input["job_description"]
    current_stage = screening_email_input["current_stage"]
    next_stage = screening_email_input["next_stage"]

    # Sender credentials
    sender_email = os.getenv("ZOHOMAIL_EMAIL")
   
    app_password = os.getenv("ZOHOMAIL_PASSWORD")
    

    # Azure OpenAI config
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = "gpt-35-turboo"

    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version="2024-02-01",
        temperature=0.7
    )

    # Prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are an HR assistant. Craft a professional, polite, and encouraging email to a job applicant.
    Inform them that they have successfully qualified the current stage of the hiring process.
    Include the following details:
    Recipient: {candidate_name}

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
    
    Please do not include any subject line in the email body.
    
    """)

    for candidate in recipients:
        candidate_name = candidate["name"]
        candidate_email = candidate["email"]

        try:
            # Generate email text from AI
            chain = prompt_template | model
            response = chain.invoke({
                "job_description": job_description,
                "current_stage": current_stage,
                "next_stage": next_stage,
                "candidate_name": candidate_name,
            })

            message = response.content if hasattr(response, "content") else str(response)
            # print(message)

            # subject = f"Congratulations! You've Advanced to {next_stage}"
            subject = f"Congratulations {candidate_name}! You have cleared {current_stage} – Next: {next_stage}"


            # Build proper MIME email
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = candidate_email
            msg["Subject"] = subject
            msg.attach(MIMEText(message, "plain"))

            # Send email
            with smtplib.SMTP("smtp.zoho.in", 587) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.sendmail(sender_email, candidate_email, msg.as_string())

            logging.info(f" Email sent successfully to {candidate_name} ({candidate_email})")

        except Exception as e:
            logging.info(f"Error processing {candidate_name} ({candidate_email}): {e}")


if __name__ == "__main__":
    test_email_data = {
        "qualified_candidates": [
            {
                "name": "Abhijeet Srivastava",
                "email": "abhijeetsrivastava2189@gmail.com"
            }
        ],
        "job_description": "AI Research Intern",
        "current_stage": "Resume Screening",
        "next_stage": "Technical Interview"
    }

    send_final_emails_activity(test_email_data)
