import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging
load_dotenv()


# --- NEW: AI-based helper to extract company name from job description text ---
# CHANGED: Replaced the previous regex-based extractor with this AI-based extractor.
def extract_company_name_from_jd_with_ai(jd_text: str, model) -> str:
    """
    Ask the LLM to read jd_text and return ONLY the company name (one short line).
    If no company can be confidently determined, the model must return the single token: NONE
    Returns empty string if NONE or not found.
    """
    if not jd_text or not jd_text.strip():
        return ""

    # Very strict prompt so model returns exactly one token (company name) or NONE.
    extract_prompt = ChatPromptTemplate.from_template("""
You are an assistant that extracts structured information from job description text.
Return ONLY the company name mentioned in the following Job Description. 
- If you can identify a company, return only the company name (one short phrase, no extra words).
- If you cannot confidently find a company name, return ONLY the token: NONE

Job Description:
{job_description}
""")

    try:
        chain = extract_prompt | model
        resp = chain.invoke({"job_description": jd_text})
        # model response might be accessible differently depending on SDK version:
        raw = ""
        if hasattr(resp, "content"):
            raw = resp.content
        else:
            raw = str(resp)

        # clean result: take first non-empty line, strip punctuation and whitespace
        first_line = ""
        for line in raw.splitlines():
            if line.strip():
                first_line = line.strip()
                break

        # if model returns NONE or similar, return empty
        if not first_line or first_line.upper().strip() == "NONE" or "no company" in first_line.lower():
            return ""

        # Trim long outputs (defensive) and remove trailing punctuation
        company = first_line.strip()
        company = company.rstrip('.,;:-')
        if len(company) > 80:  # defensive cutoff
            company = company[:80].rsplit(" ", 1)[0]

        return company

    except Exception as e:
        # on any error, fallback to empty so email still sends
        logging.warning(f"Company extraction via AI failed: {e}")
        return ""


def send_final_emails_activity(email_data: dict):
    screening_email_input = email_data
    recipients = screening_email_input["qualified_candidates"]
    job_description = screening_email_input.get("job_description", "")
    current_stage = screening_email_input.get("current_stage", "")
    next_stage = screening_email_input.get("next_stage", "")

    # Sender credentials
    sender_email = os.getenv("ZOHOMAIL_EMAIL")
    app_password = os.getenv("ZOHOMAIL_PASSWORD")

    # Azure OpenAI config
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = "gpt-35-turboo"

    # CHANGED: create a deterministic model (temperature=0.0) for extraction,
    # so the company extraction is stable and consistent.
    extractor_model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version="2024-02-01",
        temperature=0.0  # deterministic for extraction
    )

    # CHANGED: extract the company name once and reuse for all candidates
    company_name = extract_company_name_from_jd_with_ai(job_description, extractor_model)

    # Main model for email generation (kept slightly creative)
    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version="2024-02-01",
        temperature=0.7
    )

    # CHANGED: included Company: {company_name} and asked model to use it where appropriate.
    prompt_template = ChatPromptTemplate.from_template("""
    You are an HR assistant. Craft a professional, polite, and encouraging email to a job applicant.
    Inform them that they have successfully qualified the current stage of the hiring process.
    Also mention that due to large volume of applicants , there will be Automated ai interviews in the next stage.
    Include the following details:
    Recipient: {candidate_name}
    Company: {company_name}

    Job Description: {job_description}
    Current Stage: {current_stage}
    Next Stage: {next_stage}

    Ensure the email includes:
    - A congratulatory tone
    - Reference to the job role and selection stage and the company name where appropriate
    - What the next stage involves
    - Next steps or instructions
    - Encouragement to prepare
    - Professional email format with subject and body

    Keep it reusable and concise.
    
    Please do not include any subject line in the email body.
    
    Also add this at the end of the email:
    Best regards,
    {company_name} HR Team
    """)

    for candidate in recipients:
        # defensive access
        candidate_name = candidate.get("name", "")
        candidate_email = candidate.get("email", "")

        try:
            # Generate email text from AI, passing the company_name we extracted earlier
            chain = prompt_template | model
            response = chain.invoke({
                "job_description": job_description,
                "current_stage": current_stage,
                "next_stage": next_stage,
                "candidate_name": candidate_name,
                "company_name": company_name
            })

            message = response.content if hasattr(response, "content") else str(response)
            # print(message)

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
        # Example JD with explicit company name; try changing to test extractor behavior
        "job_description": "Company: Aurora AI Labs\nRole: AI Research Intern\nWe are hiring...",
        "current_stage": "Resume Screening",
        "next_stage": "Technical Interview"
    }

    send_final_emails_activity(test_email_data)
