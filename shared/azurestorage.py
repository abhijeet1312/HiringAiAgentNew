import os
import uuid
from datetime import datetime
from typing import List

from azure.storage.blob import BlobServiceClient, ContentSettings
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Response models
class ResumeUploadResponse(BaseModel):
    resumes: List[str]

class JobDescriptionUploadResponse(BaseModel):
    job_descriptions: str

class UploadResponse(BaseModel):  # Legacy compatibility
    resumes: List[str]
    job_descriptions: List[str]

# Azure Blob Storage Configuration
class AzureBlobConfig:
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        self.container_name = os.getenv("AZURE_CONTAINER_NAME")
        
        

        if not self.connection_string or not self.container_name:
            raise ValueError("Missing Azure Storage configuration in environment variables")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

# Initialize Azure config
azure_config = AzureBlobConfig()

def validate_pdf_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is a PDF"""
    return file.content_type == "application/pdf"

def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename using timestamp and UUID"""
    timestamp = int(datetime.now().timestamp() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(original_filename)
    return f"{timestamp}{unique_id}{name}{ext}"

async def upload_file_to_azure(file: UploadFile, folder: str) -> str:
    """Upload file to Azure Blob Storage and return its URL"""
    try:
        # Generate unique blob name
        unique_filename = generate_unique_filename(file.filename)
        blob_path = f"{folder}/{unique_filename}"

        # Read file content
        file_content = await file.read()

        # Upload to Azure
        blob_client = azure_config.container_client.get_blob_client(blob_path)
        blob_client.upload_blob(
            file_content,
            overwrite=True,
            content_settings=ContentSettings(content_type=file.content_type)
        )

        # Generate blob URL
        blob_url = blob_client.url
        return blob_url

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure upload failed: {str(e)}")

def delete_file_from_azure(blob_path: str):
    """Delete a blob from Azure Storage"""
    try:
        blob_client = azure_config.container_client.get_blob_client(blob_path)
        blob_client.delete_blob()
        # print(f"Deleted {blob_path} from Azure container {azure_config.container_name}")
    except Exception as e:
        print(f"Error deleting blob {blob_path}: {str(e)}")
import tempfile
import os
import re
from urllib.parse import unquote,urlparse
from fastapi import HTTPException

def extract_text_from_azure(blob_path: str) -> str:
    """
    Extract text from PDF/Word/Text stored in Azure Blob Storage with improved handling.
    :param blob_path: The blob path inside the container (e.g., 'resumes/file.pdf')
    """
    try:
        # Decode URL-encoded characters in the blob path
        decoded_blob_path = unquote(blob_path)
        # print(f"Original blob path: {blob_path}")
        # print(f"Decoded blob path: {decoded_blob_path}")
        # print(f"Fetching blob from Azure: {azure_config.container_name}/{decoded_blob_path}")
        
        blob_client = azure_config.container_client.get_blob_client(decoded_blob_path)
        blob_data = blob_client.download_blob().readall()
        
        # Extract text based on file type
        if decoded_blob_path.lower().endswith('.pdf'):
            text = extract_pdf_text(blob_data)
        elif decoded_blob_path.lower().endswith('.txt'):
            text = extract_txt_text(blob_data)
        elif decoded_blob_path.lower().endswith(('.doc', '.docx')):
            text = extract_docx_text(blob_data)
        else:
            # Fallback decode attempt
            try:
                text = blob_data.decode('utf-8')
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text = blob_data.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to decode file: {decoded_blob_path}"
                    )
        
        # Clean and normalize the extracted text
        cleaned_text = clean_extracted_text(text)
        return cleaned_text
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from Azure blob {decoded_blob_path if 'decoded_blob_path' in locals() else blob_path}: {str(e)}"
        )

def extract_pdf_text(blob_data: bytes) -> str:
    """Extract text from PDF using multiple methods in order of preference."""
    extraction_methods = [
        ("pdfminer.six (simple)", extract_pdf_with_pdfminer),
        ("pdfminer.six (advanced)", extract_pdf_with_pdfminer_advanced),
        ("PyPDF2 (fallback)", extract_pdf_with_pypdf2)
    ]
    
    for method_name, method_func in extraction_methods:
        try:
            # print(f"Trying {method_name}...")
            text = method_func(blob_data)
            
            # Check if extraction was successful
            if text and text.strip() and len(text.strip()) > 10:
                # print(f"Successfully extracted text using {method_name}")
                return text
            else:
                abhijeet="srivast"
                # print(f"{method_name} returned empty or insufficient text")
                
        except ImportError as e:
            print(f"{method_name} not available: {str(e)}")
            continue
        except Exception as e:
            print(f"{method_name} failed: {str(e)}")
            continue
    
    # If all methods fail
    raise HTTPException(
        status_code=500,
        detail="All PDF extraction methods failed. Please check if the PDF is valid and readable."
    )

def extract_pdf_with_pdfminer(blob_data: bytes) -> str:
    """Extract text using pdfminer.six - primary method."""
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        import io
        
        # Create layout parameters for better text extraction
        laparams = LAParams(
            boxes_flow=0.5,      # Controls how text boxes are grouped
            word_margin=0.1,     # Word separation threshold
            char_margin=2.0,     # Character separation threshold
            line_margin=0.5,     # Line separation threshold
            detect_vertical=False # Don't detect vertical text
        )
        
        # Extract text with custom parameters
        text = extract_text(
            io.BytesIO(blob_data),
            laparams=laparams,
            maxpages=0,          # Process all pages
            password="",         # No password
            caching=True         # Enable caching for performance
        )
        
        return text
        
    except ImportError:
        raise ImportError("pdfminer.six not installed")
    except Exception as e:
        print(f"pdfminer extraction failed: {str(e)}")
        return ""

def extract_pdf_with_pdfminer_advanced(blob_data: bytes) -> str:
    """Advanced pdfminer extraction with more control - alternative method."""
    try:
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.pdfpage import PDFPage
        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        import io
        
        # Set up PDF processing
        rsrcmgr = PDFResourceManager()
        output_string = io.StringIO()
        
        # Configure layout parameters for resume parsing
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            detect_vertical=False,
            all_texts=False
        )
        
        device = TextConverter(rsrcmgr, output_string, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        # Process each page
        for page in PDFPage.get_pages(
            io.BytesIO(blob_data),
            caching=True
        ):
            interpreter.process_page(page)
        
        text = output_string.getvalue()
        device.close()
        output_string.close()
        
        return text
        
    except ImportError:
        raise ImportError("pdfminer.six not installed")
    except Exception as e:
        print(f"Advanced pdfminer extraction failed: {str(e)}")
        return ""

def extract_pdf_with_pypdf2(blob_data: bytes) -> str:
    """Fallback PDF extraction using PyPDF2."""
    try:
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(blob_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        return ""

def extract_txt_text(blob_data: bytes) -> str:
    """Extract text from TXT files with encoding detection."""
    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return blob_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, use errors='ignore'
    return blob_data.decode('utf-8', errors='ignore')

def extract_docx_text(blob_data: bytes) -> str:
    """Extract text from DOCX files."""
    try:
        import docx2txt
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(blob_data)
            temp_file_path = temp_file.name
        
        text = docx2txt.process(temp_file_path)
        os.remove(temp_file_path)
        return text
        
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="docx2txt not installed. Install it using: pip install docx2txt"
        )

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text for better parsing."""
    if not text:
        return ""
    
    # Replace problematic Unicode characters with ASCII equivalents
    replacements = {
        '•': '- ',           # Bullet point
        '◦': '- ',           # White bullet
        '§': '',             # Section sign
        'ï': '',             # Various accented characters
        '–': '-',            # En dash
        '—': '-',            # Em dash
        ''': "'",            # Smart quotes
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',          # Ellipsis
        '\u00a0': ' ',       # Non-breaking space
        '\u2022': '- ',      # Bullet point
        '\u25e6': '- ',      # White bullet
        '\u2013': '-',       # En dash
        '\u2014': '-',       # Em dash
        '\ufb01': 'fi',      # fi ligature - THIS IS THE KEY FIX
        '\ufb02': 'fl',      # fl ligature
        '\ufb03': 'ffi',     # ffi ligature
        '\ufb04': 'ffl',     # ffl ligature
        '\u2018': "'",       # Left single quote
        '\u2019': "'",       # Right single quote
        '\u201c': '"',       # Left double quote
        '\u201d': '"',       # Right double quote
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = text.strip()
    
    # Final cleanup - remove any remaining problematic characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Ensure text is not empty after cleaning
    if not text or len(text.strip()) < 10:
        raise ValueError("Extracted text is empty or too short after cleaning")
    
    return text

# Additional helper function for debugging
def debug_text_extraction(blob_path: str) -> dict:
    """Debug function to analyze text extraction issues."""
    try:
        blob_client = azure_config.container_client.get_blob_client(blob_path)
        blob_data = blob_client.download_blob().readall()
        
        # Try to detect encoding
        import chardet
        detected = chardet.detect(blob_data[:1000])  # Check first 1000 bytes
        
        return {
            "file_size": len(blob_data),
            "detected_encoding": detected,
            "first_100_chars": str(blob_data[:100]),
            "file_extension": blob_path.split('.')[-1].lower()
        }
    except Exception as e:
        return {"error": str(e)}
    
    


async def cleanup_azure_files_after_processing(azure_objects: list) -> dict:
    """Delete files from Azure Blob Storage after processing."""

    cleanup_results = {
        "total_files": len(azure_objects),
        "successfully_deleted": 0,
        "failed_deletions": 0,
        "errors": []
    }

    for azure_obj in azure_objects:
        try:
            if not isinstance(azure_obj, dict) or 'container' not in azure_obj or 'blob_path' not in azure_obj:
                cleanup_results["errors"].append(f"Invalid Azure object info: {azure_obj}")
                cleanup_results["failed_deletions"] += 1
                continue

            container = azure_obj['container']
            blob_path = azure_obj['blob_path']
            
            # Decode URL-encoded characters in the blob path (KEY FIX)
            decoded_blob_path = unquote(blob_path)
            # print(f"Original blob path: {blob_path}")
            # print(f"Decoded blob path: {decoded_blob_path}")
            blob_service_client = azure_config.blob_service_client.from_connection_string(azure_config.connection_string)
            container_client = blob_service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(decoded_blob_path)

            blob_client.delete_blob()

            # print(f"Deleted {decoded_blob_path} from container {container}")
            cleanup_results["successfully_deleted"] += 1

        except Exception as e:
            error_msg = f"Error deleting Azure blob {azure_obj}: {str(e)}"
            print(error_msg)
            cleanup_results["errors"].append(error_msg)
            cleanup_results["failed_deletions"] += 1

    return cleanup_results








def parse_azure_url_to_container_blob_path(azure_url: str) -> dict:
    """
    Parse Azure Blob Storage URL to extract container and blob path.

    Example URL:
    https://<account>.blob.core.windows.net/<container>/<folder>/<filename>
    """
    try:
        parsed = urlparse(azure_url)

        # Example netloc: aurjobsaiagentsstorage.blob.core.windows.net
        # Example path: /aurjobs123/job_descriptions/file.pdf

        path_parts = parsed.path.lstrip('/').split('/', 1)

        if len(path_parts) != 2:
            raise ValueError("URL does not contain a container and blob path")

        container_name = path_parts[0]
        blob_path = path_parts[1]

        return {
            "container": container_name,
            "blob_path": blob_path
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error parsing Azure Blob URL {azure_url}: {str(e)}"
        )




import pandas as pd
import io

def is_excel_file(blob_path: str) -> bool:
    """Check if a blob path likely refers to an Excel file"""
    return blob_path.lower().endswith((".xlsx", ".xls"))


import io
import re
import pandas as pd
from urllib.parse import unquote

URL_RE = re.compile(r"https?://[^\s\"']+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{6,12}")

def _clean_val(v):
    if pd.isna(v) or v is None:
        return ""
    if isinstance(v, str):
        v = v.replace("\xa0", " ").strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        return v
    return str(v)

def parse_excel_candidates_from_azure(blob_path: str) -> list[dict]:
    """
    Read Excel from Azure and return one candidate dict per row.
    Each candidate will contain:
      - name, email, phone (if found)
      - resume_url (if a URL found in row)
      - resume_text (concatenation of all cleaned cell values)  <-- new
    """
    try:
        decoded_blob_path = unquote(blob_path)
        blob_client = azure_config.container_client.get_blob_client(decoded_blob_path)
        blob_data = blob_client.download_blob().readall()

        df = pd.read_excel(io.BytesIO(blob_data), header=0, dtype=object)
        # normalize columns
        df.columns = [str(c).strip().replace("\xa0", " ").lower() for c in df.columns]
        df = df.where(pd.notnull(df), None)

        candidates = []
        for idx, row in df.iterrows():
            # collect raw cell values in order
            row_vals = [row[col] for col in df.columns]
            cleaned_vals = [_clean_val(v) for v in row_vals if _clean_val(v) != ""]

            # build a single resume_text by joining with newlines (or space)
            resume_text = "\n".join(cleaned_vals).strip()

            # try find explicit url in row values
            resume_url = ""
            for v in row_vals:
                try:
                    sval = _clean_val(v)
                    if not sval:
                        continue
                    m = URL_RE.search(sval)
                    if m:
                        resume_url = m.group(0)
                        break
                except Exception:
                    continue

            # try find phone
            phone = ""
            for v in row_vals:
                s = _clean_val(v)
                if not s:
                    continue
                # simple numeric cleanup
                digits = re.sub(r"[^\d+]", "", s)
                if len(re.sub(r"[^\d]", "", digits)) >= 7:
                    phone = digits
                    break
                m = PHONE_RE.search(s)
                if m:
                    phone = re.sub(r"[^\d+]", "", m.group(0))
                    break

            # try find email
            email = ""
            for v in row_vals:
                s = _clean_val(v)
                if not s:
                    continue
                if "@" in s and "." in s.split("@")[-1]:
                    email = s
                    break

            # name heuristics: prefer 'name' column, else first non-url, non-phone text
            name = ""
            if "name" in df.columns:
                name = _clean_val(row.get("name", "") or "")
            if not name:
                for v in cleaned_vals:
                    if not URL_RE.search(v) and not PHONE_RE.search(v) and "@" not in v:
                        name = v
                        break

            candidate = {
                "name": name,
                "email": email,
                "phone": phone,
                "resume_url": resume_url,   # may be empty
                "resume_text": resume_text  # always present (could be empty string)
            }
            candidates.append(candidate)

        return candidates

    except Exception as e:
        raise RuntimeError(f"Error parsing Excel from Azure: {e}")
