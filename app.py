import os
import gradio as gr
from pathlib import Path
import requests
import inspect
import pandas as pd
import random
import datetime
import subprocess
import json
import re
import base64
from typing import Any
from dotenv import load_dotenv
from agent import CmwAgent
from utils import TRACES_DIR, ensure_valid_answer
# Dataset functionality moved to dataset_manager.py
from dataset_manager import dataset_manager
# File upload functionality moved to file_manager.py
from file_manager import file_manager
# Login functionality moved to login_manager.py
from login_manager import login_manager
import queue

# Load environment variables from .env file
load_dotenv()

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# --- Main Agent Definition ---
# Instantiate the agent once (choose provider as needed)
AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "gemini")
try:
    agent = None
    _agent_init_started = False
except Exception as e:
    agent = None
    _agent_init_started = False
    print(f"Error initializing CmwAgent: {e}")

# Helper to save DataFrame as CSV and upload via API
def save_df_to_csv(df, path):
    try:
        # Convert DataFrame to CSV string
        csv_content = df.to_csv(index=False, encoding="utf-8")
        
        # Upload via API
        success = file_manager.save_and_commit_file(
            file_path=path,
            content=csv_content,
            commit_message=f"Add results CSV {path}"
        )
        if success:
            print(f"✅ Results CSV uploaded successfully: {path}")
        else:
            print(f"⚠️ Results CSV upload failed, saved locally only: {path}")
            # Fallback to local save
            df.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Results CSV upload error: {e}, saving locally only")
        # Fallback to local save
        df.to_csv(path, index=False, encoding="utf-8")
    
    return path

# --- Provide init log for download on app load ---
def get_init_log():
    init_log_path = getattr(agent, "init_log_path", None)
    if init_log_path and os.path.exists(init_log_path):
        return init_log_path
    return None

# --- Provide init log content for display ---
def get_init_log_content():
    try:
        path = get_init_log()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return content if content.strip() else "(Init log is empty)"
        # Fallback: quick status if no file
        if agent is None:
            return "🟡 Initializing agent..."
        return "✅ Agent initialized. (No init log file found)"
    except Exception as _e:
        return "(Failed to read init log)"

def generate_run_id(timestamp: str, idx: int) -> str:
    """Generate a unique run ID for a question."""
    return f"{timestamp}_q{idx+1:02d}"

def upload_questions_with_results(results_log: list, timestamp: str, username: str, total_score: str, success_type: str = "final"):
    """
    Upload all questions with their results to the runs_new dataset.
    
    Args:
        results_log: List of question results
        timestamp: Timestamp for run IDs
        username: Username for the run
        total_score: Final score from evaluator
        success_type: Type of upload ("final evaluated results" or "unevaluated results")
    """
    successful_uploads = 0
    for idx, result in enumerate(results_log):
        try:
            run_id = generate_run_id(timestamp, idx)
            
            # Get LLM stats JSON for this run
            llm_stats_json = agent._get_llm_stats_json()
            
            # Create updated run data for this question
            run_data = create_run_data_for_runs_new(
                run_id,
                idx,
                len(results_log),
                result,
                llm_stats_json,
                username,
                total_score
            )
            
            success = dataset_manager.upload_run_data(run_data, split="runs_new")
            if success:
                print(f"✅ Uploaded question {idx+1} with {success_type}. Run ID: {run_id}")
                successful_uploads += 1
            else:
                print(f"⚠️ Failed to upload question {idx+1} with {success_type}")
                
        except Exception as e:
            print(f"⚠️ Failed to upload question {idx+1}. Error: {e}")
    
    return successful_uploads

def create_run_data_for_runs_new(
    run_id: str,
    idx: int,
    total_questions: int,
    result: dict,
    llm_stats_json: dict,
    username: str = "N/A",
    total_score: str = "N/A"
) -> dict:
    """
    Create run data for the runs_new split.
    
    Args:
        run_id: Unique identifier for the run
        idx: Index of the question in the batch (0-based)
        total_questions: Total number of questions in the batch
        result: Individual result dictionary
        llm_stats_json: LLM statistics JSON
        username: Username of the person running the agent
        total_score: Overall score for the complete evaluation run
        
    Returns:
        dict: Run data for upload to runs_new split
    """
    # Extract trace data from result
    trace = result.get("trace", {})
    
    # Extract final_result from trace
    final_result = trace.get("final_result", {})
    
    file_name = trace.get("file_name", "")
    
    question = trace.get("question", "")
    
    return {
        "run_id": run_id,
        "questions_count": f"{idx+1}/{total_questions}",
        "input_data": json.dumps([{
            "task_id": result.get("task_id", f"task_{idx+1:03d}"),
            "question": question or "N/A",
            "file_name": file_name or "N/A"
        }]),
        "reference_answer": final_result.get("reference", "N/A"),
        "final_answer": final_result.get("submitted_answer", "N/A"),
        "reference_similarity": float(final_result.get("similarity_score", 0.0)),
        "question": question or "N/A",
        "file_name": file_name or "N/A",
        "file_size": trace.get("file_size", 0),
        "llm_used": final_result.get("llm_used", "N/A"),  # LLM used
        "llm_stats_json": json.dumps(llm_stats_json),  # LLM statistics JSON
        "total_score": total_score or "N/A",  # Overall score for the complete evaluation run
        "start_time": trace.get("start_time") or "N/A",  # Start time with fallback
        "end_time": trace.get("end_time") or "N/A",  # End time with fallback
        "total_execution_time": float(trace.get("total_execution_time", 0.0)),  # Total execution time with fallback, ensure float
        "tokens_total": int(trace.get("tokens_total", 0)),  # Tokens total with fallback, ensure int
        "llm_traces_json": json.dumps(trace.get("llm_traces", {})),
        "logs_json": json.dumps(trace.get("logs", [])),
        "per_llm_stdout_json": json.dumps(trace.get("per_llm_stdout", [])),
        "full_debug": trace.get("debug_output", "N/A"),
        "error": final_result.get("error", "N/A"),  # Error information
        "username": username.strip() if username else "N/A"
    }

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the CmwAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")
    
    # Use login manager to validate login
    is_valid, username, error_message = login_manager.validate_login_for_operation(profile, "evaluation")
    if not is_valid:
        return error_message, None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (already done globally)
    if agent is None:
        return "Error initializing agent. Check logs for details.", None
    agent_code = f"https://huggingface.co/spaces/{username}/agent-course-final-assignment/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run the Agent
    results_log = []
    results_log_df = []
    answers_payload = []
    print(f"Running CmwAgent on {len(questions_data)} questions...")
    # Select all questions randomly
    questions_data = random.sample(questions_data, len(questions_data))
    # DEBUG: Select one random task instead of all
    # questions_data = random.sample(questions_data, 1)
    #questions_data = [questions_data[0]]
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name", "")  # Extract file_name from question data
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        # Download file if one is referenced
        file_data = None
        if file_name and file_name.strip():
            try:
                print(f"\U0001F4C1 Downloading file: {file_name} for task {task_id}")
                file_url = f"{api_url}/files/{task_id}"
                file_response = requests.get(file_url, timeout=30)
                file_response.raise_for_status()
                
                # Convert file to base64
                file_data = base64.b64encode(file_response.content).decode('utf-8')
                print(f"✅ Downloaded and encoded file: {file_name} ({len(file_data)} chars)")
            except Exception as e:
                print(f"⚠️ Failed to download file {file_name} for task {task_id}: {e}")
                file_data = None
        
        try:
            # Pass both question text and file data to agent
            if file_data:
                # Create enhanced question with file context
                enhanced_question = f"{question_text}\n\n[File attached: {file_name} - base64 encoded data available]"
                agent_result = agent(enhanced_question, file_data=file_data, file_name=file_name)
            else:
                agent_result = agent(question_text)
            
            # Extract answer and additional info from agent result
            # Extract data from the trace structure
            trace = agent_result  # The entire trace is now the result
            final_result = trace.get("final_result", {})
            submitted_answer = final_result.get("submitted_answer", "N/A")
            
            # Use helper function to ensure valid answer
            submitted_answer = ensure_valid_answer(submitted_answer)
            
            reference_similarity = final_result.get("similarity_score", 0.0)
            llm_used = final_result.get("llm_used", "unknown")
            reference_answer = final_result.get("reference", "N/A")
            question_text = trace.get("question", "")
            file_name = trace.get("file_name", "")
        
            
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "task_id": task_id, 
                "trace": trace,
                "full_debug": ""
            })
            # Shorter results for dataframe for gradio table 
            results_log_df.append({
                "task_id": task_id, 
                "question": question_text, 
                "file_name": file_name, 
                "submitted_answer": submitted_answer,
                "reference_answer": reference_answer,
                "reference_similarity": reference_similarity,
                "llm_used": llm_used
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "task_id": task_id, 
                "question": question_text, 
                "file_name": file_name, 
                "submitted_answer": f"AGENT ERROR: {e}",
                "reference_answer": reference_answer,
                "reference_similarity": 0.0,
                "llm_used": "none",
                "trace": trace, 
                "full_debug": "",
                "error": str(e)
            })
            results_log_df.append({
                "task_id": task_id, 
                "question": question_text, 
                "file_name": file_name, 
                "submitted_answer": f"AGENT ERROR: {e}",
                "reference_answer": "N/A",
                "reference_similarity": 0.0,
                "llm_used": "none"
            })

    # --- Convert results to dataframe ---
    results_df = pd.DataFrame(results_log_df)
    
    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", results_df

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Note: Questions will be uploaded after evaluator response with final scores
    print(f"📊 Prepared {len(results_log)} questions for evaluation")

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    total_score = "N/A (not evaluated)"
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        status_message = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print(status_message)
        print("Submission successful.")
        # Extract just the score percentage from the result data
        total_score = f"{result_data.get('score', 'N/A')}% ({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
            
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        print(status_message)
        # Set error score result
        total_score = "N/A (Submission Failed)"
        
        print(f"⚠️ Submission failed: {e}")
            
    # Upload questions once after submission attempt (success or failure)
    try:
        if len(results_log) > 0:
            print(f"✅ Uploading all questions with results: {timestamp}")
            successful_uploads = upload_questions_with_results(results_log, timestamp, username, total_score, "final")
            
            # Log complete evaluation run status
            if successful_uploads == len(results_log):
                print(f"✅ All evaluation runs uploaded with results: {timestamp}")
            else:
                print(f"⚠️ Failed to upload some evaluation runs: {successful_uploads}/{len(results_log)} questions uploaded")
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")
        
    return status_message, results_df

def get_dataset_stats_html():
    """
    Get dataset statistics and return as HTML.
    """
    try:
        from datasets import load_dataset
        
        # Load each config separately
        configs = ['init', 'runs_new']
        stats_html = "<div style='margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px;'>"
        stats_html += "<h3>📊 Dataset Statistics</h3>"
        
        for config_name in configs:
            try:
                # Load specific config
                config_data = load_dataset("arterm-sedov/agent-course-final-assignment", config_name)
                
                stats_html += f"<div style='margin: 15px 0; padding: 10px; background: #e9ecef; border-radius: 5px;'>"
                stats_html += f"<h4>🔧 Config: {config_name.upper()}</h4>"
                
                # Get statistics for each split in this config
                for split_name in config_data.keys():
                    split_data = config_data[split_name]
                    stats_html += f"<div style='margin: 8px 0;'>"
                    stats_html += f"<strong>{split_name.upper()} Split:</strong> {len(split_data)} records"
                    stats_html += "</div>"
                
                # Add latest run info for runs_new config
                if config_name == "runs_new" and "default" in config_data:
                    runs_new_data = config_data["default"]
                    if len(runs_new_data) > 0:
                        latest_run = runs_new_data[-1]
                        stats_html += f"<div style='margin: 10px 0; padding: 8px; background: #d4edda; border-radius: 3px;'>"
                        stats_html += f"<strong>Latest Run:</strong> {latest_run.get('run_id', 'N/A')}"
                        stats_html += f"<br><strong>Total Score:</strong> {latest_run.get('total_score', 'N/A')}"
                        stats_html += f"<br><strong>Username:</strong> {latest_run.get('username', 'N/A')}"
                        stats_html += "</div>"
                
                stats_html += "</div>"
                
            except Exception as config_error:
                stats_html += f"<div style='margin: 15px 0; padding: 10px; background: #f8d7da; border-radius: 5px;'>"
                stats_html += f"<h4>❌ Config: {config_name.upper()}</h4>"
                stats_html += f"<div style='margin: 8px 0; color: #721c24;'>Error loading config: {config_error}</div>"
                stats_html += "</div>"
        
        stats_html += "</div>"
        return stats_html
        
    except Exception as e:
        return f"<div style='margin: 20px 0; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;'>⚠️ Could not load dataset statistics: {e}</div>"

def get_logs_html():
    logs_dir = "logs"
    rows = []
    files = []
    
    # Get space ID for repository links
    space_id = os.getenv("SPACE_ID", "arterm-sedov/agent-course-final-assignment")
    repo_base_url = f"https://huggingface.co/spaces/{space_id}/resolve/main"
    
    if os.path.exists(logs_dir):
        for fname in os.listdir(logs_dir):
            fpath = os.path.join(logs_dir, fname)
            if os.path.isfile(fpath):
                timestamp, dt = extract_timestamp_from_filename(fname)
                if not dt:
                    # Fallback to modification time for files without timestamp in filename
                    dt = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
                    timestamp = dt.strftime('%Y-%m-%d %H:%M:%S (mtime)')
                files.append((fname, timestamp, dt, fpath))
        # Sort all files by datetime descending (newest first)
        files.sort(key=lambda x: x[2], reverse=True)
        for fname, timestamp, dt, fpath in files:
            # Create repository download link
            repo_download_url = f"{repo_base_url}/logs/{fname}?download=true"
            download_link = f'<a href="{repo_download_url}" target="_blank" rel="noopener noreferrer">Download from Repo</a>'
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            rows.append(f"<tr><td>{fname}</td><td>{date_str}</td><td>{download_link}</td></tr>")
    
    table_html = (
        "<table border='1' style='width:100%;border-collapse:collapse;'>"
        "<thead><tr><th>File Name</th><th>Date/Time</th><th>Download</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return table_html

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename using comprehensive regex patterns for all log formats in @/logs.
    Returns (timestamp_str, datetime_obj) or (None, None) if no timestamp found.
    """
    import re
    
    # Handle multiple extensions by removing all extensions
    name = filename
    while '.' in name:
        name = os.path.splitext(name)[0]
    
    # 1. 14-digit datetime: YYYYMMDDHHMMSS (must be exact 14 digits)
    m = re.match(r'^(\d{14})$', name)
    if m:
        timestamp_str = m.group(1)
        try:
            dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            return timestamp_str, dt
        except ValueError:
            pass
    
    # 2. Leaderboard format: 2025-07-02 090007
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[ _]+(\d{2})(\d{2})(\d{2})', name)
    if m:
        y, mo, d, h, mi, s = m.groups()
        try:
            dt = datetime.datetime.strptime(f"{y}{mo}{d}{h}{mi}{s}", "%Y%m%d%H%M%S")
            return f"{y}-{mo}-{d} {h}:{mi}:{s}", dt
        except ValueError:
            pass
    
    # 3. LOG prefix with 12-digit timestamp: LOG202506281412
    m = re.match(r'^LOG(\d{12})$', name)
    if m:
        timestamp_str = m.group(1)
        try:
            dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            return f"LOG{timestamp_str}", dt
        except ValueError:
            pass
    
    # 4. LOG prefix with 8-digit date and optional suffix: LOG20250628_2, LOG20250629_1
    m = re.match(r'^LOG(\d{8})(?:_(\d+))?$', name)
    if m:
        date_str, suffix = m.groups()
        try:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            timestamp_str = f"LOG{date_str}"
            if suffix:
                timestamp_str += f"_{suffix}"
            return timestamp_str, dt
        except ValueError:
            pass
    
    # 5. INIT prefix with date and time: INIT_20250704_000343
    m = re.match(r'^INIT_(\d{8})_(\d{6})$', name)
    if m:
        date_str, time_str = m.groups()
        try:
            dt = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return f"INIT_{date_str}_{time_str}", dt
        except ValueError:
            pass
    
    # 6. Date with underscore and time: 20250702_202757, 20250703_135654
    m = re.match(r'^(\d{8})_(\d{6})$', name)
    if m:
        date_str, time_str = m.groups()
        try:
            dt = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return f"{date_str}_{time_str}", dt
        except ValueError:
            pass
    
    # 7. Date only (8 digits): 20250628
    m = re.match(r'^(\d{8})$', name)
    if m:
        date_str = m.group(1)
        try:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            return date_str, dt
        except ValueError:
            pass
    
    # 8. Files with no timestamp pattern (like "Score 60.log")
    # These will return None and fall back to modification time
    
    return None, None

def save_results_log(results_log: list) -> str:
    """
    Save the complete results log to a file and upload via API.
    
    Args:
        results_log (list): List of dictionaries containing task results
        
    Returns:
        str: Path to the saved log file, or None if failed
    """
    try:
        # Create traces directory if it doesn't exist
        os.makedirs(TRACES_DIR, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare log content
        log_content = json.dumps(results_log, indent=2, ensure_ascii=False)
        log_path = f"{TRACES_DIR}/{timestamp}_llm_trace.log"
        
        return log_path
        
    except Exception as e:
        print(f"⚠️ Failed to save results log: {e}")
        return None

def chat_with_agent(message, history):
    """
    Chat with the agent using a simple message interface.
    
    Args:
        message (str): User's message
        history (list): Chat history
        
    Returns:
        tuple: (updated_history, message)
    """
    if not message.strip():
        return history, ""
    
    if agent is None:
        error_msg = "Error: Agent not initialized. Check logs for details."
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}], ""
    
    try:
        print(f"💬 Chat request: {message}")
        
        # Build minimal chat history for agent: only user/assistant turns
        chat_history = []
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                chat_history.append({"role": role, "content": content})

        # Call the agent with the user's message and history
        # The agent now always returns a generator for streaming, but we can get trace data
        if agent is None:
            error_msg = "Error: Agent became unavailable during processing. Check logs for details."
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}], ""
        
        result = agent(message, chat_history=chat_history)
        
        # Handle the result properly - check if it's the problematic generator object
        accumulated_response = ""
        if DEBUG_MODE:
            print(f"chat_with_agent: Agent returned result type: {type(result)}")
        
        try:
            # Check if we got a generator object that should be consumed
            if hasattr(result, '__iter__') and hasattr(result, '__next__') and not isinstance(result, (str, dict, list)):
                if DEBUG_MODE:
                    print("chat_with_agent: Detected generator, attempting to consume...")
                chunk_count = 0
                for chunk in result:
                    chunk_count += 1
                    # Commented out verbose debug logging
                    # if DEBUG_MODE:
                    #     print(f"chat_with_agent: Generator yielded chunk {chunk_count}: {type(chunk)} - {str(chunk)[:100]}")
                    if isinstance(chunk, str):
                        accumulated_response += chunk
                    elif isinstance(chunk, dict):
                        # If it's a dict, try to extract text content
                        text = chunk.get("content", chunk.get("text", chunk.get("delta", "")))
                        if text:
                            accumulated_response += str(text)
                
                # Commented out verbose debug logging
                # if DEBUG_MODE:
                #     print(f"chat_with_agent: Generator yielded {chunk_count} chunks, accumulated: '{accumulated_response[:100]}'")
                
                # If no chunks were yielded, the generator was empty
                if chunk_count == 0:
                    print("chat_with_agent: WARNING - Generator yielded no content")
                    accumulated_response = "❌ No response generated. The LLM returned an empty response, likely due to API issues or rate limits."
            else:
                # Not a generator, handle as direct response
                if DEBUG_MODE:
                    print(f"chat_with_agent: Non-generator response: {str(result)[:100]}")
                accumulated_response = str(result)
        except Exception as e:
            print(f"Error consuming generator: {e}")
            # Fallback: try to get trace data directly from agent
            if hasattr(agent, 'get_trace_data'):
                trace = agent.get_trace_data()
                final_result = trace.get("final_result", {})
                accumulated_response = final_result.get("submitted_answer", f"Error processing response: {e}")
            else:
                accumulated_response = f"Error processing response: {e}"
        
        # Try to get the final answer from trace data if accumulated_response is empty or contains only step info
        if not accumulated_response or accumulated_response.strip() == "" or "generator object" in accumulated_response.lower():
            print("chat_with_agent: Trying to get final answer from trace data...")
            if hasattr(agent, 'get_trace_data'):
                trace = agent.get_trace_data()
                final_result = trace.get("final_result", {})
                submitted_answer = final_result.get("submitted_answer", "")
                if submitted_answer and submitted_answer != "No answer provided":
                    accumulated_response = submitted_answer
                    print(f"chat_with_agent: Got final answer from trace: {submitted_answer[:100]}...")
                else:
                    # Try to extract from the full trace
                    llm_traces = trace.get("llm_traces", [])
                    for llm_trace in llm_traces:
                        if "response" in llm_trace and llm_trace["response"]:
                            response_text = str(llm_trace["response"])
                            if "FINAL ANSWER" in response_text.upper():
                                # Extract the final answer part
                                import re
                                match = re.search(r'FINAL ANSWER\s*:?\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
                                if match:
                                    accumulated_response = match.group(1).strip()
                                    print(f"chat_with_agent: Extracted final answer from response: {accumulated_response[:100]}...")
                                    break
        
        # Final safety check - catch any generator object representations
        if not accumulated_response or accumulated_response.strip() == "" or "generator object" in accumulated_response.lower():
            print("chat_with_agent: Final fallback - providing helpful error message")
            # Try to get more info from the agent
            if hasattr(agent, 'get_trace_data'):
                trace = agent.get_trace_data()
                final_result = trace.get("final_result", {})
                llm_used = final_result.get("llm_used", "unknown")
                error_info = final_result.get("error", "No error details available")
                accumulated_response = f"❌ **No response generated**\n\n**Details:**\n- LLM used: {llm_used}\n- Error: {error_info}\n\n**Possible causes:**\n- API rate limits (OpenRouter: 429 error)\n- Authentication issues\n- Service unavailable\n\nPlease try again later or check the Init logs for more details."
            else:
                accumulated_response = "❌ **No response generated**\n\nThe LLM failed to produce a response. This usually indicates:\n- API rate limits\n- Authentication issues\n- Service unavailable\n\nPlease try again later."
        
        # Parse the accumulated response to separate step indicators from final answer
        answer = accumulated_response or "No answer generated"
        
        # Get the trace data from the agent (this is now collected internally)
        if agent is not None and hasattr(agent, 'get_trace_data'):
            trace = agent.get_trace_data()
        else:
            trace = {}
        final_result = trace.get("final_result", {}) if trace else {}
        llm_used = final_result.get("llm_used", "unknown")
        
        # Parse the response to separate step indicators from the final answer
        import re
        
        # Extract the final answer (look for the actual answer after step indicators)
        # Try multiple patterns to catch the FINAL ANSWER
        final_answer_patterns = [
            r'FINAL ANSWER:\s*(.*)',
            r'FINAL ANSWER\s*:\s*(.*)',
            r'FINAL ANSWER:\s*(.*?)(?=\n\n|\n---|$)',
            r'FINAL ANSWER\s*:\s*(.*?)(?=\n\n|\n---|$)'
        ]
        
        final_answer_match = None
        for pattern in final_answer_patterns:
            final_answer_match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                break
        
        # Also try to extract from the accumulated response directly
        if not final_answer_match:
            for pattern in final_answer_patterns:
                final_answer_match = re.search(pattern, accumulated_response, re.DOTALL | re.IGNORECASE)
                if final_answer_match:
                    break
        
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
        else:
            # If no FINAL ANSWER marker, try to extract the last meaningful content
            # Look for content that's not step indicators or process info
            lines = answer.split('\n')
            final_answer = ""
            
            # First, try to find content that looks like a real answer (not process info)
            for line in reversed(lines):
                line = line.strip()
                if (line and 
                    not line.startswith('📍') and 
                    not line.startswith('🔄') and 
                    not line.startswith('🔧') and 
                    not line.startswith('✅') and
                    not line.startswith('Validate:') and
                    not line.startswith('Intent:') and
                    not line.startswith('Plan:') and
                    not line.startswith('Execute:') and
                    not line.startswith('Result:') and
                    not line.startswith('No answer provided') and
                    len(line) > 10):  # Only consider substantial content
                    final_answer = line
                    break
            
            # If still no answer, try to extract from the tool execution results
            if not final_answer or final_answer == "No answer provided":
                # Look for tool execution results in the accumulated response
                if "list_applications" in answer and "success" in answer.lower():
                    # Extract the actual data from the tool response
                    # Look for the JSON data in the response
                    json_pattern = r'\{.*?"success".*?"raw_response".*?\}'
                    json_match = re.search(json_pattern, answer, re.DOTALL)
                    if json_match:
                        try:
                            import json
                            tool_data = json.loads(json_match.group(0))
                            if tool_data.get("success") and "raw_response" in tool_data:
                                apps = tool_data["raw_response"]
                                if isinstance(apps, list) and len(apps) > 0:
                                    # Format the applications list
                                    app_list = []
                                    for app in apps:
                                        name = app.get("name", "Unknown")
                                        alias = app.get("alias", "Unknown")
                                        app_list.append(f"• **{name}** (system name: {alias})")
                                    final_answer = "Here is the list of applications available in the Platform:\n\n" + "\n".join(app_list)
                                    print(f"chat_with_agent: Extracted applications from tool response: {len(apps)} apps")
                        except Exception as e:
                            print(f"chat_with_agent: Error parsing tool response: {e}")
            
            # If still no answer, try to get it from the agent's trace data
            if not final_answer or final_answer == "No answer provided":
                if hasattr(agent, 'get_trace_data'):
                    trace = agent.get_trace_data()
                    final_result = trace.get("final_result", {})
                    submitted_answer = final_result.get("submitted_answer", "")
                    if submitted_answer and submitted_answer != "No answer provided":
                        final_answer = submitted_answer
                        print(f"chat_with_agent: Got final answer from trace: {submitted_answer[:100]}...")
            
            if not final_answer:
                final_answer = answer
        
        # Debug: Print what we're about to display
        print(f"chat_with_agent: Final answer to display: '{final_answer[:500]}...'")
        print(f"chat_with_agent: Final answer length: {len(final_answer)}")
        
        # Create the main response with the final answer prominently displayed
        response = f"🤖 **Agent Response** (using {llm_used}):\n\n{final_answer}"
        
        # Add step indicators as a collapsible section if they exist
        step_content = ""
        step_lines = []
        for line in answer.split('\n'):
            if '📍' in line or '🔄' in line or '🔧' in line or '✅' in line:
                step_lines.append(line)
        
        if step_lines:
            step_content = '\n'.join(step_lines)
        
        # Get detailed information about the multi-model approach
        response_parts = []
        response_parts.append(response)
        
        # Add step indicators as a collapsible section if they exist
        if step_content.strip():
            response_parts.append({
                "role": "assistant",
                "content": step_content,
                "metadata": {"title": "🔍 Process Details"}
            })
        
        # Add information about the multi-model approach
        if hasattr(agent, 'llm_tracking'):
            multi_model_info = []
            multi_model_info.append("🔍 **Multi-Model Approach:**")
            
            # Show which models were attempted
            attempted_models = []
            for provider, tracking in agent.llm_tracking.items():
                if tracking['total_attempts'] > 0:
                    status = "✅ Success" if tracking['successes'] > 0 else "❌ Failed"
                    attempted_models.append(f"• **{provider}**: {status} ({tracking['successes']}/{tracking['total_attempts']} attempts)")
            
            if attempted_models:
                multi_model_info.append("\n".join(attempted_models))
            
            # Add overall statistics
            total_attempts = sum(tracking['total_attempts'] for tracking in agent.llm_tracking.values())
            total_successes = sum(tracking['successes'] for tracking in agent.llm_tracking.values())
            
            if total_attempts > 0:
                overall_success_rate = (total_successes / total_attempts) * 100
                multi_model_info.append(f"\n📊 **Overall**: {total_successes}/{total_attempts} successful responses ({overall_success_rate:.1f}% success rate)")
            
            if len(multi_model_info) > 1:
                response_parts.append({
                    "role": "assistant",
                    "content": "\n".join(multi_model_info),
                    "metadata": {"title": "🤖 Model Statistics"}
                })
        
        # Add information about tools used if available
        if 'llm_traces' in trace:
            tool_usage = []
            for llm_trace in trace.get('llm_traces', []):
                if 'tool_calls' in llm_trace and llm_trace['tool_calls']:
                    for tool_call in llm_trace['tool_calls']:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_usage.append(f"• {tool_name}")
            
            if tool_usage:
                tools_info = "🛠️ **Tools Used:**\n" + "\n".join(set(tool_usage))
                response_parts.append({
                    "role": "assistant",
                    "content": tools_info,
                    "metadata": {"title": "🛠️ Tools Used"}
                })
        
        # Add execution time if available
        if 'total_execution_time' in trace:
            exec_time = trace['total_execution_time']
            exec_info = f"⏱️ **Execution Time**: {exec_time:.2f} seconds"
            response_parts.append({
                "role": "assistant",
                "content": exec_info,
                "metadata": {"title": "⏱️ Performance"}
            })
        
        # Return updated history with proper message format for Gradio chatbot
        # For now, just return the main response and add collapsible sections as separate messages
        updated_history = history + [{"role": "user", "content": message}]
        
        # Add the main response
        updated_history.append({"role": "assistant", "content": response})
        
        # Add collapsible sections as separate messages
        for part in response_parts[1:]:
            if isinstance(part, dict):
                updated_history.append(part)
        
        return updated_history, ""
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"Chat error: {e}")
        updated_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]
        return updated_history, ""

def chat_with_agent_stream(message, history):
	"""
	Stream assistant output by yielding partial responses. Compute once; reveal in small chunks.
	"""
	if not message.strip():
		yield history, ""
		return
	if agent is None:
		yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Error: Agent not initialized. Check logs for details."}], ""
		return
	try:
		print(f"💬 Chat (stream) request: {message}")
		working_history = history + [{"role": "user", "content": message}]
		yield working_history, ""

		# Build chat history for agent
		chat_history = []
		for turn in history:
			role = turn.get("role")
			content = turn.get("content", "")
			if role in ("user", "assistant") and content:
				chat_history.append({"role": role, "content": content})

		# Try true streaming if agent exposes a streaming generator
		stream_gen = None
		for method_name in ("stream", "stream_response", "stream_chat", "stream_answer"):
			candidate = getattr(agent, method_name, None)
			if callable(candidate):
				try:
					stream_gen = candidate(message, chat_history=chat_history)
					break
				except TypeError:
					# Different signature; try calling with just message
					try:
						stream_gen = candidate(message)
						break
					except Exception:
						pass

		if stream_gen is not None:
			accum = ""
			try:
				for chunk in stream_gen:
					# Accept either plain string chunks or dict events with an answer delta
					if isinstance(chunk, dict):
						text = chunk.get("answer_delta") or chunk.get("delta") or chunk.get("text") or ""
					else:
						text = str(chunk)
					if not text:
						continue
					accum += text
					yield working_history + [{"role": "assistant", "content": accum}], ""
				# Finish without extras (no second full call)
				yield working_history + [{"role": "assistant", "content": accum}], ""
				return
			except Exception:
				# Fall back to non-streaming path
				pass

		# Use unified streaming path with enhanced terminal output and error handling
		accum = ""
		terminal_output = ""
		chunk_count = 0
		
		if agent is not None:
			# Try to use the agent's call method which should return a generator
			try:
				result = agent(message, chat_history=chat_history)
				print(f"chat_with_agent_stream: Agent returned result type: {type(result)}")
				
				# Check if result is a generator and consume it properly
				if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
					print("chat_with_agent_stream: Consuming generator...")
					has_content = False
					for delta in result:
						print(f"chat_with_agent_stream: Generator yielded: {type(delta)} - {str(delta)[:100]}")
						if isinstance(delta, str) and delta.strip():
							chunk_count += 1
							accum += delta
							has_content = True
							# Yield partial updates for streaming
							display_content = accum
							if hasattr(agent, '_stream_terminal_output'):
								terminal_chunk = agent._stream_terminal_output()
								if terminal_chunk:
									terminal_output += terminal_chunk
									display_content += f"\n\n**Terminal Output:**\n{terminal_output}"
							yield working_history + [{"role": "assistant", "content": display_content}], ""
						elif isinstance(delta, dict):
							# If it's a dict, try to extract text content
							text = delta.get("content", delta.get("text", delta.get("delta", "")))
							if text and str(text).strip():
								chunk_count += 1
								accum += str(text)
								has_content = True
								# Yield partial updates for streaming
								display_content = accum
								if hasattr(agent, '_stream_terminal_output'):
									terminal_chunk = agent._stream_terminal_output()
									if terminal_chunk:
										terminal_output += terminal_chunk
										display_content += f"\n\n**Terminal Output:**\n{terminal_output}"
								yield working_history + [{"role": "assistant", "content": display_content}], ""
					
					# If no meaningful content was yielded, provide fallback message
					if not has_content or chunk_count == 0:
						print("chat_with_agent_stream: WARNING - Generator yielded no meaningful content")
						# Try to get error information from the agent
						if hasattr(agent, 'get_trace_data'):
							trace = agent.get_trace_data()
							final_result = trace.get("final_result", {})
							llm_used = final_result.get("llm_used", "unknown")
							error_info = final_result.get("error", "No error details available")
							accum = f"❌ **No response generated**\n\n**Details:**\n- LLM used: {llm_used}\n- Error: {error_info}\n\n**Possible causes:**\n- API rate limits (OpenRouter: 429 error)\n- Authentication issues\n- Service unavailable\n\n**Fallback system:** The agent should try multiple LLM providers, but all may be currently unavailable.\n\nPlease try again later or check the Init logs for more details."
						else:
							accum = "❌ **No response generated**\n\nThe LLM failed to produce a response. This usually indicates:\n- API rate limits\n- Authentication issues\n- Service unavailable\n\nPlease try again later."
						yield working_history + [{"role": "assistant", "content": accum}], ""
				else:
					# Not a generator, convert to string
					chunk_count = 1
					accum = str(result)
					print(f"chat_with_agent_stream: Non-generator result: {accum}")
					
					# Check for and stream terminal output if available
					if hasattr(agent, '_stream_terminal_output'):
						terminal_chunk = agent._stream_terminal_output()
						if terminal_chunk:
							terminal_output += terminal_chunk
					
					# Combine main response with terminal output if present
					display_content = accum
					if terminal_output:
						display_content += f"\n\n**Terminal Output:**\n{terminal_output}"
					
					yield working_history + [{"role": "assistant", "content": display_content}], ""
				
				# If no chunks were yielded, handle empty generator
				if chunk_count == 0:
					print("chat_with_agent_stream: WARNING - Generator yielded no content")
					# Try to get trace data for error information
					if hasattr(agent, 'get_trace_data'):
						trace = agent.get_trace_data()
						final_result = trace.get("final_result", {})
						llm_used = final_result.get("llm_used", "unknown")
						error_info = final_result.get("error", "No error details available")
						accum = f"❌ **No response generated**\n\n**Details:**\n- LLM used: {llm_used}\n- Error: {error_info}\n\n**Possible causes:**\n- API rate limits (OpenRouter: 429 error)\n- Authentication issues\n- Service unavailable\n\nPlease try again later or check the Init logs for more details."
					else:
						accum = "❌ **No response generated**\n\nThe LLM failed to produce a response. This usually indicates:\n- API rate limits\n- Authentication issues\n- Service unavailable\n\nPlease try again later."
					yield working_history + [{"role": "assistant", "content": accum}], ""
			except Exception as e:
				print(f"chat_with_agent_stream: Error in streaming: {e}")
				# Check if the error might be related to generator object display
				if "generator object" in str(e).lower() or chunk_count == 0:
					accum = "❌ **No response generated**\n\nThe LLM failed to produce a response. This usually indicates:\n- API rate limits\n- Authentication issues\n- Service unavailable\n\nPlease try again later."
				else:
					accum = f"❌ Error processing response: {e}"
				yield working_history + [{"role": "assistant", "content": accum}], ""
		else:
			# Fallback when agent is None or doesn't have stream method
			accum = "Error: Agent not properly initialized. Please check the initialization logs."
			yield working_history + [{"role": "assistant", "content": accum}], ""

		# After stream finishes, get the finalized trace and add tool information
		if agent is not None:
			trace = getattr(agent, "_trace_get_full", None)
			if callable(trace):
				trace = trace()
			else:
				trace = {"final_result": {"submitted_answer": accum, "llm_used": "unknown"}}
		else:
			trace = {"final_result": {"submitted_answer": accum, "llm_used": "unknown"}}
		
		# Extract tools, model, and execution time information
		tools_list = []
		llm_used = "unknown"
		exec_time = None
		
		try:
			# Extract tools
			if 'llm_traces' in trace:
				for llm_trace in trace.get('llm_traces', []) or []:
					for tool_call in (llm_trace.get('tool_calls') or []):
						tool_name = tool_call.get('name', 'unknown')
						if tool_name:
							tools_list.append(f"• {tool_name}")
			
			# Extract model and execution time
			if 'final_result' in trace:
				final_result = trace['final_result']
				llm_used = final_result.get('llm_used', 'unknown')
				exec_time = final_result.get('execution_time', None)
		except Exception as _e:
			pass
		
		# Combined metadata message with both tools and stats (no duplication)
		combined_sections = []
		
		# Add tools section if tools were used
		if tools_list:
			unique_tools = sorted(set(tools_list))
			tools_text = "\n".join(unique_tools)
			combined_sections.append(f"**🛠️ Tools Used:**\n{tools_text}")
		
		# Add stats section (model and execution time)
		stats_lines = []
		if llm_used and llm_used != "unknown":
			stats_lines.append(f"Model: {llm_used}")
		if isinstance(exec_time, (int, float)):
			stats_lines.append(f"Execution Time: {exec_time:.2f} s")
		
		if stats_lines:
			stats_text = "\n".join(stats_lines)
			combined_sections.append(f"**ℹ️ Details:**\n{stats_text}")
		
		# Show combined information in one collapsible section
		if combined_sections:
			combined_text = "\n\n".join(combined_sections)
			yield working_history + [{"role": "assistant", "content": combined_text, "metadata": {"title": "📊 Response Details"}}], ""
	except Exception as e:
		err = f"❌ Error: {e}"
		print(f"Chat stream error: {e}")
		yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": err}], ""

def get_available_models():
    """
    Get information about initialized models and their status.
    """
    if agent is None:
        status = "🟡 Initializing agent..." if '_agent_init_started' in globals() and _agent_init_started else "❌ Agent not initialized"
        return status
    
    models_info = []
    
    # Check only initialized models
    for provider_key, llm_instance in agent.llm_instances.items():
        if llm_instance is None:
            continue
        
        config = agent.LLM_CONFIG.get(provider_key, {})
        provider_name = config.get("name", provider_key.title())
        
        # Get the active model configuration for this provider
        active_model_config = agent.active_model_config.get(provider_key, {})
        model_name = active_model_config.get("model", "Unknown")
        token_limit = active_model_config.get("token_limit", "Unknown")
        
        models_info.append(f"**{provider_name}:**\n• {model_name} (max {token_limit} tokens)\n")
    
    return "\n".join(models_info)

# Timer poller to update status text and stop timer when ready
def poll_models():
    text = get_available_models()
    show_logs_btn = False
    try:
        show_logs_btn = (agent is None) and ('_agent_init_started' in globals() and _agent_init_started)
    except Exception:
        show_logs_btn = False
    # Stop timer once agent is initialized
    if agent is not None:
        return text, gr.update(active=False), gr.update(visible=False)
    return text, gr.update(), gr.update(visible=show_logs_btn)

# Stream agent initialization logs into chatbot
def stream_agent_init_logs(chat_history):
	log_queue = queue.Queue()
	accum_text = ""
	messages = chat_history or []
	
	# Mark initialization as started so status shows a proper message
	global _agent_init_started
	try:
		_agent_init_started = True
	except Exception:
		pass
	
	def sink(chunk: str):
		try:
			log_queue.put(chunk)
		except Exception:
			pass
	
	import threading, time
	def worker():
		global agent
		try:
			agent_local = CmwAgent(provider=AGENT_PROVIDER, log_sink=sink)
			agent = agent_local
		except Exception as e:
			log_queue.put(f"\n🔴 Agent init failed: {e}\n")
	
	thread = threading.Thread(target=worker, daemon=True)
	thread.start()
	
	last_activity = time.time()
	quiet_seconds_to_finish = 1.0
	
	# Continuously flush chunks into the chatbot
	while True:
		try:
			chunk = log_queue.get(timeout=0.25)
			last_activity = time.time()
			accum_text += chunk
			
			# Add terminal output streaming section
			terminal_section = ""
			if agent is not None and hasattr(agent, '_stream_terminal_output'):
				terminal_chunk = agent._stream_terminal_output()
				if terminal_chunk:
					if not hasattr(stream_agent_init_logs, '_terminal_buffer'):
						stream_agent_init_logs._terminal_buffer = ""
					stream_agent_init_logs._terminal_buffer += terminal_chunk
					terminal_section = f"\n\n**🖥️ Terminal Output:**\n```\n{stream_agent_init_logs._terminal_buffer}\n```"
			
			# Combine initialization logs with terminal output
			display_content = accum_text + terminal_section
			
			# Update the last assistant message or append a new one
			if messages and messages[-1].get("role") == "assistant":
				messages[-1] = {"role": "assistant", "content": display_content}
			else:
				messages = messages + [{"role": "assistant", "content": display_content}]
			yield messages
		except queue.Empty:
			# If thread finished or agent is set and we've been quiet long enough, stop streaming
			if (not thread.is_alive()) or (agent is not None and (time.time() - last_activity) > quiet_seconds_to_finish):
				break
			continue
	# Append completion line if agent initialized
	if agent is not None:
		completion_note = "\n✅ Initialization complete."
		if completion_note not in accum_text:
			accum_text += completion_note
			if messages and messages[-1].get("role") == "assistant":
				messages[-1] = {"role": "assistant", "content": accum_text}
			else:
				messages = messages + [{"role": "assistant", "content": accum_text}]
	# Final yield to ensure UI has the latest
	yield messages

# Background agent initializer (lazy)
def _init_agent_background():
	global agent, _agent_init_started
	if _agent_init_started or agent is not None:
		return
	_agent_init_started = True
	try:
		print("🟡 Initializing CmwAgent in background...")
		agent_local = CmwAgent(provider=AGENT_PROVIDER)
		agent = agent_local
		print("🟢 CmwAgent initialized.")
	except Exception as e:
		agent = None
		print(f"🔴 Failed to initialize CmwAgent: {e}")

def _start_agent_init_thread_with_sink(update_fn=None):
	import threading
	def _sink_writer(text_chunk: str):
		try:
			if update_fn:
				update_fn(text_chunk)
		except Exception:
			pass
	thread = threading.Thread(target=lambda: CmwAgent(provider=AGENT_PROVIDER, log_sink=_sink_writer) and _assign_agent(), daemon=True)
	thread.start()
	return None

# Helper to assign the global agent if constructed in thread
def _assign_agent():
	global agent
	# This relies on CmwAgent writing to global when constructed; instead we construct here
	agent_local = CmwAgent(provider=AGENT_PROVIDER)
	agent = agent_local
	return None

# --- Build Gradio Interface using Blocks ---
# Ensure Gradio can serve local static resources via /gradio_api/file=
RESOURCES_DIR = Path(__file__).parent / "resources"
try:
    existing_allowed = os.environ.get("GRADIO_ALLOWED_PATHS", "")
    parts = [p for p in existing_allowed.split(os.pathsep) if p]
    if str(RESOURCES_DIR) not in parts:
        parts.append(str(RESOURCES_DIR))
    os.environ["GRADIO_ALLOWED_PATHS"] = os.pathsep.join(parts)
    print(f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}")
except Exception as _e:
    print(f"Warning: could not set GRADIO_ALLOWED_PATHS: {_e}")
with gr.Blocks(css_paths=[Path(__file__).parent / "resources" / "css" / "gradio_comindware.css"]) as demo:
    gr.Markdown("# Analyst Copilot", elem_classes=["hero-title"]) 
    # Start agent initialization when the app loads

    with gr.Tabs():
        
        
        with gr.TabItem("Chat with Agent"): 
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ## 💬 Chat with the Comindware Analyst Copilot
                    
                    **Welcome!** This agent focuses on the **Comindware Platform** entity operations (applications, templates, attributes) and uses deterministic tools to execute precise changes.
                    
                    ### 🎯 **How it works:**
                    
                    - **Platform Operations First**: Validates your intent and executes tools for entity changes (e.g., create/edit attributes)
                    - **Multi-Model Orchestration**: Tries multiple LLM providers with intelligent fallback
                    - **Compact Structured Output**: Intent → Plan → Validate → Execute → Result
                    """, elem_classes=["chat-hints"]) 
                with gr.Column():
                    gr.Markdown("""
                    ### 💡 **Try asking:**
                    
                    - List all applications in the Platform
                    - List all record templates in app 'ERP'
                    - List all attributes in template 'Counterparties', app 'ERP'
                    - Create plain text attribute 'Comment', app 'HR', template 'Candidates'
                    - Create 'Customer ID' text attribute, app 'ERP', template 'Counterparties', custom input mask: ([0-9]{10}|[0-9]{12})
                    - For attribute 'Contact Phone' in app 'CRM', template 'Leads', change display format to Russian phone
                    - Fetch attribute: system name 'Comment', app 'HR', template 'Candidates'
                    - Archive/unarchive attribute, system name 'Comment', app 'HR', template 'Candidates'
                    """, elem_classes=["chat-hints"]) 
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=500,
                        show_label=True,
                        container=True,
                        type="messages"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["cmw-button"])
                    
                    # Clear button
                    clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes=["cmw-button"])
                
                with gr.Column(scale=1, elem_classes=["sidebar-card"]):
                    # Model information panel (single block)
                    with gr.Column(elem_classes=["model-card"]):
                        gr.Markdown("### 🤖 Model status & stats")
                        models_info = gr.Markdown(get_available_models())
                        # Automatically refresh model info on load as well
                        demo.load(get_available_models, outputs=models_info)
                        status_timer = gr.Timer(1.0, active=True)
                        with gr.Row():
                            open_logs_btn = gr.Button("📜 Open Init logs", elem_classes=["cmw-button"], visible=False)
                            refresh_models_btn = gr.Button("🔄 Refresh model info", elem_classes=["cmw-button"]) 
                        status_timer.tick(fn=poll_models, outputs=[models_info, status_timer, open_logs_btn])
                    # Quick action buttons (grouped like model card)
                    with gr.Column(elem_classes=["quick-actions-card"]):
                        gr.Markdown("### ⚡ Quick actions")
                        quick_list_apps_btn = gr.Button("🔎 List all apps", elem_classes=["cmw-button"])
                        quick_math_btn = gr.Button("🧩 Create text attribute", elem_classes=["cmw-button"]) 
                        quick_code_btn = gr.Button("🛠️ Edit phone mask", elem_classes=["cmw-button"]) 
                        qa_capital_btn = gr.Button("Capital of France?", elem_classes=["cmw-button"]) 
                        qa_math_btn = gr.Button("15 * 23 + 7 = ?", elem_classes=["cmw-button"]) 
                        qa_code_btn = gr.Button("Python prime check function", elem_classes=["cmw-button"]) 
                        qa_explain_btn = gr.Button("Explain ML vs DL briefly", elem_classes=["cmw-button"]) 
            
            # Event handlers
            def send_message(message, history):
                return chat_with_agent(message, history)
            
            def clear_chat():
                return [], ""
            
            def quick_math(history):
                message = (
                    "Draft a plan to CREATE a text attribute 'Customer ID' in application 'ERP', template 'Counterparties' "
                    "with display_format=CustomMask and mask ([0-9]{10}|[0-9]{12}), system_name=CustomerID. "
                    "Provide Intent, Plan, Validate, and a DRY-RUN payload preview (compact JSON) for the tool call, "
                    "but DO NOT execute any changes yet. Wait for my confirmation."
                )
                return chat_with_agent(message, history)
            
            def quick_code(history):
                message = (
                    "Prepare a safe EDIT plan for attribute 'Contact Phone' (system_name=ContactPhone) in application 'CRM', template 'Leads' "
                    "to change display_format to PhoneRuMask. Provide Intent, Plan, Validate checklist (risk notes), and a DRY-RUN payload preview. "
                    "Do NOT execute changes yet—await my approval."
                )
                return chat_with_agent(message, history)
            
            def quick_list_apps(history):
                message = (
                    "List all applications in the Platform. "
                    "Format nicely using Markdown. "
                    "Show system names and descriptions if any."
                )
                return chat_with_agent(message, history)
            
            def qa_capital_example(history):
                message = "Capital of France?"
                return chat_with_agent(message, history)
            
            def qa_arith_example(history):
                message = "15 * 23 + 7 = ?"
                return chat_with_agent(message, history)
            
            def qa_prime_example(history):
                message = "Write a Python function to check if a number is prime."
                return chat_with_agent(message, history)
            
            def qa_explain_example(history):
                message = "Explain ML vs DL briefly"
                return chat_with_agent(message, history)
            
            # Connect event handlers
            send_btn.click(
                fn=chat_with_agent,  # Use the working non-streaming version
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                fn=chat_with_agent,  # Use the working non-streaming version
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, msg]
            )
            
            refresh_models_btn.click(
                fn=poll_models,
                outputs=[models_info, status_timer, open_logs_btn]
            )
            
            open_logs_btn.click(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => { const t=[...document.querySelectorAll('button[role=tab]')].find(b=>b.textContent.trim()==='Init logs'); if(t) t.click(); }"
            )

            
            quick_math_btn.click(
                fn=quick_math,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
            
            quick_code_btn.click(
                fn=quick_code,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
            
            quick_list_apps_btn.click(
                fn=quick_list_apps,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
            
            qa_capital_btn.click(
                fn=qa_capital_example,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
            qa_math_btn.click(
                fn=qa_arith_example,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
            qa_code_btn.click(
                fn=qa_prime_example,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
            qa_explain_btn.click(
                fn=qa_explain_example,
                inputs=[chatbot],
                outputs=[chatbot, msg]
            )
        with gr.TabItem("Init logs"):
            init_chat = gr.Chatbot(label="Initialization logs", height=400, type="messages", render_markdown=False, elem_classes=["terminal-chat"])
            # Show a starting message
            init_chat.value = [{"role": "assistant", "content": "Starting agent initialization..."}]
            demo.load(stream_agent_init_logs, inputs=init_chat, outputs=init_chat)

        with gr.TabItem("Readme"):
            gr.Markdown("""
            ## 🕵🏻‍♂️ Comindware Analyst Copilot - Entity Creation System

            **Welcome to the Comindware Analyst Copilot - an AI-powered system for creating entities in the Comindware Platform!**
            
            ### 🚀 **What is this project**:
            
            - **Input**: Users provide natural language requests to create entities in the CMW Platform
            - **Challenge**: Translate natural language into CMW Platform API calls for entity creation
            - **Solution**: The agent uses multiple LLMs and specialized tools to create templates, attributes, workflows, and other platform entities
            - **Results**: The agent can successfully create entities with 50-65% success rate, up to 80% with all LLMs available
            
            **Dataset Results**: [View live results](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment/viewer/runs_new)
            
            **For more project details**, see the [README.md](https://huggingface.co/spaces/arterm-sedov/agent-course-final-assignment/blob/main/README.md)
            
            This is an experimental multi-LLM agent system that demonstrates advanced AI agent capabilities for business process automation. The project showcases:

            ### 🎯 **Project Goals**
            
            - **Entity Creation**: Create templates, attributes, workflows, and other CMW Platform entities
            - **Multi-LLM Orchestration**: Dynamically switches between Google Gemini, Groq, OpenRouter, and HuggingFace models
            - **Comprehensive Tool Suite**: CMW Platform API integration, web search, code execution, file analysis, and more
            - **Robust Fallback System**: Automatic model switching when one fails
            - **Complete Transparency**: Full trace logging of reasoning and tool usage
            - **Real-world Reliability**: Battle-tested for CMW Platform entity creation

            ### 🔬 **Why This Project?**
            
            This project represents advanced AI agent development for business process management. The experimental nature comes from:

            - **Multi-Provider Testing**: Exploring different LLM providers and their capabilities for entity creation tasks
            - **Tool Integration**: Creating a modular system where tools can chain together for complex entity creation
            - **Performance Optimization**: Balancing speed, accuracy, and cost across multiple models
            - **Transparency**: Making AI reasoning visible and debuggable for business users
            - **CMW Platform Integration**: Bridging natural language requests with platform API capabilities

            ### 📊 **What You'll Find Here**
            
            - **Live Entity Creation**: Test the agent for creating CMW Platform entities. See the **Evaluation** tab. 
                - When starting, the agent initializes LLMs and outputs debugging logs. Select **Logs** at the top to view the init log.
                - NOTE: LLM availability is subject to inference limits with each provider
            - **Dataset Tracking**: All entity creation attempts are uploaded to the HuggingFace dataset for analysis. See the **Dataset** tab
            - **Performance Metrics**: Detailed timing, token usage, and success rates for entity creation. See the **Dataset** tab
            - **Complete Traces**: See exactly how the agent thinks and uses tools for entity creation. See the **Log files** tab

            ### 🏢 **CMW Platform Integration**
            
            This agent is designed to work with the Comindware Platform, a business process management and workflow automation platform. The agent can:
            
            - **Create Templates**: Define data structures with custom attributes
            - **Configure Workflows**: Set up business processes and automation rules
            - **Manage Entities**: Create, update, and configure platform objects
            - **API Integration**: Interact with CMW Platform APIs for entity management
            
            For more information about the Comindware Platform, see the [CMW Platform Documentation](https://github.com/arterm-sedov/cbap-mkdocs-ru).

            This project demonstrates what's possible when you combine multiple AI models with intelligent tool orchestration for business process automation.
            """)

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Comindware Analyst Copilot Evaluation...")
    demo.launch(debug=True, share=False)