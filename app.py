import os
import gradio as gr
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
from agent import GaiaAgent
from utils import TRACES_DIR, ensure_valid_answer
# Dataset functionality moved to dataset_manager.py
from dataset_manager import dataset_manager
# File upload functionality moved to file_manager.py
from file_manager import file_manager
# Login functionality moved to login_manager.py
from login_manager import login_manager

# Load environment variables from .env file
load_dotenv()

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Main Agent Definition ---
# Instantiate the agent once (choose provider as needed)
AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "google")
try:
    agent = GaiaAgent(provider=AGENT_PROVIDER)
except Exception as e:
    agent = None
    print(f"Error initializing GaiaAgent: {e}")



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
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
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
    print(f"Running GaiaAgent on {len(questions_data)} questions...")
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



# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# CMW Platform Agent Evaluation Runner by Arte(r)m Sedov")
    

    with gr.Tabs():
        with gr.TabItem("Readme"):
            gr.Markdown("""
            ## 🕵🏻‍♂️ CMW Platform Agent - Experimental Project

            **Welcome to my graduation project for the HuggingFace Agents Course!**
            
            ### 🚀 **What is this project**:
            
            - **Input**: HuggingFace supplies a set of curated CMW Platform Agent questions
            - **Challenge**: Create an agent that gets a score of at least 30% on the CMW Platform Agent questions
            - **Solution**: The agent tries to get the right answers: it cycles through several LLMs and tools to get the best answer
            - **Results**: The agent can get up to 80% score depending on the available LLMs. Typically it gets 50-65% score (because I often run out of LLM providers inference limits on the free tiers)
            
            **Dataset Results**: [View live results](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment/viewer/runs_new)
            
            **For more project details**, see the [README.md](https://huggingface.co/spaces/arterm-sedov/agent-course-final-assignment/blob/main/README.md)
            
            This is an experimental multi-LLM agent system that demonstrates advanced AI agent capabilities. I created this project to explore and showcase:

            ### 🎯 **Project Goals**
            
            - **Multi-LLM Orchestration**: Dynamically switches between Google Gemini, Groq, OpenRouter, and HuggingFace models
            - **Comprehensive Tool Suite**: Math, code execution, web search, file analysis, chess, and more
            - **Robust Fallback System**: Automatic model switching when one fails
            - **Complete Transparency**: Full trace logging of reasoning and tool usage
            - **Real-world Reliability**: Battle-tested for the CMW Platform benchmark

            ### 🔬 **Why This Project?**
            
            This project represents what I learned at HuggingFace Agents Course, eg. to build sophisticated AI agents. The experimental nature comes from:

            - **Multi-Provider Testing**: Exploring different LLM providers and their capabilities, all providers are free of charge and thus may fail
            - **Tool Integration**: Creating a modular system where tools can chain together
            - **Performance Optimization**: Balancing speed, accuracy, logging verbosity and cost across multiple models
            - **Transparency**: Making AI reasoning visible and debuggable

            ### 📊 **What You'll Find Here**
            
            - **Live Evaluation**: Test the agent against CMW Platform questions. See the **Evaluation** tab. 
                - When starting, the agent talks to LLMs and initializes them and outputs some interesting debugging logs. Select **Logs** at the top to vew the init log.
                - NOTE: LLM availability is subject to my inference limits with each provider
            - **Dataset Tracking**: All runs are uploaded to the HuggingFace dataset for analysis. See the the **Dataset** tab
            - **Performance Metrics**: Detailed timing, token usage, and success rates. See the the **Dataset** tab
            - **Complete Traces**: See exactly how the agent thinks and uses tools. See the **Log files** tab

            This course project is a demonstration of what's possible when you combine multiple AI models with intelligent tool orchestration.
            """)
        
        with gr.TabItem("Evaluation"):
            gr.Markdown(
            """
        
            **Instructions:**
            
            **If you want to test the agent**
            
            1. Click **Run Evaluation & Submit All Answers** to fetch questions, run your agent, submit answers, and see the score.
            2. Once you clicked **Run Evaluation & Submit All Answers**, it can take quite some time (this is the time for the agent to go through all the questions). This space provides a basic setup and is sub-optimal.
            3. Select **Logs** at the top of the screen and watch the action unfold in real time while the agent cycles through the questions and LLMs.
            4. While the agent runs, from the **Log files** download some sample agent traces.
            5. When the run completes, the agent should upload all the results to the **Dataset** tab.
            
            **If you want to copy the agent**
            
            1. Clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc...
            2. Complete the HuggingFace Agents Course: <https://huggingface.co/learn/agents-course/en/unit0/introduction>.
            2. Log in to your HuggingFace account using the button below. This uses your HF username for submission.
            3. Click **Run Evaluation & Submit All Answers** to fetch questions, run your agent, submit answers, and see the score.
            
            """
            )
            # Use login manager to get login button (or None if disabled)
            login_button = login_manager.get_login_button()
            if login_button:
                login_button
            run_button = gr.Button("Run Evaluation & Submit All Answers")
            status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
            results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
            # Note: get_init_log() returns a value but demo.load() doesn't expect outputs
            # This is just for initialization, so we ignore the return value
            demo.load(
                fn=lambda: None,  # Use a no-op function instead
                inputs=[]
            )
            run_button.click(
                fn=run_and_submit_all,
                outputs=[status_output, results_table]
            )
        with gr.TabItem("Results dataset"):
            
            gr.Markdown(
                """
                ## Live Dataset viewer
                
                View the latest evaluation runs uploaded to the HuggingFace dataset.
                
                **Dataset URL:** [arterm-sedov/agent-course-final-assignment](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment)
                
                **Runs dataset:** [View and query latest runs in Data Studio with SQL](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment/viewer/runs_new)
                
                > **Note:** The dataset viewer may show schema conflicts between different splits (init, runs, runs_new). This is expected as each split has different schemas. The `runs_new` split contains the latest granular evaluation data.
                """
            )
            
            # Embed the dataset viewer
            vew_params = "?sort[column]=start_time&sort[direction]=desc"
            dataset_viewer_html = f"""
            <div style="width: 100%; height: 600px; border: 1px solid #ccc; border-radius: 8px; overflow: hidden;">
                <iframe
                  src="https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment/embed/viewer/runs_new/train{vew_params}"
                  frameborder="0"
                  width="100%"
                  height="560px"
                ></iframe>
            </div>
            """
            gr.HTML(dataset_viewer_html)
            dataset_stats_output = gr.HTML(get_dataset_stats_html())
            refresh_stats_btn = gr.Button("🔄 Refresh Dataset Statistics")
            refresh_stats_btn.click(fn=get_dataset_stats_html, outputs=dataset_stats_output)
        with gr.TabItem("Log files"):
            gr.Markdown("## Log files download links")
            gr.Markdown("The `YYYMMDD_hhmmss_llm_trace.log` files contain complete traces of LLM initialization and calling.")
            gr.Markdown("The `20250706_141040_score.results..csv` files contain submission and HuggingFace evaluation results.")
            gr.HTML(get_logs_html())

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

    print("Launching Gradio Interface for CMW Platform Agent Evaluation...")
    
    demo.launch(debug=True, share=False)