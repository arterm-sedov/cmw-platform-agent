from gradio_client import Client
import os
import sys
from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


def normalize_base_url(raw: str) -> str:
	url = raw.strip()
	if not (url.startswith("http://") or url.startswith("https://")):
		url = f"http://{url}"
	if not url.endswith("/"):
		url = url + "/"
	return url


def main() -> int:
	base_url = normalize_base_url(os.getenv("BASE_URL", "http://10.9.7.7:7860/"))
	session_id = os.getenv("SESSION_ID", "api5464") or None
	
	# Load CMW Platform credentials from environment
	cmw_base_url = os.getenv("CMW_BASE_URL", "")
	cmw_login = os.getenv("CMW_LOGIN", "")
	cmw_password = os.getenv("CMW_PASSWORD", "")
	
	print(f"Loaded as API: {base_url} ✔")
	print(f"CMW Platform: {cmw_base_url or 'Not configured'}")
	print(f"CMW Login: {cmw_login or 'Not configured'}")
	print(f"CMW Password: {'*' * len(cmw_password) if cmw_password else 'Not configured'}")

	# Final-only endpoint
	try:
		client = Client(base_url)
		kwargs = {
			"question": "Перечисли приложения в платформе",
		}
		
		# Add CMW credentials if available
		if cmw_login:
			kwargs["username"] = cmw_login
		if cmw_password:
			kwargs["password"] = cmw_password
		if cmw_base_url:
			kwargs["base_url"] = cmw_base_url
			
		if session_id:
			kwargs["session_id"] = session_id
			
		result = client.predict(api_name="/ask", **kwargs)
		print("/ask (final):", result)
	except Exception as e:
		print("/ask error:", e)
	return 0
	# Streaming endpoint
	try:
		client_stream = Client(base_url)
		kwargs = {
			"question": "Перечисли приложения в платформе",
		}
		
		# Add CMW credentials if available
		if cmw_login:
			kwargs["username"] = cmw_login
		if cmw_password:
			kwargs["password"] = cmw_password
		if cmw_base_url:
			kwargs["base_url"] = cmw_base_url
			
		if session_id:
			kwargs["session_id"] = session_id
			
		job = client_stream.submit(api_name="/ask_stream", **kwargs)
		print("/ask_stream (stream):")
		
		# Iterate through streaming chunks as they arrive
		try:
			chunk_count = 0
			for chunk in job:
				chunk_count += 1
				print(f"Chunk {chunk_count}: {chunk}")
				
		except Exception as stream_error:
			print(f"Streaming error: {stream_error}")
			# Fallback: try to get the final result
			try:
				result = job.result()
				print(f"Final result: {result}")
			except:
				print("No result available")
			
	except Exception as e:
		print("/ask_stream error:", e)

	return 0


if __name__ == "__main__":
	sys.exit(main())


