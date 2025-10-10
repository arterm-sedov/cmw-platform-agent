from gradio_client import Client
import os
import sys


def normalize_base_url(raw: str) -> str:
	url = raw.strip()
	if url.startswith("http://http://"):
		url = url.replace("http://http://", "http://", 1)
	if url.startswith("https://https://"):
		url = url.replace("https://https://", "https://", 1)
	if not (url.startswith("http://") or url.startswith("https://")):
		url = f"http://{url}"
	if not url.endswith("/"):
		url = url + "/"
	return url


def main() -> int:
	base_url = normalize_base_url(os.getenv("BASE_URL", "http://http://127.0.0.1:7860/"))
	session_hash = os.getenv("SESSION_HASH", "") or None
	print(f"Loaded as API: {base_url} ✔")

	# Final-only endpoint
	try:
		client = Client(base_url)
		kwargs = {"question": "Hello from /ask"}
		if session_hash:
			kwargs["session_hash"] = session_hash
		result = client.predict(api_name="/ask", **kwargs)
		print("/ask (final):", result)
	except Exception as e:
		print("/ask error:", e)

	# Streaming endpoint
	try:
		client_stream = Client(base_url)
		kwargs = {"question": "Hello streaming from /ask_stream"}
		if session_hash:
			kwargs["session_hash"] = session_hash
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


