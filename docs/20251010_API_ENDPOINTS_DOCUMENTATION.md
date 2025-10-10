# CMW Platform Agent API Endpoints Documentation

**Date:** 2025-10-10  
**Version:** 1.0  
**Status:** Production Ready

## Overview

The CMW Platform Agent now exposes two REST API endpoints that allow external applications to interact with the agent programmatically. These endpoints support both single-turn and multi-turn conversations with session persistence.

## Base URL

```
http://localhost:7860
```

## Authentication

No authentication is required for these endpoints. All requests are processed with session isolation.

## Endpoints

### 1. `/ask` - Final Answer Endpoint

Returns the complete assistant response after processing is finished.

**Method:** `POST`  
**Path:** `/call/ask`  
**Content-Type:** `application/json`

#### Request Format

```json
{
  "data": ["Your question here"],
  "session_hash": "optional-session-id"
}
```

#### Parameters

- `data` (array, required): Contains the user's question as a string
- `session_hash` (string, optional): Session identifier for multi-turn conversations

#### Response Format

**Success Response:**
```json
{
  "event_id": "unique-event-id"
}
```

**Final Result (via GET):**
```json
{
  "data": ["Complete assistant response"]
}
```

#### Example Usage

**cURL:**
```bash
# Submit question
curl -X POST http://localhost:7860/call/ask \
  -H "Content-Type: application/json" \
  -d '{"data": ["Hello, who are you?"]}'

# Get result (replace EVENT_ID with actual ID)
curl -N http://localhost:7860/call/ask/EVENT_ID
```

**Python Client:**
```python
from gradio_client import Client

client = Client("http://localhost:7860/")
result = client.predict(
    question="Hello, who are you?",
    api_name="/ask"
)
print(result)
```

### 2. `/ask_stream` - Streaming Endpoint

Returns incremental chunks of the assistant response as it's being generated.

**Method:** `POST`  
**Path:** `/call/ask_stream`  
**Content-Type:** `application/json`

#### Request Format

```json
{
  "data": ["Your question here"],
  "session_hash": "optional-session-id"
}
```

#### Parameters

- `data` (array, required): Contains the user's question as a string
- `session_hash` (string, optional): Session identifier for multi-turn conversations

#### Response Format

**Success Response:**
```json
{
  "event_id": "unique-event-id"
}
```

**Streaming Results (via GET):**
```
event: generating
data: ["Hello"]

event: generating
data: ["Hello, w"]

event: generating
data: ["Hello, wo"]

event: generating
data: ["Hello, wor"]

event: generating
data: ["Hello, worl"]

event: generating
data: ["Hello, world!"]

event: complete
data: ["Hello, world!"]
```

#### Example Usage

**cURL:**
```bash
# Submit question
curl -X POST http://localhost:7860/call/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"data": ["Stream this please"]}'

# Get streaming result (replace EVENT_ID with actual ID)
curl -N http://localhost:7860/call/ask_stream/EVENT_ID
```

**Python Client:**
```python
from gradio_client import Client

client = Client("http://localhost:7860/")
job = client.submit(
    question="Stream this please",
    api_name="/ask_stream"
)

# Iterate through streaming chunks
for chunk in job:
    print(f"Chunk: {chunk}")
```

## Session Management

### Multi-turn Conversations

Both endpoints support session persistence using the `session_hash` parameter:

```python
# First message in a session
client = Client("http://localhost:7860/")
result1 = client.predict(
    question="What is 2+2?",
    api_name="/ask",
    session_hash="my-session-123"
)

# Follow-up message in the same session
result2 = client.predict(
    question="What about 3+3?",
    api_name="/ask", 
    session_hash="my-session-123"
)
```

### Session Behavior

- **With session_hash:** Messages are part of the same conversation context
- **Without session_hash:** Each request is treated as a new conversation
- **Session isolation:** Different session hashes maintain separate conversation histories

## Error Handling

### Common Error Responses

**Connection Error:**
```json
{
  "error": "Connection refused"
}
```

**Timeout Error:**
```json
{
  "error": "Request timeout"
}
```

**Invalid Request:**
```json
{
  "error": "Invalid request format"
}
```

### Error Event (Streaming)

For streaming endpoints, errors are returned as events:

```
event: error
data: ["Error message here"]
```

## Rate Limiting

- **Concurrent requests:** Limited by Gradio's queue system (default: 1)
- **Rate limits:** No explicit rate limiting implemented
- **Queue timeout:** 30 seconds per request

## Response Times

- **Final endpoint (`/ask`):** 2-10 seconds depending on complexity
- **Streaming endpoint (`/ask_stream`):** First chunk within 1-2 seconds, then incremental updates

## Best Practices

### 1. Use Appropriate Endpoint

- **Use `/ask`** for simple queries where you need the complete response
- **Use `/ask_stream`** for better user experience with real-time feedback

### 2. Session Management

- **Always use session_hash** for multi-turn conversations
- **Generate unique session IDs** for different users/conversations
- **Reuse session_hash** within the same conversation thread

### 3. Error Handling

```python
try:
    result = client.predict(question="Hello", api_name="/ask")
    print(result)
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

### 4. Streaming Best Practices

```python
# For streaming, always iterate through chunks
job = client.submit(question="Stream this", api_name="/ask_stream")
for chunk in job:
    # Process each chunk
    print(f"Received: {chunk}")
```

## Integration Examples

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

// Final endpoint
async function askQuestion(question) {
    const response = await fetch('http://localhost:7860/call/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [question] })
    });
    
    const { event_id } = await response.json();
    
    // Get result
    const resultResponse = await fetch(`http://localhost:7860/call/ask/${event_id}`);
    const result = await resultResponse.json();
    
    return result.data[0];
}
```

### Python with requests

```python
import requests
import json

def ask_question(question, session_hash=None):
    # Submit question
    payload = {"data": [question]}
    if session_hash:
        payload["session_hash"] = session_hash
    
    response = requests.post(
        "http://localhost:7860/call/ask",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    event_id = response.json()["event_id"]
    
    # Get result
    result_response = requests.get(f"http://localhost:7860/call/ask/{event_id}")
    return result_response.json()["data"][0]
```

## Testing

### Test Script

A test script is available at `agent_ng/_tests/api_test.py`:

```bash
# Run tests
python agent_ng/_tests/api_test.py

# With custom URL
BASE_URL=http://your-server:7860/ python agent_ng/_tests/api_test.py

# With session hash
SESSION_HASH=test-session-123 python agent_ng/_tests/api_test.py
```

### Manual Testing

1. **Start the agent:**
   ```bash
   python -m agent_ng.app_ng_modular
   ```

2. **Test final endpoint:**
   ```bash
   curl -X POST http://localhost:7860/call/ask \
     -H "Content-Type: application/json" \
     -d '{"data": ["Hello"]}'
   ```

3. **Test streaming endpoint:**
   ```bash
   curl -X POST http://localhost:7860/call/ask_stream \
     -H "Content-Type: application/json" \
     -d '{"data": ["Stream this"]}'
   ```

## Troubleshooting

### Common Issues

1. **"Application is initializing..."**
   - Wait for the agent to fully initialize
   - Check logs for initialization errors

2. **Connection refused**
   - Ensure the agent is running on the correct port
   - Check firewall settings

3. **Timeout errors**
   - Increase timeout values
   - Check server performance

4. **Empty responses**
   - Verify the question is not empty
   - Check agent configuration

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export GRADIO_DEBUG=1
export LOG_LEVEL=DEBUG
python -m agent_ng.app_ng_modular
```

## Changelog

### Version 1.0 (2025-10-10)
- Initial release of API endpoints
- Added `/ask` final answer endpoint
- Added `/ask_stream` streaming endpoint
- Implemented session management
- Added comprehensive documentation

## Support

For issues or questions regarding the API endpoints:

1. Check this documentation
2. Review the test script examples
3. Check the agent logs for error details
4. Verify the agent is running and accessible

---

**Note:** This documentation covers the API endpoints as implemented in the CMW Platform Agent. For Gradio-specific API details, refer to the [official Gradio documentation](https://www.gradio.app/guides/querying-gradio-apps-with-curl).
