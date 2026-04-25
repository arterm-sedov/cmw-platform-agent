"""
Shared copy for the **``file_reference``** parameter, reused by record **document** and **image** tools.

Only phrasing that applies in more than one module lives here; tool-specific @tool ``description=``
strings are next to that tool in ``tool_record_document`` / ``tool_record_image``; generic field reads: ``tool_get_record_values`` (``get_record_values``).
"""

# Pydantic ``Field(description=...)`` for **file_reference** (e.g. **attach** tools).
CHAT_FILE_REFERENCE_DESCRIPTION = (
    "Which file to work with. Use the **name with extension** the user put in this chat, **or** "
    "the **``file_reference``** value returned from **fetch_record_document_file** or **fetch_rec"
    "ord_image_file** when the file was loaded from a record. Use that same value when you call "
    "another file or read tool. If the task is to use a public file on the web, a **https** (or **"
    "http**) **URL** is fine."
)

# Reused when composing the **fetch** @tool description in each record tool module.
CHAT_FILE_REFERENCE_RESULT_HINT = (
    "On success you get a **``file_reference``** string. Pass that **same** string to the next "
    "file or text tool; do not substitute a made-up or guessed name."
)
