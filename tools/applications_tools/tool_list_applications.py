from ..tool_utils import *

ATTRIBUTE_ENDPOINT = "webapi/Solution"

@tool("list_applications", return_direct=False)
def list_applications() -> Dict[str, Any]:
    """
    List all applications, configured in the Platform.
    The resulting list depends on the user's access rights.

    Returns:
        dict: {
            "success": bool - True if application list was fetched successfully
            "status_code": int - HTTP response status code  
            "data": list|None - List of applications if successful
            "error": str|None - Error message if operation failed
        }
    """

    endpoint = f"{ATTRIBUTE_ENDPOINT}"

    result = requests_._get_request(endpoint)

    return execute_list_operation(
        response_data = result,
        result_model=AttributeResult
    )

if __name__ == "__main__":
    results = list_applications()
    print(results)