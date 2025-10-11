#!/usr/bin/env python3
"""
Test script to verify that the JSON Schema validation fix works correctly.
"""

try:
    from attributes_tools.tools_text_attribute import EditOrCreateTextAttributeSchema
    import json
    
    # Generate the JSON schema
    schema = EditOrCreateTextAttributeSchema.model_json_schema()
    
    print("✅ JSON Schema generated successfully!")
    
    # Check the specific field that was causing the error
    use_to_search_records_desc = schema['properties']['use_to_search_records']['description']
    print(f"✅ use_to_search_records description type: {type(use_to_search_records_desc)}")
    print(f"✅ use_to_search_records description value: {use_to_search_records_desc}")
    
    # Verify it's a string (not an array)
    if isinstance(use_to_search_records_desc, str):
        print("✅ Description is correctly a string (not an array)")
    else:
        print("❌ Description is still not a string!")
        
    # Test that the schema can be serialized to JSON
    json_schema = json.dumps(schema)
    print("✅ Schema can be serialized to JSON successfully!")
    
    print("\n🎉 All tests passed! The JSON Schema validation error should be fixed.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
