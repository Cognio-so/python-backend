#!/usr/bin/env python
"""
Test script to verify the Replicate API token is set correctly.
This follows the exact reference implementation from Replicate.
"""

import os
import sys
from dotenv import load_dotenv
import replicate

def main():
    """Test Replicate API token and run a simple model."""
    # Load environment variables
    load_dotenv()
    
    # Check if REPLICATE_API_TOKEN is set directly
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    
    # If not set directly, try to set it from REPLICATE_API_KEY
    if not api_token:
        api_key = os.environ.get("REPLICATE_API_KEY")
        if api_key:
            os.environ["REPLICATE_API_TOKEN"] = api_key
            api_token = api_key
            print(f"Set REPLICATE_API_TOKEN from REPLICATE_API_KEY")
    
    # Check if we have a token now
    if not api_token:
        print("ERROR: REPLICATE_API_TOKEN is not set in environment variables")
        print("Please set it directly in your environment or in your .env file")
        return False
    
    # Print masked token for verification
    masked_token = api_token[:4] + "..." + api_token[-4:] if len(api_token) > 8 else "***"
    print(f"Using REPLICATE_API_TOKEN: {masked_token}")
    
    # Try a simple API call using the exact reference implementation
    try:
        print("Attempting to call Replicate API...")
        
        # Use the exact format from the reference implementation
        input = {
            "prompt": "black forest gateau cake spelling out the words \"FLUX SCHNELL\", tasty, food photography, dynamic shot"
        }
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input=input
        )
        
        print("SUCCESS: API call completed successfully")
        print(f"Output: {output}")
        
        # Save the output to a file as in the reference implementation
        for index, item in enumerate(output):
            with open(f"output_{index}.webp", "wb") as file:
                file.write(item.read())
            print(f"output_{index}.webp written to disk")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to call Replicate API: {e}")
        
        # Additional debugging information
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 