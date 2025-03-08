import os
import sys
import unittest
import replicate
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestReplicateAuth(unittest.TestCase):
    """Test Replicate API authentication."""
    
    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()
        
        # Check if REPLICATE_API_KEY exists
        self.api_key = os.getenv("REPLICATE_API_KEY")
        
        # Set the API token explicitly
        if self.api_key:
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
    
    def test_replicate_auth(self):
        """Test Replicate API authentication and connection."""
        print("Testing Replicate API authentication...")
        
        # Check if REPLICATE_API_KEY exists
        self.assertIsNotNone(self.api_key, "REPLICATE_API_KEY not found in environment variables")
        
        # Print masked API key for verification
        masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
        print(f"Found API key: {masked_key}")
        
        # Try a simple API call
        try:
            print("Attempting to call Replicate API...")
            # Use a very simple model for testing
            output = replicate.run(
                "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                input={"prompt": "a photo of a test image"}
            )
            print("SUCCESS: API call completed successfully")
            print(f"Output: {output}")
            self.assertIsNotNone(output, "Replicate API returned None")
        except Exception as e:
            print(f"ERROR: Failed to call Replicate API: {e}")
            import traceback
            print("\nDetailed error information:")
            print(traceback.format_exc())
            self.fail(f"Replicate API call failed: {e}")

def test_replicate_auth_function():
    """Legacy function for backward compatibility."""
    print("Testing Replicate API authentication...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if REPLICATE_API_KEY exists
    api_key = os.getenv("REPLICATE_API_KEY")
    if not api_key:
        print("ERROR: REPLICATE_API_KEY not found in environment variables")
        print("Please make sure you have a .env file with REPLICATE_API_KEY set")
        return False
    
    # Print masked API key for verification
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"Found API key: {masked_key}")
    
    # Set the API token explicitly
    os.environ["REPLICATE_API_TOKEN"] = api_key
    print(f"Set REPLICATE_API_TOKEN environment variable")
    
    # Try a simple API call
    try:
        print("Attempting to call Replicate API...")
        # Use a very simple model for testing
        output = replicate.run(
            "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
            input={"prompt": "a photo of a test image"}
        )
        print("SUCCESS: API call completed successfully")
        print(f"Output: {output}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to call Replicate API: {e}")
        print("Please check your API key and make sure it's valid")
        
        # Additional debugging information
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())
        
        return False

if __name__ == "__main__":
    # For backward compatibility, run the function if called directly
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy":
        success = test_replicate_auth_function()
        sys.exit(0 if success else 1)
    else:
        # Otherwise, run the unittest
        unittest.main() 