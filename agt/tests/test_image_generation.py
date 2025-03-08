import asyncio
import os
import sys
import unittest
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after adding parent to path
from agent import image_generator_node, VaaniState

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestImageGeneration(unittest.TestCase):
    """Test the image generation functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()
        
        # Check if REPLICATE_API_TOKEN is set
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            # Try to set it from REPLICATE_API_KEY
            api_key = os.getenv("REPLICATE_API_KEY")
            if api_key:
                os.environ["REPLICATE_API_TOKEN"] = api_key
                self.api_token = api_key
    
    async def async_test_image_generation(self):
        """Async test for image generation."""
        logger.info("Testing image generation function...")
        
        # Check if REPLICATE_API_TOKEN is set
        self.assertIsNotNone(self.api_token, "REPLICATE_API_TOKEN is not set in environment variables")
        
        logger.info(f"REPLICATE_API_TOKEN is set (first 4 chars: {self.api_token[:4]}...)")
        
        # Create a test state with a simple prompt
        test_prompt = "Generate an image of a sunset over mountains"
        logger.info(f"Test prompt: {test_prompt}")
        
        test_state = VaaniState(
            messages=[HumanMessage(content=test_prompt)]
        )
        
        # Call the image generator function
        logger.info("Calling image_generator_node...")
        try:
            result = await image_generator_node(test_state)
            
            # Print the result
            logger.info("\nResult:")
            logger.info(result["messages"][0].content)
            
            # Check that the result doesn't contain an error
            self.assertNotIn("Error", result["messages"][0].content, 
                            "Image generation returned an error")
            
            return result
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Image generation failed with exception: {e}")
    
    def test_image_generation(self):
        """Test wrapper for the async test."""
        result = asyncio.run(self.async_test_image_generation())
        self.assertIsNotNone(result, "Image generation returned None")

async def test_image_generation_function():
    """Legacy function for backward compatibility."""
    logger.info("Testing image generation function...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if REPLICATE_API_TOKEN is set
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        logger.error("REPLICATE_API_TOKEN is not set in environment variables")
        return False
    
    logger.info(f"REPLICATE_API_TOKEN is set (first 4 chars: {api_token[:4]}...)")
    
    # Create a test state with a simple prompt
    test_prompt = "Generate an image of a sunset over mountains"
    logger.info(f"Test prompt: {test_prompt}")
    
    test_state = VaaniState(
        messages=[HumanMessage(content=test_prompt)]
    )
    
    # Call the image generator function
    logger.info("Calling image_generator_node...")
    try:
        result = await image_generator_node(test_state)
        
        # Print the result
        logger.info("\nResult:")
        logger.info(result["messages"][0].content)
        
        return "Error" not in result["messages"][0].content
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # For backward compatibility, run the function if called directly
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy":
        success = asyncio.run(test_image_generation_function())
        logger.info(f"\nTest {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)
    else:
        # Otherwise, run the unittest
        unittest.main() 