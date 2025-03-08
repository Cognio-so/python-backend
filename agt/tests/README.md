# Tests for Cognio Agent

This directory contains test files for the Cognio Agent application.

## Available Tests

### `test_replicate.py`

Tests the Replicate API authentication and connection. This test verifies that:
- The Replicate API key is properly configured in the environment
- The application can successfully authenticate with the Replicate API
- A simple model can be run using the Replicate API

Run this test with:
```bash
python -m tests.test_replicate
```

### `test_image_generation.py`

Tests the image generation functionality in the agent. This test verifies that:
- The image generation function can be called successfully
- The function properly handles the user's prompt
- The function returns a valid response

Run this test with:
```bash
python -m tests.test_image_generation
```

## Running All Tests

To run all tests, you can use:
```bash
python -m unittest discover -s tests
```

Note: Some tests may require API keys to be properly configured in your `.env` file. 