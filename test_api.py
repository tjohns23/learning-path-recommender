"""
Example script demonstrating API usage
This script shows how to call the API endpoints
"""

import requests
import json
from typing import Optional


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_response(response: requests.Response, title: str = "Response"):
    """Pretty print API response."""
    print(f"\n{title}:")
    print(f"  Status Code: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"  Body:")
            print(json.dumps(data, indent=2))
        except:
            print(f"  Body: {response.text}")
    else:
        print(f"  Error: {response.text}")


def main():
    """Test API endpoints."""
    
    BASE_URL = "http://localhost:8000"
    
    print_header("LEARNING PATH RECOMMENDER API - TEST EXAMPLES")
    
    # Test 1: Health check
    print_header("Test 1: Health Check")
    print("Endpoint: GET /health")
    print("Purpose: Check if API is running and models are loaded")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response, "Health Check Response")
    except requests.exceptions.ConnectionError:
        print("\n ERROR: Could not connect to API at http://localhost:8000")
        print("Make sure the API is running with:")
        print("  python -m uvicorn src.api:app --reload")
        return
    
    # Test 2: Get recommendations for user 0
    print_header("Test 2: Get Recommendations for User 0")
    print("Endpoint: GET /recommend/{user_id}")
    print("Purpose: Get top-K recommended learning items for a specific user")
    
    try:
        response = requests.get(f"{BASE_URL}/recommend/0")
        print_response(response, "Recommendations for User 0")
    except Exception as e:
        print(f"\n ERROR: {e}")
    
    # Test 3: Get recommendations for user 25 with custom top_k
    print_header("Test 3: Get Recommendations for User 25 (Custom Top-K=3)")
    print("Endpoint: GET /recommend/{user_id}?top_k={k}")
    print("Purpose: Get custom number of recommendations")
    
    try:
        response = requests.get(f"{BASE_URL}/recommend/25", params={"top_k": 3})
        print_response(response, "Recommendations for User 25 (top_k=3)")
    except Exception as e:
        print(f"\n ERROR: {e}")
    
    # Test 4: Get recommendations for user 49
    print_header("Test 4: Get Recommendations for User 49")
    print("Endpoint: GET /recommend/{user_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/recommend/49")
        print_response(response, "Recommendations for User 49")
    except Exception as e:
        print(f"\n ERROR: {e}")
    
    # Test 5: Get recommendations for non-existent user
    print_header("Test 5: Error Handling - Non-existent User")
    print("Endpoint: GET /recommend/{user_id}")
    print("Expected: 404 Not Found")
    
    try:
        response = requests.get(f"{BASE_URL}/recommend/999")
        print_response(response, "Error Response (Expected 404)")
    except Exception as e:
        print(f"\n ERROR: {e}")
    
    # Test 6: Multiple users batch test
    print_header("Test 6: Batch Test - Multiple Users")
    print("Getting recommendations for users 0-9")
    
    try:
        for user_id in range(10):
            response = requests.get(f"{BASE_URL}/recommend/{user_id}", params={"top_k": 3})
            if response.status_code == 200:
                data = response.json()
                num_recs = len(data.get("recommendations", []))
                avg_score = data.get("average_relevance_score", 0)
                print(f"  User {user_id:2d}: {num_recs} recommendations, avg score: {avg_score:.3f}")
            else:
                print(f"  User {user_id:2d}: ERROR {response.status_code}")
    except Exception as e:
        print(f"\n ERROR: {e}")
    
    print_header("API TEST COMPLETE")
    print("\nAPI Documentation (Swagger UI):")
    print(f"  {BASE_URL}/docs")
    print("\nAlternative API Documentation (ReDoc):")
    print(f"  {BASE_URL}/redoc")
    print("=" * 80)


if __name__ == "__main__":
    main()
