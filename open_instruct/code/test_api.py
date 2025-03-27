import requests
import pytest

def test_add_function():
    # API endpoint
    url = "http://phobos-cs-aus-453.reviz.ai2.in:8000/test_program"

    # Test data
    payload = {
        "program": """
def add(a, b):
    return a + b
""",
        "tests": [
            "assert True",
            "assert add(1, 2) == 3",
            "assert add(-1, 1) == 0",
            "assert add(0, 0) == 1"  # This test will fail
        ],
        "max_execution_time": 1.0
    }

    # Send POST request
    response = requests.post(url, json=payload)
    response_json = response.json()
    
    # Print results
    print("Status Code:", response.status_code)
    print("Response:", response_json) 

    assert response_json['results'] == [1, 1, 1, 0]

def test_health_check():
    url = "http://phobos-cs-aus-453.reviz.ai2.in:8000/health"
    response = requests.get(url)
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

if __name__ == "__main__":
    test_add_function()
    test_health_check() 