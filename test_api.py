#!/usr/bin/env python3
"""
Test script for byov-tts-server
Assumes the server is running on localhost:7861
"""

import json
import os
from pathlib import Path

import requests

# Configuration
# Server must be running already!
API_BASE_URL = "http://localhost:7861"
TEST_DATA_DIR = Path("test_data/test_api")


def setup_test_directories():
    """Create test data directory structure"""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Test directory created: {TEST_DATA_DIR}")


def test_health_check():
    """Test the /health endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        
        assert data["status"] == "ok", "Health check failed"
        assert data["model_loaded"] == True, "Model not loaded"
        
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_list_voices():
    """Test the /voices endpoint"""
    print("\n" + "="*60)
    print("TEST: List Voices")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/voices")
        response.raise_for_status()
        
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        
        assert "voices" in data, "No voices key in response"
        print(f"✓ Found {len(data['voices'])} voice(s)")
        
        return data
    except Exception as e:
        print(f"✗ List voices failed: {e}")
        return None


def test_generate_basic():
    """Test basic speech generation"""
    print("\n" + "="*60)
    print("TEST: Basic Generation")
    print("="*60)
    
    request_data = {
        "voice_id": "test_voice",
        "variation": "variation",
        "text": "Hello world! This is a test of the text to speech system."
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )
        response.raise_for_status()
        
        # Save the audio file
        output_file = TEST_DATA_DIR / "basic_generation.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Audio saved to: {output_file}")
        print(f"File size: {file_size:,} bytes")
        
        assert file_size > 0, "Generated audio file is empty"
        print("✓ Basic generation passed")
        return True
    except Exception as e:
        print(f"✗ Basic generation failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response: {e.response.text}")
        return False


def test_generate_default_variation():
    """Test generation with default variation (using voice_id)"""
    print("\n" + "="*60)
    print("TEST: Default Variation")
    print("="*60)
    
    request_data = {
        "voice_id": "test_voice",
        "text": "This test uses the default voice variation."
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )
        response.raise_for_status()
        
        output_file = TEST_DATA_DIR / "default_variation.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"Status Code: {response.status_code}")
        print(f"Audio saved to: {output_file}")
        print(f"File size: {file_size:,} bytes")
        
        print("✓ Default variation test passed")
        return True
    except Exception as e:
        print(f"✗ Default variation test failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response: {e.response.text}")
        return False


def test_generate_with_speed():
    """Test generation with different speed settings"""
    print("\n" + "="*60)
    print("TEST: Speed Variations")
    print("="*60)
    
    speeds = [0.5, 1.0, 1.5]
    
    for speed in speeds:
        request_data = {
            "voice_id": "test_voice",
            "variation": "variation",
            "text": "Testing different speech speeds.",
            "speed": speed
        }
        
        print(f"\nTesting speed: {speed}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json=request_data
            )
            response.raise_for_status()
            
            output_file = TEST_DATA_DIR / f"speed_{speed}.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_file)
            print(f"  Audio saved to: {output_file}")
            print(f"  File size: {file_size:,} bytes")
            print(f"  ✓ Speed {speed} passed")
            
        except Exception as e:
            print(f"  ✗ Speed {speed} failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Error response: {e.response.text}")
            return False
    
    print("\n✓ All speed variations passed")
    return True


def test_generate_long_text():
    """Test generation with longer text"""
    print("\n" + "="*60)
    print("TEST: Long Text Generation")
    print("="*60)
    
    long_text = """
    The mountain stood tall against the evening sky, its peaks catching 
    the last rays of sunlight. Down in the valley below, a small village 
    began to light its lanterns one by one. The traveler paused on the path 
    to take in the scene, knowing that tomorrow would bring new adventures. 
    But for now, the peaceful moment was enough, a reminder that sometimes 
    the journey itself is more valuable than the destination.
    """
    
    request_data = {
        "voice_id": "test_voice",
        "variation": "variation",
        "text": long_text.strip()
    }
    
    print(f"Text length: {len(request_data['text'])} characters")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )
        response.raise_for_status()
        
        output_file = TEST_DATA_DIR / "long_text_generation.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"Status Code: {response.status_code}")
        print(f"Audio saved to: {output_file}")
        print(f"File size: {file_size:,} bytes")
        
        print("✓ Long text generation passed")
        return True
    except Exception as e:
        print(f"✗ Long text generation failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response: {e.response.text}")
        return False


def test_generate_narrative():
    """Test generation with narrative text"""
    print("\n" + "="*60)
    print("TEST: Narrative Text Generation")
    print("="*60)
    
    narrative_text = """The research team discovered something remarkable during their latest expedition. As they analyzed the data from the previous month, patterns began to emerge that no one had anticipated. The lead scientist called a meeting to discuss the findings. Everyone agreed that this discovery would change how we understand the subject, opening up entirely new avenues for future exploration and collaboration."""
    
    request_data = {
        "voice_id": "test_voice",
        "variation": "variation",
        "text": narrative_text
    }
    
    print(f"Text length: {len(request_data['text'])} characters")
    print(f"Text preview: {request_data['text'][:100]}...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )
        response.raise_for_status()
        
        output_file = TEST_DATA_DIR / "narrative_text.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"Status Code: {response.status_code}")
        print(f"Audio saved to: {output_file}")
        print(f"File size: {file_size:,} bytes")
        
        print("✓ Narrative text generation passed")
        return True
    except Exception as e:
        print(f"✗ Narrative text generation failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response: {e.response.text}")
        return False


def test_error_invalid_voice():
    """Test error handling for invalid voice_id"""
    print("\n" + "="*60)
    print("TEST: Error Handling - Invalid Voice")
    print("="*60)
    
    request_data = {
        "voice_id": "nonexistent_voice",
        "text": "This should fail"
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 404:
            error_data = response.json()
            print(f"Error Response: {json.dumps(error_data, indent=2)}")
            print("✓ Error handling passed (404 as expected)")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error test failed unexpectedly: {e}")
        return False


def test_error_empty_text():
    """Test error handling for empty text"""
    print("\n" + "="*60)
    print("TEST: Error Handling - Empty Text")
    print("="*60)
    
    request_data = {
        "voice_id": "test_voice",
        "text": ""
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=request_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"Error Response: {json.dumps(error_data, indent=2)}")
            print("✓ Error handling passed (400 as expected)")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error test failed unexpectedly: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("byov-tts-server Test Suite")
    print("="*60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Data Directory: {TEST_DATA_DIR}")
    
    # Setup
    setup_test_directories()
    
    # Run tests
    results = {
        "Health Check": test_health_check(),
        "List Voices": test_list_voices() is not None,
        "Basic Generation": test_generate_basic(),
        "Default Variation": test_generate_default_variation(),
        "Speed Variations": test_generate_with_speed(),
        "Long Text Generation": test_generate_long_text(),
        "Narrative Text": test_generate_narrative(),
        "Error - Invalid Voice": test_error_invalid_voice(),
        "Error - Empty Text": test_error_empty_text(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

