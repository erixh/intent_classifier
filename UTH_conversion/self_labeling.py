import sqlite3
import requests
import json
import time
from tqdm import tqdm


def build_prompt(domain, action_text):
    return f"what is the user's likely intent if they are on {domain} and click {action_text}?"


def label_with_mistral_api(prompt, api_url="http://localhost:8000/classify", max_retries=3, timeout=30):
    """
    Send prompt to FastAPI server running Mistral model for intent classification
    
    Args:
        prompt (str): The prompt to classify
        api_url (str): FastAPI endpoint URL
        max_retries (int): Number of retry attempts
        timeout (int): Request timeout in seconds
        
    Returns:
        str: Classified intent or fallback
    """
    
    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.2,  # Low temperature for consistent classification
        "system_prompt": """You are an intent classifier for web interactions. 
        Given a user action on a website, classify the intent that you think the user is trying to achieve.
        Respond with the intent category, and a confidence score between 0 and 1."""
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            # Make the API request
            response = requests.post(
                api_url, 
                data=json.dumps(payload), 
                headers=headers, 
                timeout=timeout
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Extract the intent from response
                if "intent" in result:
                    return result["intent"].strip().lower()
                elif "response" in result:
                    return result["response"].strip().lower()
                elif "text" in result:
                    return result["text"].strip().lower()
                else:
                    # If response format is unexpected, try to extract first word
                    text = str(result).strip().lower()
                    return text.split()[0] if text else "navigate"
                    
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Request timeout on attempt {attempt + 1}/{max_retries}")
            
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}/{max_retries}. Is the FastAPI server running?")
            
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error on attempt {attempt + 1}/{max_retries}: {e}")
            
        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # Fallback to simple rule-based classification if API fails
    print(f"API failed after {max_retries} attempts. Using fallback classification.")
    return fallback_classification(prompt)


def fallback_classification(prompt):
    """Print error when API is unavailable instead of generating noisy rule-based data"""
    print(f"ERROR: API classification failed for prompt: '{prompt[:50]}...'")
    print("WARNING: Skipping classification to avoid noisy data. Check your FastAPI server.")
    return None  # Return None to indicate failed classification


def pseudo_label_all(db_path="intents.db", api_url="http://localhost:8000/classify"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, domain, action_text FROM intents WHERE inferred_intent IS NULL
    """).fetchall()

    print(f"Labeling {len(rows)} intents using Mistral API at {api_url}")

    for row in tqdm(rows, desc="Labeling intents"):
        id, domain, action = row
        prompt = build_prompt(domain, action)
        inferred_intent = label_with_mistral_api(prompt, api_url)

        # Only update database if classification was successful
        if inferred_intent is not None:
            cur.execute("""
                UPDATE intents SET inferred_intent = ?, label_source = ? WHERE id = ?
            """, (inferred_intent, "mistral_api", id))
        else:
            print(f"Skipping database update for failed classification (ID: {id})")

    conn.commit()
    conn.close()


# For testing the API connection
def test_api_connection(api_url="http://localhost:8000/classify"):
    """Test if the FastAPI server is responding"""
    test_prompt = "what is the user's likely intent if they are on amazon.com and click Add to Cart?"
    
    try:
        result = label_with_mistral_api(test_prompt, api_url)
        print(f"API test successful. Response: {result}")
        return True
    except Exception as e:
        print(f"API test failed: {e}")
        return False

