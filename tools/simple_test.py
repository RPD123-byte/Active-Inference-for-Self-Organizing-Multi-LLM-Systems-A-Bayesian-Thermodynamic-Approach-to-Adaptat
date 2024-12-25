import json
from tools.basic_scraper import scrape_website

# Mock the HumanMessage class for testing
class HumanMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

def test_single_url(url: str):
    # Initialize state
    state = {"scraper_response": []}

    # Create mock research function
    def mock_research():
        return type('obj', (object,), {
            'content': json.dumps({"selected_page_url": url})
        })()

    print(f"Testing URL: {url}")
    print("="*80)

    try:
        # Run the scraper
        result = scrape_website(state, mock_research)

        if not state["scraper_response"]:
            print("No response received from the scraper.")
            return

        # Get the response
        last_response = state["scraper_response"][-1].content

        # Safely parse the JSON content
        response = json.loads(last_response)

        print("\nResponse:")
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Test with your HTML URL
    test_single_url("https://medium.com/@a1guy/prompt-engineering-via-prompt-patterns-fact-check-list-pattern-9fb5f3f76e5a")
