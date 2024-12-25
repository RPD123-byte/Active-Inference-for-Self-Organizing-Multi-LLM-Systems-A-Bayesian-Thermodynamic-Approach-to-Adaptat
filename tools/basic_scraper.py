import json
from typing import Dict, Optional, TypedDict
from urllib.parse import urlparse
import os
import requests
from io import BytesIO
import pdfplumber

from scrapegraphai.graphs import SmartScraperGraph
from langchain_core.messages import HumanMessage

from models.openai_models import get_open_ai, get_open_ai_json
from utils.helper_functions import load_config

# Load configuration from config.yaml
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
load_config(config_path)

class ScrapedResult(TypedDict):
    url: str
    title: Optional[str]
    content: str
    metadata: Dict
    error: Optional[str]

def format_response(result: ScrapedResult) -> Dict:
    """Format the scraping result into the expected response structure"""
    return {
        "source": result["url"],
        "content": result["error"] if result["error"] else result["content"],
        "title": result["title"],
        "metadata": result["metadata"]
    }

def extract_text_from_pdf(url: str) -> str:
    """Extract text from a PDF URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

def scrape_website(state: Dict, research: callable = None) -> Dict:
    """
    Main scraping function that handles the complete scraping process using scrapegraphai.

    Args:
        state (Dict): The AgentGraphState containing the scraper_response list.
        research (callable, optional): A callable that returns an object with a .content attribute containing JSON.

    Returns:
        Dict: Updated state with the scraper_response appended.
    """
    try:
        # Instantiate the language model using your existing model function
        llm = get_open_ai(temperature=0, model='gpt-4o')  # Ensure 'gpt-4o' is correct

        # Parse research data to get the URL
        if research:
            research_data = json.loads(research().content)
            url = research_data.get("selected_page_url") or research_data.get("error")
            if not url:
                raise ValueError("No URL found in research data.")
        else:
            raise ValueError("Research callable is None.")

        # Define the prompt to extract full content
        prompt = "Extract the full text content from the following website."

        # Define the configuration for the scraping pipeline
        graph_config = {
            "llm": {
                "api_key": os.getenv('OPENAI_API_KEY'),  # Ensure API key is set
                "model": "gpt-4o",
            },
            "verbose": True,
            "headless": False,  # Headless mode as per your requirement
        }

        # Create the SmartScraperGraph instance
        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=graph_config
        )

        # Determine if URL is a PDF
        parsed_url = urlparse(url)
        if parsed_url.path.endswith('.pdf'):
            # Extract text from PDF
            content = extract_text_from_pdf(url)
            scraped_result: ScrapedResult = {
                "url": url,
                "title": None,  # PDFs might not have a title
                "content": content[:4000] if content else "",
                "metadata": {},
                "error": None
            }
        else:
            # Existing scraping logic for HTML pages
            scrape_result = smart_scraper_graph.run()

            if isinstance(scrape_result, dict) and ('content' in scrape_result or 'full_text_content' in scrape_result):
                # Extract content based on available keys
                content = scrape_result.get("content") or scrape_result.get("full_text_content", "")
                title = scrape_result.get("title", None)
                scraped_url = scrape_result.get("url", url)

                scraped_result: ScrapedResult = {
                    "url": scraped_url,
                    "title": title,
                    "content": content[:4000] if content else "",
                    "metadata": scrape_result,
                    "error": None
                }
            else:
                print(f"Scrape Result: {scrape_result}")  # Debugging
                scraped_result = ScrapedResult({
                    "url": url,
                    "title": None,
                    "content": "",
                    "metadata": {},
                    "error": "Unknown scraping error."
                })

        # Format the response
        response_content = format_response(scraped_result)

        # Append the response to the state
        state["scraper_response"].append(
            HumanMessage(role="system", content=json.dumps(response_content))
        )

    except Exception as e:
        # Handle any unexpected errors
        error_content = {
            "source": url if 'url' in locals() else "unknown",
            "content": f"Error scraping website: {str(e)}",
            "title": None,
            "metadata": {}
        }
        state["scraper_response"].append(
            HumanMessage(role="system", content=json.dumps(error_content))
        )

    finally:
        return {"scraper_response": state["scraper_response"]}

