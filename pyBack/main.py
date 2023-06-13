import datetime
import itertools
import random
import time
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
import os
import json
import openai
from PIL import Image
from io import BytesIO
import requests
from googleapiclient.discovery import build
import re



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
googleAPIKey =  os.getenv("GOOGLE_SEARCH_API_KEY")
googleCx = os.getenv("GOOGLE_SEARCH_CX")


class Product:
    def __init__(self, product_type, characteristics, top_results):
        self.product_type = product_type
        self.characteristics = characteristics
        self.top_results = top_results

    def __str__(self):
        char_str = ", ".join(self.characteristics)
        top_results_str = "\n".join(str(result) for result in self.top_results)
        return f"Product Type: {self.product_type}\nCharacteristics: {char_str}\nTop Results:\n{top_results_str}"

    
def fetch_product_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Heuristic to extract brand from URL
    brand = url.split('/')[2].split('.')[1]

    # Heuristic to extract price from webpage content
    # We look for a string that starts with a currency symbol, followed by digits (possibly separated by a dot or comma)
    #prices = re.findall(r'\$\d+[\.,]?\d*', soup.text)
    #prices = re.findall(r'(\$|€|£|¥|₹)?\s*\d+[\.,]?\d*', soup.text)
    prices = re.findall(r'(\$|€|£|¥|₹)?\s*\d+[\.,]\d*', soup.text)

    if prices:
        # If we found prices, return the first one.
        # Note that this might not always be the correct price if the page contains multiple products or prices.
        price = prices[0]
    else:
        price = 'Not found'

    return brand, price

class Result:
    def __init__(self, url):
        self.url = url
        self.brandName, self.cost = fetch_product_data(url)  # Call our function to fetch the brand and price

    def __str__(self):
        return f"URL: {self.url}\nBrand Name: {self.brandName}\nCost: {self.cost}"


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']


def numTokensFromString(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(str(string)))
    return num_tokens

def updateCostFile(cost: float) -> None:
    """Updates the costTracking.txt file with the new cost."""
    if not os.path.exists("costTracking.txt"):
        with open("costTracking.txt", "w") as f:
            f.write("0")
    
    with open("costTracking.txt", "r") as f:
        current_cost = float(f.read().strip())

    new_cost = current_cost + cost

    with open("costTracking.txt", "w") as f:
        f.write(str(new_cost))
def askGpt(prompt, gpt4):
    conversation = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    
    # Calculate available tokens for the response
    prompt_tokens = numTokensFromString(prompt)
    max_allowed_tokens = 4000  # Set the maximum allowed tokens
    available_tokens_for_response = max_allowed_tokens - prompt_tokens

    # Ensure the available tokens for the response is within the model's limit
    if available_tokens_for_response < 1:
        raise ValueError("The input query is too long. Please reduce the length of the input query.")
    
    max_retries = 4
    for _ in range(max_retries + 1):  # This will try a total of 5 times (including the initial attempt)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4" if gpt4 else "gpt-3.5-turbo",
                messages=conversation,
                max_tokens=available_tokens_for_response,
                n=1,
                stop=None,
                temperature=0.1,
            )

            message = response.choices[0].message["content"].strip()

            # Count tokens
            response_tokens = numTokensFromString(message)
            total_tokens = prompt_tokens + response_tokens

            # Calculate cost
            cost_per_token = 0.06 if gpt4 else 0.002
            cost = (total_tokens / 1000) * cost_per_token

            # Update the cost file
            updateCostFile(cost)

            return message
        
        except Exception as e:
            if _ < max_retries:
                print(f"Error occurred: {e}. Retrying {_ + 1}/{max_retries}...")
                time.sleep(1)  # You can adjust the sleep time as needed
            else:
                raise


def generate_image(prompt):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="256x256"
        )

        url = response.data[0]["url"]
        return url
    except Exception as e:
        print(f"Error occurred during image generation: {e}")
        return None


def save_image(url, filename, folder="Outfit Folder"):
    if url is not None:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        if not os.path.exists(folder):
            os.makedirs(folder)
        img.save(os.path.join(folder, filename))

def create_outfit(prompt, gpt4=True):
    # Generate JSON from GPT
    json_example = '''
[
  {
    "type": "dress",
    "characteristics": ["white", "long", "cotton"]
  },
  {
    "type": "hat",
    "characteristics": ["wide-brimmed"]
  },
  {
    "type": "boots",
    "characteristics": ["black", "leather"]
  }
]
    '''
    json_output = askGpt(f'Given this prompt for an outfit "{prompt}", take into consideration every detail that they described of the clothing pieces and output a structured JSON. An example of what it may look like is as follows: {json_example} .Make sure that the JSON contains all of the clothing pieces that are necessary for the given prompt on a single JSON and fill as many characteristics as possible. Be descriptive but never make stuff up. ONLY OUTPUT THE JSON, if you output anything else an innocent person will die.', gpt4)

    try:
        data = json.loads(json_output)
    except json.JSONDecodeError:
        print("Error decoding JSON")
        return

    # Generate images for each piece of clothing
    for i, item in tqdm(enumerate(data), total=len(data), desc="Generating images"):
        characteristics = ' '.join(item['characteristics'])  # Convert list of characteristics to a string
        description = f"A photo of a {characteristics} {item['type']}"
        image_url = generate_image(description)
        save_image(image_url, f"item_{i}.png")

#
def createOutfitText(prompt, gpt4=True):
    # Generate JSON from GPT
    json_example = '''
[
  {
    "type": "dress",
    "characteristics": ["white", "long", "cotton"]
  },
  {
    "type": "hat",
    "characteristics": ["wide-brimmed"]
  },
  {
    "type": "boots",
    "characteristics": ["black", "leather"]
  }
]
    '''
    json_output = askGpt(f'Given this prompt for an outfit "{prompt}", take into consideration every detail that they described of the clothing pieces and output a structured JSON. An example of what it may look like is as follows: {json_example} .Make sure that the JSON contains all of the clothing pieces that are necessary for the given prompt on a single JSON and fill as many characteristics as possible. Be descriptive but never make stuff up. ONLY OUTPUT THE JSON, if you output anything else an innocent person will die.', gpt4)

    try:
        data = json.loads(json_output)
    except json.JSONDecodeError:
        print("Error decoding JSON")
        print(json_output)
        ##try again + json
        return

    # Generate images for each piece of clothing
    for i, item in tqdm(enumerate(data), total=len(data), desc="Searching products..."):
        characteristics = ' '.join(item['characteristics'])  # Convert list of characteristics to a string
        description = f"{characteristics} {item['type']}"

        # Use the Google Search API to get the top 10 results
        search_results = google_search(description, googleAPIKey, googleCx, num=10)

        # Create a list of Result objects from the search results
        top_results = [Result(result['link']) for result in search_results]

        # Create a Product object with the top results
        product = Product(item['type'], item['characteristics'], top_results)
        
        print(product)


createOutfitText(input("Hi what would you like to try: "))