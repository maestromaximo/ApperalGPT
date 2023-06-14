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
from flask import Flask, request, jsonify
from flask_cors import CORS


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
googleAPIKey =  os.getenv("GOOGLE_SEARCH_API_KEY")
googleCx = os.getenv("GOOGLE_SEARCH_CX")




app = Flask(__name__)
CORS(app)



class Product:
    def __init__(self, product_type, characteristics, top_results):
        self.product_type = product_type
        self.characteristics = characteristics
        self.top_results = top_results

    def __str__(self):
        char_str = ", ".join(self.characteristics)
        top_results_str = "\n".join(str(result) for result in self.top_results)
        return f"Product Type: {self.product_type}\nCharacteristics: {char_str}\nTop Results:\n{top_results_str}"

    def to_dict(self):
        return {
            "product_type": self.product_type,
            "characteristics": self.characteristics,
            "top_results": [result.to_dict() for result in self.top_results],  # assuming Result also has a to_dict() method
        }


    
def fetch_product_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Heuristic to extract brand from URL
    brand = url.split('/')[2].split('.')[1]

    # Heuristic to extract price from webpage content
    prices = re.findall(r'\$\d+[\.,]?\d*', soup.text)

    if prices:
        price = prices[0]
    else:
        price = 'Not found'

    # Try to find the first image on the page
    image_tag = soup.find('img')
    if image_tag and 'src' in image_tag.attrs:
        image_url = image_tag['src']
        # If the URL is relative, add the base URL
        if not image_url.startswith('http'):
            base_url = '/'.join(url.split('/')[:3])
            image_url = base_url + image_url
    else:
        image_url = "No image avalible"

    return brand, price, image_url


class Result:
    def __init__(self, url):
        self.url = url
        self.brandName, self.cost, self.image_url = fetch_product_data(url)

    def __str__(self):
        return f"URL: {self.url}\nBrand Name: {self.brandName}\nCost: {self.cost}\nImage URL: {self.image_url}"

    def to_dict(self):
        return {
            "url": self.url,
            "brandName": self.brandName,
            "cost": self.cost,
            "image_url": self.image_url,
        }




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
        # Return a dictionary with an error message
        return {"error": "Error decoding JSON"}

    # Generate images for each piece of clothing
    for i, item in tqdm(enumerate(data), total=len(data), desc="Generating images"):
        characteristics = ' '.join(item['characteristics'])  # Convert list of characteristics to a string
        description = f"A photo of a {characteristics} {item['type']}"
        image_url = generate_image(description)
        save_image(image_url, f"item_{i}.png")

#
def createOutfitText(prompt, gpt4=True):
    # Generate JSON from GPT
    products = []
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
        products.append(product)
        #print(product)
    return products

# @app.route('/api/prompt', methods=['POST'])
# def prompt():
#     data = request.get_json()
#     prompt_text = data['prompt']
#     # Call your function with the prompt_text
#     result = createOutfitText(prompt_text)  # Assuming createOutfitText() returns the result
#     return jsonify(result=result)

@app.route('/api/prompt', methods=['POST'])
def prompt():
    data = request.get_json()
    prompt_text = data['prompt']
    # Call your function with the prompt_text
    result = createOutfitText(prompt_text)  # Assuming createOutfitText() returns the result

    # Check if the result contains an error message
    if "error" in result:
        # If there's an error, return a response with a 400 status code
        return jsonify(result=[product.to_dict() for product in result]), 400
    else:
        # If there's no error, return the result as usual
        return jsonify(result=[product.to_dict() for product in result])

if __name__ == '__main__':
    app.run(port=5000)