# servers/food_nutrition_mcp.py
import os
import requests
from typing import Optional
from mcp.server.fastmcp import FastMCP

server = FastMCP("food-nutrition")

# ==== Edamam API credentials ====
EDAMAM_APP_ID = os.getenv("EDAMAM_APP_ID")
EDAMAM_APP_KEY = os.getenv("EDAMAM_APP_KEY")

if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
    print("[food_nutrition_mcp] WARNING: Missing Edamam credentials. Set EDAMAM_APP_ID and EDAMAM_APP_KEY.")

BASE_URL = "https://api.edamam.com"


# ==== Tool: get nutrition info from food string ====
@server.tool()
def get_nutrition(
    query: Optional[str] = None,
    food: Optional[str] = None,
) -> str:
    """
    Enter a food description (e.g., '2 slices of bread' or '1 cup of rice') and get full nutritional information.

    Args:
        query (Optional[str]): Food name + quantity, e.g., "1 apple".
        food (Optional[str]): Alias for query.

    Returns:
        str: Key nutrients and their units. Returns an error message if retrieval fails.
    """
    if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
        return "Edamam API credentials not configured."

    q = query or food
    if not q:
        return "Missing input query. Try something like '1 banana' or '2 eggs'."

    # Step 1: Call /parser
    parser_url = f"{BASE_URL}/api/food-database/v2/parser"
    parser_params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "ingr": q,
    }
    try:
        parser_resp = requests.get(parser_url, params=parser_params, timeout=15)
        parser_json = parser_resp.json()
    except Exception as e:
        return f"Error calling parser API: {e}"

    parsed = parser_json.get("parsed")
    if not parsed:
        return "Could not parse the input. Try using simpler food names."

    item = parsed[0]
    food_id = item["food"]["foodId"]
    measure_uri = item["measure"]["uri"]
    quantity = item.get("quantity", 1)

    # Step 2: Call /nutrients
    nutrients_url = f"{BASE_URL}/api/food-database/v2/nutrients"
    payload = {
        "ingredients": [
            {
                "quantity": quantity,
                "measureURI": measure_uri,
                "foodId": food_id
            }
        ]
    }
    try:
        nutrient_resp = requests.post(
            nutrients_url,
            params={"app_id": EDAMAM_APP_ID, "app_key": EDAMAM_APP_KEY},
            json=payload,
            timeout=15
        )
        nutrient_json = nutrient_resp.json()
    except Exception as e:
        return f"Error calling nutrients API: {e}"

    total_nutrients = nutrient_json.get("totalNutrients") or {}
    if not total_nutrients:
        return "No nutrients found."

    # Select commonly used nutrients to display
    important_keys = [
        "ENERC_KCAL", "PROCNT", "FAT", "CHOCDF", "FIBTG", "SUGAR", "NA", "CA", "FE", "VITC"
    ]
    lines = [f"Nutrition for: {q}"]
    for k in important_keys:
        n = total_nutrients.get(k)
        if n:
            lines.append(f"- {n['label']}: {n['quantity']:.2f} {n['unit']}")

    return "\n".join(lines)


if __name__ == "__main__":
    server.run(transport="stdio")
