import requests

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI1NDYzYjFlMy04NGJlLTQyYWEtYTI1ZC1kMTg1YjlmNTY0MzMiLCJlbWFpbCI6Im16NzUxQHNjYXJsZXRtYWlsLnJ1dGdlcnMuZWR1IiwiaWF0IjoxNzYzMDgwMTc1fQ.7lMR72IjuQDwE_eQJ9QIqneLcglMoD2AcZlz8HPa1wo"

BASE_URL = "https://api.fipe.online/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}


# ------------------------------
# 1) Get all car brands
# ------------------------------
def getCarBrands() -> str:
    """
    Get all car brands from FIPE API (via fipe.online).
    """
    try:
        url = f"{BASE_URL}/carros/marcas"
        resp = requests.get(url, headers=HEADERS, timeout=10)

        if resp.status_code != 200:
            return f"Error: Could not fetch car brands (Status: {resp.status_code})"

        brands = resp.json()
        if not brands:
            return "No car brands found"

        formatted = "Car Brands Available\n\n"
        for b in brands[:20]:
            formatted += f"- {b['nome']} (Code: {b['codigo']})\n"

        formatted += f"\nTotal brands: {len(brands)}"
        return formatted

    except Exception as e:
        return f"Error: {e}"


# ------------------------------
# 2) Brand + Model family search
# ------------------------------
def searchBrandModelPrice(brand_name: str, model_keyword: str) -> str:
    """
    Search all models under a brand whose names contain model_keyword.
    Return ALL matched models with latest prices.
    """
    try:
        # Step 1: get all brands
        url = f"{BASE_URL}/carros/marcas"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return f"Error: could not fetch brands (status {resp.status_code})"

        brands = resp.json()

        # Step 2: find brand
        brand_lower = brand_name.lower()
        target_brand = None
        for b in brands:
            if brand_lower in b["nome"].lower():
                target_brand = b
                break

        if not target_brand:
            return f"Brand '{brand_name}' not found."

        brand_code = target_brand["codigo"]

        # Step 3: get all models under this brand
        url = f"{BASE_URL}/carros/marcas/{brand_code}/modelos"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        modelos = resp.json().get("modelos", [])
        if not modelos:
            return f"No models found for brand '{target_brand['nome']}'."

        # Step 4: fuzzy match ALL models
        model_kw = model_keyword.lower()
        matched = [m for m in modelos if model_kw in m["nome"].lower()]

        if not matched:
            suggestions = [m["nome"] for m in modelos[:10]]
            return (
                f"No models matching '{model_keyword}' under brand '{target_brand['nome']}'.\n"
                f"Examples: {suggestions}"
            )

        # Step 5: fetch latest price for EACH matched model
        result = f"Matched Models for {target_brand['nome']} — '{model_keyword}'\n\n"

        for idx, chosen in enumerate(matched, start=1):
            model_code = chosen["codigo"]

            # fetch years
            url = f"{BASE_URL}/carros/marcas/{brand_code}/modelos/{model_code}/anos"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            anos = resp.json()

            if not anos:
                result += f"{idx}. {chosen['nome']} — No year data.\n\n"
                continue

            latest_year = anos[0]["codigo"]

            # fetch price
            price_url = f"{BASE_URL}/carros/marcas/{brand_code}/modelos/{model_code}/anos/{latest_year}"
            resp = requests.get(price_url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                result += f"{idx}. {chosen['nome']} — Price unavailable.\n\n"
                continue

            price = resp.json()

            # append formatted result
            result += (
                f"{idx}. {chosen['nome']}\n"
                f"   Year: {price.get('AnoModelo')}\n"
                f"   Fuel: {price.get('Combustivel')}\n"
                f"   Price: {price.get('Valor')}\n"
                f"   FIPE Code: {price.get('CodigoFipe')}\n"
                f"   Reference: {price.get('MesReferencia')}\n\n"
            )

        return result

    except Exception as e:
        return f"Error: {e}"


# =======================================================
# Local test (optional)
# =======================================================
if __name__ == "__main__":
    print("=== Brand List ===")
    print(getCarBrands())

    print("\n=== Example: Brand + Model ===")
    print(searchBrandModelPrice("Toyota", "Corolla Cross"))
