from mcp.server.fastmcp import FastMCP
from app import getCarBrands, searchBrandModelPrice

mcp = FastMCP("car-price-mcp")


@mcp.tool()
async def get_car_brands() -> str:
    """
      Get all available car brands from FIPE via the fipe.online API.
      Args:
        (none)
      Returns:
        result (str): JSON string containing the list of brand identifiers and names.
    """
    return getCarBrands()


@mcp.tool()
async def search_brand_model_price(brand_name: str, model_keyword: str) -> str:
    """
      Search car model prices for a brand using a model name keyword.
      Args:
        brand_name (str): Car brand name to search, such as a manufacturer name.
        model_keyword (str): Substring or keyword of the model name.
      Returns:
        result (str): JSON string listing matching models and their latest FIPE prices.
    """
    if not brand_name or not brand_name.strip():
        return "Please provide brand_name."

    if not model_keyword or not model_keyword.strip():
        return "Please provide model_keyword."

    return searchBrandModelPrice(brand_name.strip(), model_keyword.strip())


if __name__ == "__main__":
    mcp.run(transport="stdio")
