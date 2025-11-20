# Car Price MCP Server

A Model Context Protocol (MCP) server that provides Brazilian vehicle price information using FIPE API.

## Features

- Get all available car brands
- Search for car models and prices by brand name
- Get vehicles by type (cars, motorcycles, trucks)
- Real-time pricing data from FIPE (Brazilian vehicle price reference)
- Comprehensive vehicle information including:
  - Brand and model names
  - Production years
  - Fuel types
  - Current market prices
  - FIPE reference codes

## Usage

The server provides three tools:

### get_car_brands()

Get all available car brands from FIPE API.

**Returns:**
- List of car brands with their codes and names

### search_car_price(brand_name: str)

Search for car models and prices by brand name.

**Parameters:**
- `brand_name`: Car brand name to search for (e.g., "Toyota", "Honda", "Ford")

**Returns:**
- Car models with current market prices from FIPE database

### get_vehicles_by_type(vehicle_type: str)

Get vehicles by type.

**Parameters:**
- `vehicle_type`: Type of vehicles ("carros"/"cars", "motos"/"motorcycles", "caminhoes"/"trucks")

**Returns:**
- List of vehicle brands for the specified type

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python server.py
```

## API

This MCP server uses FIPE API (https://deividfortuna.github.io/fipe/) which provides official Brazilian vehicle pricing data from Fundação Instituto de Pesquisas Econômicas.
