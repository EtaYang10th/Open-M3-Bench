"""Realistic fixture arguments for every MCP tool.

Structured as: FIXTURES[server][tool] = {arg: value, ...}.
A tool mapped to None is intentionally SKIPPED (destructive / needs prior
state / not meaningfully testable in isolation) with a reason recorded.

Image tools use real files under media/. Search tools use common queries.
Geo tools use real coordinates (NYC / Times Square). Finance uses AAPL.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEDIA = REPO / "media"
SAMPLE_PNG = str((MEDIA / "00000000.png").resolve())
QR_PNG = str((MEDIA / "_qr_healthcheck.png").resolve())
XLSX = str((MEDIA / "_mcp_healthcheck.xlsx").resolve())

# NYC / Times Square reference point
LAT, LNG = 40.7580, -73.9855

FIXTURES = {
    "weather": {
        "get_weather": {"location": "New York"},
    },
    "wiki": {
        "search": {"query": "Alan Turing", "n": 3},
        "summary": {"title": "Alan Turing"},
    },
    "ocr": {
        "perform_ocr": {"input_data": SAMPLE_PNG},
        "get_supported_languages": {},
    },
    "amazon": {
        "search_products": {"keywords": "usb c cable", "n": 2},
        "get_product": {"asin": "B08N5WRWNW"},  # Echo Dot 4th gen
    },
    "google-maps": {
        "geocode": {"address": "1600 Amphitheatre Parkway, Mountain View, CA"},
        "reverse_geocode": {"lat": LAT, "lng": LNG},
        "places_text_search": {"query": "coffee near Times Square", "maxResultCount": 3},
        "places_nearby_search": {"location": {"latitude": LAT, "longitude": LNG},
                                  "radiusMeters": 500, "includedTypes": ["restaurant"]},
        "place_details": {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"},  # Google Sydney
        "place_photo_media": None,  # needs a photo_resource from a prior search
        "compute_route": {"origin": {"location": {"latLng": {"latitude": 40.7128, "longitude": -74.0060}}},
                           "destination": {"location": {"latLng": {"latitude": 40.7580, "longitude": -73.9855}}},
                           "travelMode": "DRIVE"},
        "directions_legacy": {"origin": "New York, NY", "destination": "Boston, MA"},
        "distance_matrix": {"origins": ["New York, NY"], "destinations": ["Boston, MA"]},
        "timezone": {"lat": LAT, "lng": LNG},
        "elevation_by_locations": {"locations": ["40.7580,-73.9855"]},
        "elevation_along_path": {"path": ["40.7128,-74.0060", "40.7580,-73.9855"], "samples": 3},
        "roads_snap_to_roads": {"path": ["40.7128,-74.0060", "40.7138,-74.0050"], "interpolate": True},
        "roads_nearest_roads": {"points": ["40.7128,-74.0060", "40.7138,-74.0050"]},
        "roads_speed_limits": {"path": ["40.7128,-74.0060", "40.7138,-74.0050"], "units": "KPH"},
        "geolocate_home": {"considerIp": True},
        "ping": {},
        "static_map": {"center": "Times Square, New York", "zoom": 14},
        "street_view_image": {"location": "Times Square, New York"},
    },
    "tmdb": {
        "search_movies": {"query": "Inception"},
    },
    "pyzbar-mcp": {
        "scan_barcode": {"image_paths": [QR_PNG]},
    },
    "openlibrary_mcp": {
        "get_book_info": {"title": "The Hobbit", "author": "Tolkien"},
    },
    "imagesorcery-mcp": {
        "crop": {"input_path": SAMPLE_PNG, "x1": 0, "y1": 0, "x2": 50, "y2": 50,
                 "output_path": str((MEDIA / "_crop_healthcheck.png").resolve())},
        "blur": {"input_path": SAMPLE_PNG, "areas": [{"x1": 0, "y1": 0, "x2": 40, "y2": 40}],
                 "output_path": str((MEDIA / "_blur_healthcheck.png").resolve())},
        "draw_arrows": {"input_path": SAMPLE_PNG,
                        "arrows": [{"x1": 0, "y1": 0, "x2": 40, "y2": 40}],
                        "output_path": str((MEDIA / "_arrow_healthcheck.png").resolve())},
        "change_color": {"input_path": SAMPLE_PNG, "palette": "grayscale",
                         "output_path": str((MEDIA / "_color_healthcheck.png").resolve())},
        "draw_circles": {"input_path": SAMPLE_PNG,
                         "circles": [{"center_x": 30, "center_y": 30, "radius": 10}],
                         "output_path": str((MEDIA / "_circle_healthcheck.png").resolve())},
        "draw_lines": {"input_path": SAMPLE_PNG,
                       "lines": [{"x1": 0, "y1": 0, "x2": 40, "y2": 40}],
                       "output_path": str((MEDIA / "_line_healthcheck.png").resolve())},
        "draw_texts": {"input_path": SAMPLE_PNG,
                       "texts": [{"text": "hi", "x": 10, "y": 10}],
                       "output_path": str((MEDIA / "_text_healthcheck.png").resolve())},
    },
    "healthcare-mcp": {
        "fda_drug_lookup": {"drug_name": "ibuprofen"},
        "pubmed_search": {"query": "diabetes", "max_results": 2},
        "medrxiv_search": {"query": "covid", "max_results": 2},
        "calculate_bmi": {"height_meters": 1.75, "weight_kg": 70},
        "ncbi_bookshelf_search": {"query": "anatomy", "max_results": 2},
        "extract_dicom_metadata": None,  # needs a real DICOM file, not available
        "health_topics": {"topic": "diabetes"},
        "clinical_trials_search": {"condition": "diabetes", "max_results": 2},
        "lookup_icd_code": {"description": "diabetes", "max_results": 2},
        "get_usage_stats": {},
        "get_all_usage_stats": {},
    },
    "food_nutrition_mcp": {
        "get_nutrition": {"query": "1 apple"},
    },
    "mcp-yolo": {
        "detect-all-objects": {"imageFileUri": SAMPLE_PNG, "includeDescription": False},
        "detect-objects-by-text": {"imageFileUri": SAMPLE_PNG, "textPrompt": "person", "includeDescription": False},
    },
    "linkimage-mcp": {
        "fetch_unsplash_image": {"url": "https://unsplash.com/photos/a-cat-sitting-on-a-window-sill-1-tR4jaGwRs"},
    },
    "google-air": {
        "current_conditions": {"lat": LAT, "lng": LNG},
        "forecast": {"lat": LAT, "lng": LNG, "hours": 6},
        "history": {"lat": LAT, "lng": LNG,
                    "startTime": "2026-07-20T00:00:00Z", "endTime": "2026-07-20T06:00:00Z"},
        "heatmap_tile": {"z": 4, "x": 4, "y": 6, "indexType": "UAQI_RED_GREEN"},
    },
    "Reddit-MCP-Server": {
        "search_hot_posts": {"subreddit": "python", "limit": 3},
        "get_post_content": None,  # needs a valid post_id from a prior search
    },
    "nationalparks": {
        "findParks": {"stateCode": "CA", "limit": 3},
        "getParkDetails": {"parkCode": "yose"},
        "getAlerts": {"parkCode": "yose", "limit": 3},
        "getVisitorCenters": {"parkCode": "yose", "limit": 3},
        "getCampgrounds": {"parkCode": "yose", "limit": 3},
        "getEvents": {"parkCode": "yose", "limit": 3},
    },
    "metmuseum-mcp": {
        "list-departments": {"__intent": "health check"},
        "search-museum-objects": {"q": "sunflowers", "hasImages": True, "__intent": "health check"},
        "get-museum-object": {"objectId": 436524, "__intent": "health check"},  # Van Gogh
    },
    "okx": {
        "get_price": {"instrument": "BTC-USDT"},
        "get_candlesticks": {"instrument": "BTC-USDT", "bar": "1H", "limit": 5},
    },
    "hugeicons-mcp": {
        "list_icons": {},
        "search_icons": {"query": "user"},
        "get_platform_usage": {"platform": "react"},
        "get_icon_glyphs": {"icon_name": "user"},
        "get_icon_glyph_by_style": {"icon_name": "user", "style": "stroke-rounded"},
    },
    "yahoo-finance": {
        "get_historical_stock_prices": {"ticker": "AAPL", "start_date": "2026-06-01", "end_date": "2026-06-10"},
        "get_stock_info": {"ticker": "AAPL"},
        "get_yahoo_finance_news": {"ticker": "AAPL"},
        "get_stock_actions": {"ticker": "AAPL"},
        "get_financial_statement": {"ticker": "AAPL", "financial_type": "income_stmt"},
        "get_holder_info": {"ticker": "AAPL", "holder_type": "major_holders"},
        "get_option_expiration_dates": {"ticker": "AAPL"},
        "get_option_chain": None,  # needs a valid expiration_date from prior call
        "get_recommendations": {"ticker": "AAPL", "recommendation_type": "recommendations"},
    },
    "math": {
        "add": {"firstNumber": 2, "secondNumber": 3},
        "subtract": {"minuend": 5, "subtrahend": 2},
        "multiply": {"firstNumber": 4, "SecondNumber": 3},
        "division": {"numerator": 10, "denominator": 2},
        "sum": {"numbers": [1, 2, 3, 4]},
        "modulo": {"numerator": 10, "denominator": 3},
        "mean": {"numbers": [1, 2, 3, 4]},
        "median": {"numbers": [1, 2, 3, 4]},
        "mode": {"numbers": [1, 2, 2, 3]},
        "min": {"numbers": [1, 2, 3]},
        "max": {"numbers": [1, 2, 3]},
        "floor": {"number": 3.7},
        "ceiling": {"number": 3.2},
        "round": {"number": 3.5},
        "sin": {"number": 1.0},
        "arcsin": {"number": 0.5},
        "cos": {"number": 1.0},
        "arccos": {"number": 0.5},
        "tan": {"number": 1.0},
        "arctan": {"number": 1.0},
        "radiansToDegrees": {"number": 3.14159},
        "degreesToRadians": {"number": 180},
    },
    "nixos": {
        "nix": {"action": "search", "query": "python", "type": "packages", "limit": 3},
        "nix_versions": {"package": "python", "limit": 3},
    },
    "car-price": {
        "get_car_brands": {},
        "search_brand_model_price": {"brand_name": "Fiat", "model_keyword": "Uno"},
    },
}

# Excel: operates on a shared fixture workbook created by the runner (prep step).
# EXCEL_WB is created with Sheet1 containing a small data grid A1:C4.
EXCEL_WB = str((MEDIA / "_verify_excel.xlsx").resolve())
FIXTURES["excel"] = {
    "create_workbook": {"filepath": str((MEDIA / "_verify_excel_new.xlsx").resolve())},
    "create_worksheet": {"filepath": EXCEL_WB, "sheet_name": "Extra"},
    "write_data_to_excel": {"filepath": EXCEL_WB, "sheet_name": "Sheet1",
                            "data": [["x", "y"], [1, 2], [3, 4]], "start_cell": "E1"},
    "read_data_from_excel": {"filepath": EXCEL_WB, "sheet_name": "Sheet1",
                             "start_cell": "A1", "end_cell": "C4"},
    "apply_formula": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "cell": "D1", "formula": "=SUM(B2:B4)"},
    "validate_formula_syntax": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "cell": "D2", "formula": "=SUM(B2:B4)"},
    "format_range": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_cell": "A1", "end_cell": "C1", "bold": True},
    "create_chart": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "data_range": "B1:B4",
                     "chart_type": "line", "target_cell": "G1", "title": "t"},
    "create_pivot_table": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "data_range": "A1:C30",
                           "rows": ["a"], "values": ["b"], "agg_func": "sum"},
    "create_table": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "data_range": "A1:C30", "table_name": "VerifyTbl"},
    "copy_worksheet": {"filepath": EXCEL_WB, "source_sheet": "Sheet1", "target_sheet": "Sheet1Copy"},
    "delete_worksheet": {"filepath": EXCEL_WB, "sheet_name": "Sheet1Copy"},
    "rename_worksheet": {"filepath": EXCEL_WB, "old_name": "Extra", "new_name": "Renamed"},
    "get_workbook_metadata": {"filepath": EXCEL_WB},
    "merge_cells": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_cell": "A6", "end_cell": "C6"},
    "unmerge_cells": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_cell": "A6", "end_cell": "C6"},
    "get_merged_cells": {"filepath": EXCEL_WB, "sheet_name": "Sheet1"},
    "copy_range": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "source_start": "A1",
                   "source_end": "C4", "target_start": "H1"},
    "delete_range": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_cell": "H1",
                     "end_cell": "J4", "shift_direction": "up"},
    "validate_excel_range": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_cell": "A1", "end_cell": "C30"},
    "get_data_validation_info": {"filepath": EXCEL_WB, "sheet_name": "Sheet1"},
    "insert_rows": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_row": 15, "count": 1},
    "insert_columns": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_col": 2, "count": 1},
    "delete_sheet_rows": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_row": 15, "count": 1},
    "delete_sheet_columns": {"filepath": EXCEL_WB, "sheet_name": "Sheet1", "start_col": 2, "count": 1},
}

# nasa-mcp: uses DEMO_KEY by default (rate-limited but live). DONKI space-weather
# tools need a date range with known activity; use a wide historical window.
FIXTURES["nasa-mcp"] = {
    "get_astronomy_picture_of_day": {"date": "2024-01-01"},
    "get_asteroids_feed": {"start_date": "2024-01-01", "end_date": "2024-01-02"},
    "get_asteroid_lookup": {"asteroid_id": "3542519"},
    "browse_asteroids": {},
    "get_coronal_mass_ejection": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
    "get_geomagnetic_storm": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
    "get_solar_flare": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
    "get_solar_energetic_particle": {"start_date": "2024-01-01", "end_date": "2024-06-30"},
    "get_magnetopause_crossing": {"start_date": "2024-01-01", "end_date": "2024-06-30"},
    "get_radiation_belt_enhancement": {"start_date": "2024-01-01", "end_date": "2024-06-30"},
    "get_hight_speed_stream": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
    "get_wsa_enlil_simulation": {"start_date": "2024-01-01", "end_date": "2024-03-31"},
    "get_notifications": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
    "get_earth_imagery": {"lat": 40.758, "lon": -73.9855, "date": "2021-01-01"},
    "get_earth_assets": {"lat": 40.758, "lon": -73.9855, "date": "2021-01-01"},
    "get_epic_imagery": {},
    "get_epic_imagery_by_date": {"date": "2024-01-01"},
    "get_epic_dates": {},
    "get_exoplanet_data": {"table": "exoplanets", "format": "json"},
    "get_mars_rover_photos": {"rover_name": "curiosity", "sol": 1000},
    "get_mars_rover_manifest": {"rover_name": "curiosity"},
}

# paper_search: exercise the search_* tools + a couple of well-known-id reads.
FIXTURES["paper_search"] = {
    "search_arxiv": {"query": "machine learning", "max_results": 2},
    "search_pubmed": {"query": "diabetes", "max_results": 2},
    "search_biorxiv": {"query": "genomics", "max_results": 2},
    "search_medrxiv": {"query": "covid", "max_results": 2},
    "search_google_scholar": {"query": "machine learning", "max_results": 2},
    "search_iacr": {"query": "encryption", "max_results": 2, "fetch_details": False},
    "search_semantic": {"query": "machine learning", "max_results": 2},
    "search_crossref": {"query": "machine learning", "max_results": 2, "kwargs": ""},
    "get_crossref_paper_by_doi": {"doi": "10.1038/nature14539"},
    "download_arxiv": {"paper_id": "1706.03762", "save_path": str((MEDIA / "_papers").resolve())},
    "read_arxiv_paper": {"paper_id": "1706.03762", "save_path": str((MEDIA / "_papers").resolve())},
    "download_pubmed": None, "download_biorxiv": None, "download_medrxiv": None,
    "download_iacr": None, "download_semantic": None, "download_crossref": None,
    "read_pubmed_paper": None, "read_biorxiv_paper": None, "read_medrxiv_paper": None,
    "read_iacr_paper": None, "read_semantic_paper": None, "read_crossref_paper": None,
}

# ppt: exercise a representative subset (server-info + create + auto-generate).
# Full 37-tool coverage requires long stateful chains; we test the core ones
# that prove the server truly works and produces files.
_PPTX = str((MEDIA / "_verify_ppt.pptx").resolve())
FIXTURES["ppt"] = {
    "get_server_info": {},
    "create_presentation": {"id": "vp"},
    "add_slide": {"layout_index": 1, "title": "Hello", "presentation_id": "vp"},
    "get_presentation_info": {"presentation_id": "vp"},
    "extract_presentation_text": {"presentation_id": "vp"},
    "list_presentations": {},
    "add_table": {"slide_index": 0, "rows": 2, "cols": 2, "left": 1, "top": 1,
                  "width": 5, "height": 3, "data": [["a", "b"], ["c", "d"]], "presentation_id": "vp"},
    "save_presentation": {"file_path": _PPTX, "presentation_id": "vp"},
    "auto_generate_presentation": {"topic": "Renewable Energy", "slide_count": 3, "presentation_id": "vp"},
    # The remaining ppt tools need specific prior state/templates; skip explicitly.
    "create_presentation_from_template": None,
    "create_presentation_from_templates": None,
    "open_presentation": None,
    "get_template_file_info": None,
    "get_template_info": None,
    "set_core_properties": {"title": "T", "presentation_id": "vp"},
    "get_slide_info": {"slide_index": 0, "presentation_id": "vp"},
    "extract_slide_text": {"slide_index": 0, "presentation_id": "vp"},
    "populate_placeholder": None,
    "add_bullet_points": None,
    "manage_text": None,
    "manage_image": None,
    "format_table_cell": None,
    "add_shape": {"slide_index": 0, "shape_type": "rectangle", "left": 1, "top": 1,
                  "width": 2, "height": 1, "presentation_id": "vp"},
    "add_chart": None,
    "apply_professional_design": None,
    "apply_picture_effects": None,
    "manage_fonts": None,
    "list_slide_templates": {},
    "apply_slide_template": None,
    "create_slide_from_template": None,
    "optimize_slide_text": None,
    "manage_hyperlinks": None,
    "update_chart_data": None,
    "add_connector": None,
    "manage_slide_masters": None,
    "manage_slide_transitions": None,
    "switch_presentation": {"presentation_id": "vp"},
}

