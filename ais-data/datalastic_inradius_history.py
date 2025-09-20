#!/usr/bin/env python3
"""
Datalastic Inradius History Report API Client

This script handles the asynchronous report system for historical vessel location data.
It submits report jobs, polls for completion, and downloads the results.

API Documentation: https://datalastic.com/api-reference/
"""

import requests
import json
import time
import zipfile
import csv
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse
import sys


class DatalasticInradiusHistoryAPI:
    """
    Client for Datalastic Inradius History Report API
    """
    
    # API Configuration Constants
    BASE_URL = "https://api.datalastic.com/api/v0"
    REPORT_ENDPOINT = "/report"
    
    # Report status constants
    STATUS_PENDING = "_PENDING_"
    STATUS_IN_PROGRESS = "_IN_PROGRESS_"
    STATUS_DONE = "_DONE_"
    
    # Rate limiting constants
    RATE_LIMIT_PER_MINUTE = 600
    MIN_REQUEST_INTERVAL = 0.1  # seconds between requests
    POLL_INTERVAL = 30  # seconds between status checks
    
    # Default parameters
    DEFAULT_RADIUS = 10  # nautical miles
    DEFAULT_VESSEL_TYPE = "all"
    DEFAULT_MAX_VESSELS = 500  # maximum vessels to return
    
    def __init__(self, api_key: str):
        """
        Initialize the Datalastic API client
        
        Args:
            api_key (str): Your Datalastic API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Apply rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None, method: str = "GET") -> Dict[str, Any]:
        """
        Make a request to the Datalastic API with error handling
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Request parameters for GET requests
            data (Dict): Request data for POST requests
            method (str): HTTP method (GET or POST)
            
        Returns:
            Dict: API response data
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        # Add API key to parameters or data
        if method == "GET" and params:
            params['api-key'] = self.api_key
        elif method == "POST" and data:
            data['api-key'] = self.api_key
        
        try:
            if method == "GET":
                print(f"Making GET request to: {url}")
                print(f"Params: {params}")
                response = self.session.get(url, params=params, timeout=30)
            else:
                print(f"Making POST request to: {url}")
                print(f"Data: {json.dumps(data, indent=2)}")
                response = self.session.post(url, json=data, timeout=30)
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return {"error": "Invalid JSON response"}
    
    def submit_inradius_history_report(self, 
                                     latitude: float, 
                                     longitude: float, 
                                     from_date: str,
                                     to_date: str,
                                     radius: int = DEFAULT_RADIUS,
                                     vessel_type: str = DEFAULT_VESSEL_TYPE,
                                     max_vessels: int = DEFAULT_MAX_VESSELS) -> Dict[str, Any]:
        """
        Submit a report job for historical vessel location data
        
        Args:
            latitude (float): Latitude of the center point
            longitude (float): Longitude of the center point
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            radius (int): Search radius in nautical miles (max 50)
            vessel_type (str): Type of vessels to search for
            max_vessels (int): Maximum number of vessels to return (max 500)
            
        Returns:
            Dict: Report submission response with report_id
        """
        payload = {
            "report_type": "inradius_history",
            "lat": latitude,
            "lon": longitude,
            "radius": min(radius, 50),  # API limit is 50 NM
            "vessel_type": vessel_type,
            "max_vessels": min(max_vessels, 500),  # API limit is 500
            "from": from_date,
            "to": to_date
        }
        
        print(f"Submitting inradius history report...")
        print(f"Location: ({latitude}, {longitude})")
        print(f"Date range: {from_date} to {to_date}")
        print(f"Radius: {radius} NM, Vessel type: {vessel_type}, Max vessels: {max_vessels}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = self._make_request(self.REPORT_ENDPOINT, data=payload, method="POST")
        print(f"API Response: {json.dumps(response, indent=2)}")
        return response
    
    def check_report_status(self, report_id: str) -> Dict[str, Any]:
        """
        Check the status of a submitted report
        
        Args:
            report_id (str): The report ID returned from submission
            
        Returns:
            Dict: Report status information
        """
        params = {"report_id": report_id}
        return self._make_request(self.REPORT_ENDPOINT, params=params)
    
    def wait_for_report_completion(self, report_id: str, max_wait_minutes: int = 30) -> Dict[str, Any]:
        """
        Wait for a report to complete and return the final result
        
        Args:
            report_id (str): The report ID to wait for
            max_wait_minutes (int): Maximum time to wait in minutes
            
        Returns:
            Dict: Final report result or error
        """
        print(f"Waiting for report {report_id} to complete...")
        print(f"Maximum wait time: {max_wait_minutes} minutes")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        while time.time() - start_time < max_wait_seconds:
            status_response = self.check_report_status(report_id)
            
            if "error" in status_response:
                print(f"Error checking status: {status_response['error']}")
                return status_response
            
            # Extract status from nested data structure
            data = status_response.get("data", {})
            status = data.get("status", "UNKNOWN")
            print(f"Report status: {status}")
            print(f"Full status response: {json.dumps(status_response, indent=2)}")
            
            if status == self.STATUS_DONE:
                print("Report completed successfully!")
                return status_response
            elif status in [self.STATUS_PENDING, self.STATUS_IN_PROGRESS]:
                print(f"Report still processing... waiting {self.POLL_INTERVAL} seconds")
                time.sleep(self.POLL_INTERVAL)
            else:
                print(f"Unexpected status: {status}")
                return {"error": f"Unexpected status: {status}"}
        
        print(f"Timeout: Report did not complete within {max_wait_minutes} minutes")
        return {"error": "Report timeout"}
    
    def download_report_data(self, result_url: str) -> Dict[str, Any]:
        """
        Download the actual report data from the result URL (ZIP file)
        
        Args:
            result_url (str): URL to download the report data from
            
        Returns:
            Dict: Downloaded report data
        """
        print(f"Downloading report data from: {result_url}")
        
        try:
            response = self.session.get(result_url, timeout=60)
            response.raise_for_status()
            
            # Check if it's a ZIP file
            if result_url.endswith('.zip'):
                return self._process_zip_data(response.content)
            else:
                # Try to parse as JSON
                return response.json()
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading report data: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error parsing downloaded data: {e}")
            return {"error": "Invalid JSON in downloaded data"}
    
    def _process_zip_data(self, zip_content: bytes) -> Dict[str, Any]:
        """
        Process ZIP file content and extract CSV data
        
        Args:
            zip_content (bytes): ZIP file content
            
        Returns:
            Dict: Processed vessel data
        """
        try:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write ZIP content to temporary file
                zip_path = os.path.join(temp_dir, "report.zip")
                with open(zip_path, 'wb') as f:
                    f.write(zip_content)
                
                # Extract ZIP file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Look for CSV files
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
                
                print(f"Found files in ZIP: CSV={csv_files}, JSON={json_files}")
                
                # Process CSV file
                vessels = []
                if csv_files:
                    csv_path = os.path.join(temp_dir, csv_files[0])
                    vessels = self._parse_csv_file(csv_path)
                
                # Process meta.json if available
                meta_data = {}
                if json_files:
                    meta_path = os.path.join(temp_dir, json_files[0])
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                
                return {
                    "data": vessels,
                    "meta": meta_data,
                    "total_vessels": len(vessels)
                }
                
        except Exception as e:
            print(f"Error processing ZIP data: {e}")
            return {"error": str(e)}
    
    def _parse_csv_file(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Parse CSV file and convert to list of vessel dictionaries
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            List[Dict]: List of vessel data
        """
        vessels = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Convert CSV row to vessel data
                    vessel = {
                        'uuid': row.get('uuid', ''),
                        'lat': float(row.get('lat', 0)) if row.get('lat') else None,
                        'lon': float(row.get('lon', 0)) if row.get('lon') else None,
                        'speed': float(row.get('speed', 0)) if row.get('speed') else None,
                        'course': float(row.get('course', 0)) if row.get('course') else None,
                        'heading': float(row.get('heading', 0)) if row.get('heading') else None,
                        'navstat': row.get('navstat', ''),
                        'destination': row.get('destination', ''),
                        'last_pos_epoch': row.get('last_pos_epoch', ''),
                        'last_pos_utc': row.get('last_pos_utc', ''),
                        'distance_nm': float(row.get('distance_nm', 0)) if row.get('distance_nm') else None
                    }
                    vessels.append(vessel)
                    
        except Exception as e:
            print(f"Error parsing CSV file: {e}")
            
        return vessels
    
    def get_inradius_history_data(self, 
                                 latitude: float, 
                                 longitude: float, 
                                 from_date: str,
                                 to_date: str,
                                 radius: int = DEFAULT_RADIUS,
                                 vessel_type: str = DEFAULT_VESSEL_TYPE,
                                 max_vessels: int = DEFAULT_MAX_VESSELS,
                                 max_wait_minutes: int = 30) -> Dict[str, Any]:
        """
        Complete workflow: submit report, wait for completion, and download data
        
        Args:
            latitude (float): Latitude of the center point
            longitude (float): Longitude of the center point
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            radius (int): Search radius in nautical miles (max 50)
            vessel_type (str): Type of vessels to search for
            max_vessels (int): Maximum number of vessels to return (max 500)
            max_wait_minutes (int): Maximum time to wait for completion
            
        Returns:
            Dict: Historical vessel location data
        """
        # Step 1: Submit the report
        submit_response = self.submit_inradius_history_report(
            latitude, longitude, from_date, to_date, radius, vessel_type, max_vessels
        )
        
        if "error" in submit_response:
            return submit_response
        
        # Extract report_id from the nested data structure
        data = submit_response.get("data", {})
        report_id = data.get("report_id")
        
        if not report_id:
            print(f"Available fields in response: {list(submit_response.keys())}")
            if "data" in submit_response:
                print(f"Available fields in data: {list(submit_response['data'].keys())}")
            return {"error": "No report_id returned from submission", "response": submit_response}
        
        print(f"Report submitted successfully. Report ID: {report_id}")
        
        # Step 2: Wait for completion
        final_response = self.wait_for_report_completion(report_id, max_wait_minutes)
        
        if "error" in final_response:
            return final_response
        
        # Step 3: Download the data
        # Extract result_url from nested data structure
        data = final_response.get("data", {})
        result_url = data.get("result_url")
        if not result_url:
            print(f"Available fields in final response: {list(final_response.keys())}")
            if "data" in final_response:
                print(f"Available fields in data: {list(final_response['data'].keys())}")
            return {"error": "No result_url in final response", "response": final_response}
        
        return self.download_report_data(result_url)
    
    def format_historical_data(self, data: Dict[str, Any]) -> None:
        """
        Format and display historical vessel data in a readable format
        
        Args:
            data (Dict): Raw API response data
        """
        if 'error' in data:
            print(f"Error: {data['error']}")
            return
        
        if 'data' not in data or not data['data']:
            print("No historical vessel data found for the specified location and date range.")
            return
        
        vessels = data['data']
        total_vessels = data.get('total_vessels', len(vessels))
        
        print(f"\n{'='*80}")
        print(f"HISTORICAL VESSEL DATA REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Total vessels found: {total_vessels}")
        
        if total_vessels == 0:
            print("No vessels were found in the specified location and time period.")
            print("This could mean:")
            print("- No vessels were in the area during that time")
            print("- The radius might be too small")
            print("- The date might not have data available")
            return
        
        print(f"{'='*80}")
        
        for i, vessel in enumerate(vessels, 1):
            print(f"\nVessel #{i}")
            print(f"  UUID: {vessel.get('uuid', 'N/A')}")
            print(f"  Position: {vessel.get('lat', 'N/A')}, {vessel.get('lon', 'N/A')}")
            print(f"  Speed: {vessel.get('speed', 'N/A')} knots")
            print(f"  Course: {vessel.get('course', 'N/A')}°")
            print(f"  Heading: {vessel.get('heading', 'N/A')}°")
            print(f"  Navigation Status: {vessel.get('navstat', 'N/A')}")
            print(f"  Destination: {vessel.get('destination', 'N/A')}")
            print(f"  Last Position UTC: {vessel.get('last_pos_utc', 'N/A')}")
            print(f"  Last Position Epoch: {vessel.get('last_pos_epoch', 'N/A')}")
            print(f"  Distance: {vessel.get('distance_nm', 'N/A')} NM")
            print("-" * 40)
    
    def save_to_file(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        Save historical vessel data to a JSON file
        
        Args:
            data (Dict): Historical vessel data
            filename (str): Optional filename, defaults to timestamp-based name
            
        Returns:
            str: Filename of the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inradius_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Data saved to: {filename}")
        return filename


def main():
    """
    Main function to run the Inradius History Report API client
    """
    parser = argparse.ArgumentParser(description='Datalastic Inradius History Report API Client')
    parser.add_argument('--api-key', required=True, help='Your Datalastic API key')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the center point')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the center point')
    parser.add_argument('--from-date', required=True, help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--to-date', required=True, help='End date (YYYY-MM-DD format)')
    parser.add_argument('--radius', type=int, default=10, help='Search radius in nautical miles (max 50)')
    parser.add_argument('--vessel-type', default='all', help='Type of vessels to search for')
    parser.add_argument('--max-vessels', type=int, default=500, help='Maximum number of vessels to return (max 500)')
    parser.add_argument('--max-wait', type=int, default=30, help='Maximum wait time in minutes (default: 30)')
    parser.add_argument('--save', action='store_true', help='Save data to JSON file')
    parser.add_argument('--output-file', help='Output filename for saved data')
    
    args = parser.parse_args()
    
    # Validate coordinates
    if not (-90 <= args.lat <= 90):
        print("Error: Latitude must be between -90 and 90")
        sys.exit(1)
    
    if not (-180 <= args.lon <= 180):
        print("Error: Longitude must be between -180 and 180")
        sys.exit(1)
    
    # Validate dates
    try:
        from_date_obj = datetime.strptime(args.from_date, '%Y-%m-%d')
        to_date_obj = datetime.strptime(args.to_date, '%Y-%m-%d')
        
        if from_date_obj > to_date_obj:
            print("Error: From date must be before or equal to to date")
            sys.exit(1)
            
        # Check if date range is not too large (max 7 days for historical data)
        if (to_date_obj - from_date_obj).days > 7:
            print("Warning: Date range is larger than 7 days. API may limit results.")
            
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    # Initialize API client
    api = DatalasticInradiusHistoryAPI(args.api_key)
    
    # Get historical vessel data
    historical_data = api.get_inradius_history_data(
        latitude=args.lat,
        longitude=args.lon,
        from_date=args.from_date,
        to_date=args.to_date,
        radius=args.radius,
        vessel_type=args.vessel_type,
        max_vessels=args.max_vessels,
        max_wait_minutes=args.max_wait
    )
    
    # Display formatted data
    api.format_historical_data(historical_data)
    
    # Save to file if requested
    if args.save:
        filename = api.save_to_file(historical_data, args.output_file)
        print(f"\nData saved to: {filename}")


if __name__ == "__main__":
    # Example usage with hardcoded values (uncomment to use)
    # You can also run this script with command line arguments
    
    # Example coordinates and date
    EXAMPLE_LAT = -53.49
    EXAMPLE_LON = 69.14
    EXAMPLE_FROM_DATE = "2024-09-09"
    EXAMPLE_TO_DATE = "2024-09-09"
    EXAMPLE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
    
    # Uncomment the following lines to run with example data
    # api = DatalasticInradiusHistoryAPI(EXAMPLE_API_KEY)
    # historical_data = api.get_inradius_history_data(EXAMPLE_LAT, EXAMPLE_LON, EXAMPLE_FROM_DATE, EXAMPLE_TO_DATE)
    # api.format_historical_data(historical_data)
    
    # Run with command line arguments
    main()
