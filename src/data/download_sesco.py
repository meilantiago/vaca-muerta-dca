import urllib.request
import json
import os
from pathlib import Path
import ssl

def get_sesco_urls(years=[2023, 2024]):
    """Queries the Argentine government data API to get the latest SESCO production URLs."""
    print("Querying datos.gob.ar for SESCO datasets...")
    url = "https://datos.gob.ar/api/3/action/package_search?q=produccion-de-petroleo-y-gas-por-pozo"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    # Ignore SSL verification just in case of local cert issues
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx) as response:
        data = json.loads(response.read().decode())
        
    download_urls = {}
    for result in data['result']['results']:
        if result['name'] == 'produccion-de-petroleo-y-gas-por-pozo':
            for resource in result['resources']:
                for year in years:
                    # We avoid the 'DDJJ abiertas y cerradas' to keep it to the standard dataset
                    if str(year) in resource['name'] and 'DDJJ' not in resource['name']:
                        download_urls[year] = resource['url']
    
    return download_urls

def download_data(urls, dest_folder):
    """Downloads the CSV files to the destination folder."""
    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for year, url in urls.items():
        filename = dest_path / f"produccion_{year}.csv"
        if filename.exists():
            print(f"File {filename.name} already exists. Skipping.")
            continue
            
        print(f"Downloading data for {year} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, context=ctx) as response, open(filename, 'wb') as out_file:
            # We read in chunks because the files can be large (100MB+)
            chunk_size = 1024 * 1024 * 10 # 10 MB chunks
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                print(f"  ...Downloaded {downloaded / (1024*1024):.1f} MB", end='\r')
        print(f"\nSaved to {filename}")

if __name__ == "__main__":
    # Calculate path relative to this script
    current_dir = Path(__file__).resolve().parent
    raw_data_dir = current_dir.parent.parent / "data" / "raw"
    
    urls = get_sesco_urls(years=[2023, 2024])
    if not urls:
        print("No URLs found. Check the API response.")
    else:
        download_data(urls, raw_data_dir)
        print("Download complete.")
