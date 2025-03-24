import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Config
OUTPUT_FOLDER = "downloaded_images"
TARGET_CLASS = "media-content"
CHECK_INTERVAL = 5  # seconds between scans

seen_images = set()

# Create folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Connect to running Chrome session
chrome_options = Options()
chrome_options.debugger_address = "localhost:9222"
driver = webdriver.Chrome(options=chrome_options)

print("🔥 Live image monitor running... browse and scroll!")

try:
    while True:
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        media_links = soup.find_all("a", class_=TARGET_CLASS)

        for link in media_links:
            img = link.find("img")
            if img:
                src = img.get("src") or img.get("data-src")
                if src and src not in seen_images:
                    seen_images.add(src)
                    img_url = urljoin(driver.current_url, src)

                    try:
                        response = requests.get(img_url)
                        content_type = response.headers.get("Content-Type", "")
                        if "image" in content_type:
                            ext = content_type.split("/")[-1].split(";")[0]
                            filename = os.path.join(
                                OUTPUT_FOLDER,
                                os.path.basename(src).split("?")[0] + "." + ext
                            )
                            with open(filename, "wb") as f:
                                f.write(response.content)
                            print(f"✅ Downloaded: {filename}")
                        else:
                            print(f"⚠️ Skipped non-image content: {img_url}")
                    except Exception as e:
                        print(f"❌ Error downloading {img_url}: {e}")

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
