"""Take screenshots of Streamlit app for UAS report."""
import time, sys, os, subprocess
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
OUTPUT = "reports/screenshots"
os.makedirs(OUTPUT, exist_ok=True)

NAV_OPTIONS = [
    "Beranda",
    "Exploratory Data Analysis",
    "Demo Model",
    "Evaluasi & Interpretasi",
]

def main():
    print("Starting Streamlit...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/app.py",
         "--server.headless", "true", "--server.port", "8501"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(6)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            for opt in NAV_OPTIONS:
                name = opt.lower().replace(" ", "_").replace("&", "dan")
                print(f"  Screenshot: {opt}")
                page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
                time.sleep(2)

                # Click radio button in sidebar
                radio_btn = page.locator(f'label:has-text("{opt}")')
                if radio_btn.count() > 0:
                    radio_btn.first.click()
                    time.sleep(3)

                page.screenshot(path=os.path.join(OUTPUT, f"{name}.png"), full_page=True)
                print(f"    -> saved {name}.png")

            browser.close()
        print(f"\n[OK] All screenshots saved to {OUTPUT}/")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
