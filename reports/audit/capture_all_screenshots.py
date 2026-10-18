import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1280,800")

driver = webdriver.Chrome(options=options)
driver.get("http://localhost:8501")
time.sleep(3)

# 1. First load (at initial lap 3)
driver.save_screenshot("reports/audit/screenshots/first_load.png")
driver.save_screenshot("reports/audit/screenshots/stint_lap3.png")

# Find buttons
buttons = driver.find_elements(By.CSS_SELECTOR, "button")
forward_btn = None
back_btn = None
for b in buttons:
    try:
        t = b.text
        if "Forward" in t:
            forward_btn = b
        elif "Back" in t:
            back_btn = b
    except Exception:
        pass

# 2. Advance 6 laps for mid-stint
if forward_btn:
    for _ in range(6):
        forward_btn.click()
        time.sleep(0.5)

driver.save_screenshot("reports/audit/screenshots/stint_mid.png")

# 3. Advance to last lap
if forward_btn:
    for _ in range(30):
        try:
            forward_btn.click()
            time.sleep(0.15)
        except Exception:
            break

driver.save_screenshot("reports/audit/screenshots/stint_last.png")

# 4. Narrow width check (e.g. 500x800)
driver.set_window_size(500, 800)
time.sleep(1)
driver.save_screenshot("reports/audit/screenshots/narrow_width.png")

driver.quit()
print("Successfully generated all required screenshots in reports/audit/screenshots/")
