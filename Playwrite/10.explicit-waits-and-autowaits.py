# Playwright auto-waits for actions like click
# and waits explicitly for specific selectors:
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://google.com")
    page.click("text=More information")   # auto-wait :contentReference[oaicite:8]{index=8}
    browser.close()