import time
from playwright.sync_api import sync_playwright

def scroll_down(page):
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # JS scroll :contentReference[oaicite:12]{index=12}
    time.sleep(2)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://apple.com")
    for i in range(5):
        scroll_down(page)
    browser.close()