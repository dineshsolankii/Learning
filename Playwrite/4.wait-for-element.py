from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://apple.com")
    page.wait_for_selector("h1")   # waits for element to load :contentReference[oaicite:2]{index=2}
    print(page.inner_text("h1"))
    page.wait_for_timeout(5000)
    browser.close()
