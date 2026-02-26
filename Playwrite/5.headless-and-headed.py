from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    p.chromium.launch(headless=True)

    page = browser.new_page()
    page.wait_for_timeout(5000)
    browser.close()
