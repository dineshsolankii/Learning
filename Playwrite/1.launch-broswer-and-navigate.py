from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Launch browser with UI visible
    browser = p.chromium.launch(headless=False)
    # New tab/page
    page = browser.new_page()
    # Navigate to URL
    page.goto("https://www.google.com")
    # Print page title
    print("Page Title:", page.title())
    # Wait 10 seconds (10000 milliseconds)
    # This is a hard wait mainly for debugging
    page.wait_for_timeout(10000)

    # Close browser
    browser.close()