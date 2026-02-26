from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Launch browser
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    # Disable navigation timeout to prevent 30000ms error
    page.set_default_navigation_timeout(0)
    # Navigate to Google search results instead of homepage
    page.goto("https://www.google.com/search?q=playwright+python")
    # Use a realistic selector (Google’s homepage has no <h1>)
    # Here we look for a result heading
    text = page.locator("h3").first.text_content()
    print("Extracted text:", text)
    browser.close()