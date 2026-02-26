from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Launch browser
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # Navigate to the W3Schools TryIt editor
    page.goto("https://www.w3schools.com/tags/tryit.asp?filename=tryhtml_select")
    # Wait for iframe to be present
    page.wait_for_selector("iframe#iframeResult")
    # Get a handle to the frame where the actual HTML form lives
    iframe = page.frame_locator("iframe#iframeResult")
    # Now select option by value within the iframe
    iframe.locator("select").select_option("saab")
    # Pause so we can see the change
    page.wait_for_timeout(3000)

    browser.close()