from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.w3schools.com/html/html_forms.asp")
    page.fill("input[name='firstname']", "dinesh")
    page.fill("input[name='lastname']", "solanki")
    # page.click("input[type='submit']")
    page.wait_for_timeout(5000)
    browser.close()