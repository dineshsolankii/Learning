from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.route("**/*", lambda route: route.continue_())  # intercept :contentReference[oaicite:9]{index=9}
    page.goto("https://google.com")
    browser.close()