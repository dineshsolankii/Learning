from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(storage_state="auth.json")
    page = context.new_page()
    page.goto("https://google.com")
    context.storage_state(path="auth.json")   # save session :contentReference[oaicite:11]{index=11}
    browser.close()