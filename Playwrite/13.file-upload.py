from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://blueimp.github.io/jQuery-File-Upload/")
    page.set_input_files("input[type=file]", "sample.txt")  # file upload :contentReference[oaicite:10]{index=10}
    browser.close()