from playwright.sync_api import sync_playwright

def handle_dialog(dialog):
    print("Dialog says:", dialog.message)
    dialog.accept()

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.on("dialog", handle_dialog)
    page.evaluate("alert('Hello!')")     # triggers dialog :contentReference[oaicite:7]{index=7}
    browser.close()