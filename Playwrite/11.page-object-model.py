# simplified POM
class ExamplePage:
    def __init__(self, page):
        self.page = page
    def open(self):
        self.page.goto("https://google.com")

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    example = ExamplePage(page)
    example.open()
    browser.close()