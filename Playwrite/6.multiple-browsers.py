from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # browser1 = p.chromium.launch(headless=False)
    # page1 = browser1.new_page()
    # page1.goto("https://google.com")
    # page1.wait_for_selector("h1")   # waits for element to load :contentReference[oaicite:2]{index=2}
    # print(page1.inner_text("h1"))
    # page1.wait_for_timeout(5000)
    # browser1.close()

    # browser2 = p.firefox.launch(headless=False)
    # page2 = browser2.new_page()
    # page2.goto("https://apple.com")
    # page2.wait_for_selector("h1")   # waits for element to load :contentReference[oaicite:2]{index=2}
    # print(page2.inner_text("h1"))
    # page2.wait_for_timeout(5000)
    # browser2.close()

    # browser3 = p.webkit.launch(headless=False)
    # page3 = browser3.new_page()
    # page3.goto("https://microsoft.com")
    # page3.wait_for_selector("h1")   # waits for element to load :contentReference[oaicite:2]{index=2}
    # print(page3.inner_text("h1"))
    # page3.wait_for_timeout(5000)
    # browser3.close()

    for browser_type in [p.chromium, p.firefox, p.webkit]:  # cross-browser :contentReference[oaicite:4]{index=4}
        b = browser_type.launch()
        page = b.new_page()
        page.goto("https://google.com")
        print(browser_type.name, "->", page.title())
        b.close()
