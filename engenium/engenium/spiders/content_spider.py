import scrapy


class IntentSpider(scrapy.Spider):
    name = "intent_scraper"
    allowed_domains = ["amazon.com"]
    start_urls = ["https://www.amazon.com/s?k=ssd"]  # You can generalize later

    def parse(self, response):
        actions = []

        # Extract from buttons
        actions += response.xpath("//button/text()").getall()

        # Extract from anchors (only if they are visible)
        actions += response.xpath("//a[normalize-space(text()) != '']/text()").getall()

        # Extract from input placeholders or associated labels
        actions += response.xpath("//input/@placeholder").getall()
        actions += response.xpath("//label/text()").getall()

        # Extract from dropdown menus
        actions += response.xpath("//select/option/text()").getall()

        # Extract from major headings (like h1, h2) — optional
        actions += response.xpath("//h1/text() | //h2/text()").getall()

        # Clean + deduplicate
        actions = list(set(filter(None, [a.strip() for a in actions])))

        yield {
            "url": response.url,
            "domain": response.url.split("/")[2],
            "page_title": response.xpath("//title/text()").get(),
            "visible_actions": actions
        }

        # Follow internal links (keep it shallow for testing)
        for link in response.css("a::attr(href)").getall()[:5]:  # Limit to first 5 links
            if link and (link.startswith("/") or "amazon.com" in link):
                yield response.follow(link, callback=self.parse)


