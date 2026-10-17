"""Generate an RSS 2.0 feed for 完美世界电竞 CS2 news.

Runs inside GitHub Actions on a schedule. Fetches the public wmpvp API and
re-hosts every image in-repo (the source CDN blocks hotlinking), then writes
cs2-news/feed.xml. Feed served via raw.githubusercontent.com; images via jsDelivr.
"""
import datetime
import hashlib
import html
import json
import os
import re
import urllib.request
from email.utils import format_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")
MANIFEST_PATH = os.path.join(HERE, "manifest.json")
FEED_PATH = os.path.join(HERE, "feed.xml")

REPO = "Gino-Huang/huizhou-policy-social-computing"
BRANCH = "main"
CDN_BASE = f"https://cdn.jsdelivr.net/gh/{REPO}@{BRANCH}/cs2-news"

LIST_API = ("https://appengine.wmpvp.com/steamcn/community/homepage/"
            "getHomeInformation?gameTypeStr=2&pageNum=1&pageSize=20")
DETAIL_API = ("https://appactivity.wmpvp.com/steamcn/app/news/"
              "getAppNewsById?gameType=2&newsId={nid}")
ARTICLE_URL = "https://news.wmpvp.com/news.html?id={nid}&gameTypeStr=2"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
REFERER = dict(UA, Referer="https://news.wmpvp.com/")
IMG_URL_RE = re.compile(r"https://cdn\.wmpvp\.com/[^\"'\s)]+")


def fetch(url, headers=None, binary=False):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")


def esc(text):
    return html.escape(text or "", quote=False)


def img_ext(data):
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] in (b"GIF8",):
        return ".gif"
    return None


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

    def rehost(url, key):
        """Download an image (with Referer) into images/, return public URL."""
        if key in manifest:
            return manifest[key]
        try:
            data = fetch(url, headers=REFERER, binary=True)
            ext = img_ext(data)
            if not ext:
                raise ValueError("not an image")
            fname = f"{key}{ext}"
            with open(os.path.join(IMG_DIR, fname), "wb") as f:
                f.write(data)
            public = f"{CDN_BASE}/images/{fname}"
            manifest[key] = public
            print(f"saved image {key}")
            return public
        except Exception as e:  # noqa: BLE001 - keep going without this image
            print(f"image fail {key}: {e}")
            return None

    data = json.loads(fetch(LIST_API))
    news_items = [x["news"] for x in data.get("result", []) if x.get("news")]
    print(f"fetched {len(news_items)} news items")

    rss_items = []
    for n in news_items:
        nid = str(n.get("newsId"))
        title = n.get("title") or ""
        author = n.get("author") or "完美世界电竞"
        link = ARTICLE_URL.format(nid=nid)

        ts = n.get("publishTime")
        try:
            pub = format_datetime(
                datetime.datetime.fromtimestamp(int(ts) / 1000,
                                                tz=datetime.timezone.utc))
        except (TypeError, ValueError):
            pub = format_datetime(datetime.datetime.now(datetime.timezone.utc))

        content = ""
        try:
            det = json.loads(fetch(DETAIL_API.format(nid=nid)))
            content = det["result"]["news"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            print(f"detail fail {nid}: {e}")

        # Re-host thumbnail.
        thumb = n.get("thumbnail")
        img_url = rehost(thumb, nid) if thumb else None

        # Re-host every image inside the article body.
        def _swap(m):
            url = m.group(0)
            key = f"{nid}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
            return rehost(url, key) or url

        content = IMG_URL_RE.sub(_swap, content)

        enclosure = (f'\n    <enclosure url="{esc(img_url)}" type="image/jpeg"/>'
                     if img_url else "")

        # Fluent Reader takes the card thumbnail from the FIRST <img> in the
        # description HTML (it ignores <enclosure>), so lead with our
        # re-hosted thumbnail, then the full re-hosted article body.
        lead = f'<img src="{esc(img_url)}">\n' if img_url else ""
        body = lead + content

        rss_items.append(
            "  <item>\n"
            f"    <title>{esc(title)}</title>\n"
            f"    <link>{esc(link)}</link>\n"
            f"    <guid isPermaLink=\"true\">{esc(link)}</guid>\n"
            f"    <pubDate>{pub}</pubDate>\n"
            f"    <author>{esc(author)}</author>\n"
            f"    <description><![CDATA[{body}]]></description>{enclosure}\n"
            "  </item>"
        )

    now = format_datetime(datetime.datetime.now(datetime.timezone.utc))
    items_xml = "\n".join(rss_items)
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        '<channel>\n'
        '  <title>完美世界电竞 - CS2 资讯</title>\n'
        '  <link>https://news.wmpvp.com/</link>\n'
        '  <description>完美世界电竞 CS2 新闻（含缩略图）</description>\n'
        '  <language>zh-CN</language>\n'
        f'  <lastBuildDate>{now}</lastBuildDate>\n'
        f'{items_xml}\n'
        '</channel>\n'
        '</rss>\n'
    )
    with open(FEED_PATH, "w", encoding="utf-8") as f:
        f.write(feed)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"wrote {FEED_PATH} with {len(rss_items)} items, "
          f"{len(manifest)} images cached")


if __name__ == "__main__":
    main()
