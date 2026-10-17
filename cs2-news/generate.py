"""Generate an RSS 2.0 feed for 完美世界电竞 CS2 news.

Runs inside GitHub Actions on a schedule. Fetches the public wmpvp API,
re-hosts thumbnails (the source CDN blocks hotlinking) inside this repo,
and writes cs2-news/feed.xml served via jsDelivr.
"""
import datetime
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

# jsDelivr serves every committed file in the repo at a stable URL.
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


def fetch(url, headers=None, binary=False):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")


def esc(text):
    return html.escape(text or "", quote=False)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

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

        # Full article body (HTML) from the detail API.
        content = ""
        try:
            det = json.loads(fetch(DETAIL_API.format(nid=nid)))
            content = det["result"]["news"].get("content") or ""
        except Exception as e:  # noqa: BLE001 - keep the item even if body fails
            print(f"detail fail {nid}: {e}")

        # Thumbnail: re-host in-repo so readers are not blocked by hotlink protection.
        img_url = manifest.get(nid)
        thumb = n.get("thumbnail")
        if not img_url and thumb:
            try:
                img_data = fetch(thumb, headers=REFERER, binary=True)
                if img_data[:2] != b"\xff\xd8":
                    raise ValueError("not a jpeg")
                fname = f"{nid}.jpg"
                with open(os.path.join(IMG_DIR, fname), "wb") as f:
                    f.write(img_data)
                img_url = f"{CDN_BASE}/images/{fname}"
                manifest[nid] = img_url
                print(f"saved thumbnail {nid}")
            except Exception as e:  # noqa: BLE001
                print(f"thumbnail fail {nid}: {e}")

        enclosure = (f'\n    <enclosure url="{esc(img_url)}" type="image/jpeg"/>'
                     if img_url else "")

        # Fluent Reader takes the card thumbnail from the FIRST <img> in the
        # description HTML (it ignores <enclosure>), so lead with our
        # re-hosted thumbnail. Source CDN images stay broken otherwise.
        # Keep the description lean (thumbnail + excerpt): full pages are
        # read via the article link anyway.
        excerpt = re.sub(r"<[^>]+>", "", content or "")
        excerpt = re.sub(r"\s+", " ", excerpt).strip()[:300]
        body = (f'<img src="{esc(img_url)}">\n<p>{esc(excerpt)}…</p>\n'
                f'<p><a href="{esc(link)}">阅读全文</a></p>' if img_url
                else f"<p>{esc(excerpt)}…</p>\n"
                     f'<p><a href="{esc(link)}">阅读全文</a></p>')

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
          f"{len(manifest)} thumbnails cached")


if __name__ == "__main__":
    main()
