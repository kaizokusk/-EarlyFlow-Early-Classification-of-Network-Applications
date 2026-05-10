"""
domain_map.py — Final 6-class unambiguous domain map
Built from actual IITD campus DNS data (dns_names.csv)

6 Classes:
  streaming     — pure video/audio CDNs only
  social_media  — social platforms + messaging merged in
  conferencing  — real-time bidirectional audio/video calls
  cloud         — storage, productivity, dev tools, updates
  web           — search, AI, news, ecommerce, browsing
  gaming        — game servers and pure gaming platforms
"""

DOMAIN_MAP = {

    # ── STREAMING ─────────────────────────────────────────────────
    # CDN domains (suffix match catches all subdomains)
    "googlevideo.com":       ("youtube",     "streaming"),
    "ytimg.com":             ("youtube",     "streaming"),
    "ggpht.com":             ("youtube",     "streaming"),
    "nflxvideo.net":         ("netflix",     "streaming"),
    "nflximg.net":           ("netflix",     "streaming"),
    "nflxso.net":            ("netflix",     "streaming"),
    "scdn.co":               ("spotify",     "streaming"),
    "spotifycdn.com":        ("spotify",     "streaming"),
    "tospotify.com":         ("spotify",     "streaming"),
    "jtvnw.net":             ("twitch",      "streaming"),
    "twitchapps.com":        ("twitch",      "streaming"),
    "hotstarext.com":        ("hotstar",     "streaming"),
    "amazonvideo.com":       ("prime",       "streaming"),
    "crunchyroll.com":       ("crunchyroll", "streaming"),
    # Main domains added for qname fallback in DNS map
    # (DNS responses often use subdomains like apresolve.spotify.com
    #  which suffix-match these entries correctly)
    "spotify.com":           ("spotify",     "streaming"),
    "hotstar.com":           ("hotstar",     "streaming"),
    "jio.com":               ("jio",         "streaming"),
    "jiosaavn.com":          ("jiosaavn",    "streaming"),
    "netflix.com":           ("netflix",     "streaming"),
    "primevideo.com":        ("prime",       "streaming"),
    "twitch.tv":             ("twitch",      "streaming"),

    # ── SOCIAL MEDIA — CDNs + messaging merged ────────────────────
    "fbcdn.net":             ("facebook",    "social_media"),
    "fbsbx.com":             ("facebook",    "social_media"),
    "facebook.net":          ("facebook",    "social_media"),
    "cdninstagram.com":      ("instagram",   "social_media"),
    "twimg.com":             ("twitter",     "social_media"),
    "t.co":                  ("twitter",     "social_media"),
    "x.com":                 ("twitter",     "social_media"),
    "sc-cdn.net":            ("snapchat",    "social_media"),
    "sc-gw.com":             ("snapchat",    "social_media"),
    "snapchat.com":          ("snapchat",    "social_media"),
    "licdn.com":             ("linkedin",    "social_media"),
    "linkedin.com":          ("linkedin",    "social_media"),
    "redditmedia.com":       ("reddit",      "social_media"),
    "redditstatic.com":      ("reddit",      "social_media"),
    "redd.it":               ("reddit",      "social_media"),
    "redditspace.com":       ("reddit",      "social_media"),
    "reddit.com":            ("reddit",      "social_media"),
    "pinimg.com":            ("pinterest",   "social_media"),
    "pinterest.com":         ("pinterest",   "social_media"),
    "discord.com":           ("discord",     "social_media"),
    "discordapp.com":        ("discord",     "social_media"),
    "giphy.com":             ("giphy",       "social_media"),
    # Messaging merged into social_media
    "whatsapp.net":          ("whatsapp",    "social_media"),
    "whatsapp.com":          ("whatsapp",    "social_media"),
    "signal.org":            ("signal",      "social_media"),
    "slack.com":             ("slack",       "social_media"),
    "slackb.com":            ("slack",       "social_media"),
    "telegram.org":          ("telegram",    "social_media"),

    # ── CONFERENCING — real-time calls only ───────────────────────
    "zoom.us":               ("zoom",        "conferencing"),
    "zoomgov.com":           ("zoom",        "conferencing"),
    "zoom.com":              ("zoom",        "conferencing"),
    "webex.com":             ("webex",       "conferencing"),
    "webexapis.com":         ("webex",       "conferencing"),
    "wbx2.com":              ("webex",       "conferencing"),
    "skype.com":             ("skype",       "conferencing"),
    "sfx.ms":                ("teams",       "conferencing"),
    "teams.microsoft.com":   ("teams",       "conferencing"),

    # ── CLOUD — storage, productivity, dev tools, updates ─────────
    "icloud.com":            ("icloud",      "cloud"),
    "apple-cloudkit.com":    ("icloud",      "cloud"),
    "cdn-apple.com":         ("apple",       "cloud"),
    "mzstatic.com":          ("apple",       "cloud"),
    "onedrive.com":          ("onedrive",    "cloud"),
    "sharepoint.com":        ("sharepoint",  "cloud"),
    "microsoftonline.com":   ("microsoft",   "cloud"),
    "live.net":              ("onedrive",    "cloud"),
    "office.com":            ("microsoft",   "cloud"),
    "office.net":            ("microsoft",   "cloud"),
    "azureedge.net":         ("azure",       "cloud"),
    "googleapis.com":        ("google",      "cloud"),
    "dropbox.com":           ("dropbox",     "cloud"),
    "dropboxstatic.com":     ("dropbox",     "cloud"),
    "amazonaws.com":         ("aws",         "cloud"),
    "github.com":            ("github",      "cloud"),
    "githubusercontent.com": ("github",      "cloud"),
    "githubcopilot.com":     ("copilot",     "cloud"),
    "visualstudio.com":      ("vscode",      "cloud"),
    "vscode-cdn.net":        ("vscode",      "cloud"),
    "cursor.sh":             ("cursor",      "cloud"),
    "notion.so":             ("notion",      "cloud"),
    "figma.com":             ("figma",       "cloud"),
    "overleaf.com":          ("overleaf",    "cloud"),
    "canva.com":             ("canva",       "cloud"),
    # Software updates merged into cloud
    "windowsupdate.com":     ("msupdate",    "cloud"),
    "brave.com":             ("brave",       "cloud"),
    "mcafee.com":            ("mcafee",      "cloud"),
    "avast.com":             ("avast",       "cloud"),
    "kaspersky.com":         ("kaspersky",   "cloud"),
    "kaspersky-labs.com":    ("kaspersky",   "cloud"),
    "mozilla.org":           ("firefox",     "cloud"),
    "mozilla.com":           ("firefox",     "cloud"),
    "ubuntu.com":            ("ubuntu",      "cloud"),
    "snapcraft.io":          ("ubuntu",      "cloud"),
    "nvidia.com":            ("nvidia",      "cloud"),
    "opera.com":             ("opera",       "cloud"),

    # ── WEB — search, AI, news, ecommerce, browsing ──────────────
    "bing.com":              ("bing",        "web"),
    "bing.net":              ("bing",        "web"),
    "duckduckgo.com":        ("duckduckgo",  "web"),
    "chatgpt.com":           ("chatgpt",     "web"),
    "openai.com":            ("openai",      "web"),
    "perplexity.ai":         ("perplexity",  "web"),
    "claude.ai":             ("claude",      "web"),
    "anthropic.com":         ("claude",      "web"),
    "deepseek.com":          ("deepseek",    "web"),
    "grammarly.com":         ("grammarly",   "web"),
    "grammarly.io":          ("grammarly",   "web"),
    "grammarly.net":         ("grammarly",   "web"),
    "quillbot.com":          ("quillbot",    "web"),
    "piazza.com":            ("piazza",      "web"),
    "quora.com":             ("quora",       "web"),
    "wikimedia.org":         ("wikipedia",   "web"),
    "duolingo.com":          ("duolingo",    "web"),
    "mathworks.com":         ("mathworks",   "web"),
    "kaggle.io":             ("kaggle",      "web"),
    "kaggle.net":            ("kaggle",      "web"),
    "codeforces.com":        ("codeforces",  "web"),
    "leetcode.com":          ("leetcode",    "web"),
    "hackmd.io":             ("hackmd",      "web"),
    "indianexpress.com":     ("news",        "web"),
    "news18.com":            ("news",        "web"),
    "livemint.com":          ("news",        "web"),
    "tradingview.com":       ("tradingview", "web"),
    # Ecommerce merged into web
    "flipkart.com":          ("flipkart",    "web"),
    "flipkart.net":          ("flipkart",    "web"),
    "flixcart.com":          ("flipkart",    "web"),
    "myntra.com":            ("myntra",      "web"),
    "myntassets.com":        ("myntra",      "web"),
    "ajio.com":              ("ajio",        "web"),
    "meesho.com":            ("meesho",      "web"),
    "meeshoapi.com":         ("meesho",      "web"),
    "swiggy.com":            ("swiggy",      "web"),
    "zomato.com":            ("zomato",      "web"),
    "zmtcdn.com":            ("zomato",      "web"),
    "blinkit.com":           ("blinkit",     "web"),
    "grofers.com":           ("blinkit",     "web"),
    "makemytrip.com":        ("makemytrip",  "web"),
    "bookmyshow.com":        ("bookmyshow",  "web"),
    "bmscdn.com":            ("bookmyshow",  "web"),
    "paytm.com":             ("paytm",       "web"),
    "paytmbank.com":         ("paytm",       "web"),
    "phonepe.com":           ("phonepe",     "web"),
    "juspay.in":             ("juspay",      "web"),
    "amazon.in":             ("amazon",      "web"),
    "media-amazon.com":      ("amazon",      "web"),
    "ssl-images-amazon.com": ("amazon",      "web"),

    # ── GAMING — game servers and pure gaming platforms ───────────
    "steamserver.net":       ("steam",       "gaming"),
    "steamcontent.com":      ("steam",       "gaming"),
    "steampowered.com":      ("steam",       "gaming"),
    "xboxlive.com":          ("xbox",        "gaming"),
    "gamepass.com":          ("xbox",        "gaming"),
    "epicgames.com":         ("epic",        "gaming"),
    "epicgames.dev":         ("epic",        "gaming"),
    "riotgames.com":         ("riot",        "gaming"),
    "pvp.net":               ("riot",        "gaming"),
    "ea.com":                ("ea",          "gaming"),
    "supercell.com":         ("supercell",   "gaming"),
    "lunarclient.com":       ("minecraft",   "gaming"),
    "chess.com":             ("chess",       "gaming"),
}

CLASSES      = ["streaming", "social_media", "conferencing", "cloud", "web", "gaming"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def domain_to_labels(domain: str):
    """
    Longest-suffix match.
    Returns (provider, category) or None.

    e.g.
      "rr3.sn-npoe7n.googlevideo.com" -> ("youtube", "streaming")
      "apresolve.spotify.com"         -> ("spotify", "streaming")
      "web.whatsapp.com"              -> ("whatsapp", "social_media")
      "catalog.gamepass.com"          -> ("xbox",     "gaming")
      "wpad.iitd.ac.in"               -> None
    """
    if not domain:
        return None
    domain = domain.lower().rstrip(".")
    if domain in DOMAIN_MAP:
        return DOMAIN_MAP[domain]
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in DOMAIN_MAP:
            return DOMAIN_MAP[suffix]
    return None


if __name__ == "__main__":
    # Verification test — run this to confirm all streaming services work
    tests = [
        ("rr3.sn-npoe7n.googlevideo.com", "streaming", "YouTube CDN"),
        ("apresolve.spotify.com",          "streaming", "Spotify API"),
        ("ap-gae2.spotify.com",            "streaming", "Spotify CDN"),
        ("ipv4-c001.nflxvideo.net",        "streaming", "Netflix CDN"),
        ("hotstarext.com",                 "streaming", "Hotstar CDN"),
        ("api.hotstar.com",                "streaming", "Hotstar API"),
        ("vowifi.jio.com",                 "streaming", "Jio"),
        ("www.primevideo.com",             "streaming", "Prime Video"),
        ("web.whatsapp.com",               "social_media", "WhatsApp"),
        ("graph.facebook.com",             "social_media", "Facebook"),  # via fbcdn fallback? No - facebook.com excluded
        ("zoom.us",                        "conferencing", "Zoom"),
        ("amazonaws.com",                  "cloud",        "AWS"),
        ("catalog.gamepass.com",           "gaming",       "Xbox"),
        ("wpad.iitd.ac.in",                None,           "IITD internal"),
    ]

    print(f"{'Domain':45s} {'Expected':15s} {'Got':15s} {'Status'}")
    print("-" * 90)
    all_pass = True
    for domain, expected_cat, label in tests:
        result = domain_to_labels(domain)
        got_cat = result[1] if result else None
        ok = got_cat == expected_cat
        if not ok:
            all_pass = False
        status = "✅" if ok else "❌ FAIL"
        print(f"{domain:45s} {str(expected_cat):15s} {str(got_cat):15s} {status}  {label}")

    print()
    print("✅ ALL PASS" if all_pass else "❌ SOME FAILED — fix before running pipeline")
