import httpx

r = httpx.get("https://newsapi.org/v2/everything", params={
    "q": "Apple",
    "apiKey": "YOUR_KEY",
})

print(r.status_code)
print(r.text[:500])