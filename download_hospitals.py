
import requests
import json

URL = "https://adserr.com/hospitals/data.json"

print("Downloading Kerala hospital directory...")

response = requests.get(URL, timeout=60)
response.raise_for_status()

data = response.json()

with open("kerala_hospitals.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print("Download completed!")
print("Total hospitals:", len(data["hospitals"]))
print("Last checked:", data.get("checked"))
print("Saved as kerala_hospitals.json")
