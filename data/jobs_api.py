import http.client

conn = http.client.HTTPSConnection("api.francetravail.io")

headers = {
    'Authorization': "Bearer PaRZuzeXygIHLA2KZ7Dp5mfp1_U",
    'Accept': "application/json"
}

conn.request("GET", "/partenaire/offresdemploi/v2/offres/search?motsCles=data+science", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))

with open('api_response.json', 'wb') as outf:
    outf.write(data)

print(data.decode("utf-8"))