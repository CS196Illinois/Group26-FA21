from math import factorial as fact
import requests

print("factorial of 4: ", fact(4))
new_data = requests.get("https://api.covidtracking.com/v2/us/daily.json")
#print(new_data.json())
