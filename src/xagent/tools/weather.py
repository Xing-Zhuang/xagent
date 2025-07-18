import os
import requests

def get_weather(city:str):
    key = os.environ['AMAP_API_KEY']
    url =  "https://restapi.amap.com/v3/weather/weatherInfo"
     
    params = {
        'key': key,
        'city': city,
        'extensions': 'all' 
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        data = response.json()

        if data['count'] != '0' :
            info = ""
            for d in data['forecasts'][0]['casts']:
                items = [f"{k}={v}" for k, v in d.items()]
                items = ', '.join(items)
                info = info + items + "\n"
            
            return info
        
        return f"Failed to retrieve {city} data.Please check if the city is correct."
    else:
        return f"Failed to retrieve weather data. Status code: {response.status_code}"
    

docs_get_weather = """
Get the weather forecast for a given city.

Args:
    city(str): City name or city code.
"""