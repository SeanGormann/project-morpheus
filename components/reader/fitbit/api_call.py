import requests
import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import pytz


# Load environment variables from .env file
load_dotenv()

today = datetime.date.today().strftime("%Y-%m-%d")
yesterday = datetime.date.today() - datetime.timedelta(days=1)

access_token = os.getenv('TOKEN')
user_id = os.getenv('USER_ID')


header = {'Authorization':'Bearer ' + access_token}

response = requests.get(f'https://api.fitbit.com/1.2/user/-/sleep/date/{today}.json',
                         headers=header)


data = response.json()


# Extract time series entries
sleep_entries = data.get("sleep", [])

# Collect all data points
stage_records = []

for record in sleep_entries:
    for stage in record["levels"]["data"]:
        stage_records.append({
            "timestamp": pd.to_datetime(stage["dateTime"]),
            "stage": stage["level"],
            "duration": stage["seconds"],
            "date": record["dateOfSleep"]
        })

# Create DataFrame
df = pd.DataFrame(stage_records)

# Optional: convert to local timezone
local_tz = pytz.timezone("Europe/London")  # Change to your local timezone
df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)

print(df.head())

df = df.to_csv('may14th.csv') 




