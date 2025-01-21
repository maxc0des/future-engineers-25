import json
import subprocess

def wlan_connect(ssid, password):
    # Führe den nmcli Befehl aus, um mit dem WLAN zu verbinden
    try:
        result = subprocess.run(
            ['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(f"Erfolgreich mit {ssid} verbunden.")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Verbinden mit {ssid}: {e.stderr}")



with open('credentials.json', 'r') as file:
    data = json.load(file)

ssid = data['ssid']
password = data['password']
wlan_connect(ssid, password)