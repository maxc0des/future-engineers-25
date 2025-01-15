#!/bin/bash

networks=$(sudo iwlist wlan0 scan | grep 'ESSID' | sed 's/.*ESSID:"\(.*\)".*/\1/')

first_network="Netzwerk1"
second_network="Netzwerk2"

first_password="Passwort1"
second_password="Passwort2"

if echo "$networks" | grep -q "$first_network"; then
    nmcli dev wifi connect "$first_network" password "$first_password"
else
    if echo "$networks" | grep -q "$second_network"; then
        nmcli dev wifi connect "$second_network" password "$second_password"
    else
        echo "Netzwerke nicht gefunden."
    fi
fi

if [ $? -eq 0 ]; then
    echo "Erfolgreich mit dem Netzwerk verbunden!"
else
    echo "Verbindung mit dem Netzwerk fehlgeschlagen."
fi