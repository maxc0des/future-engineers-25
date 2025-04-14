#!/bin/bash

# Farben definieren
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color (zurücksetzen)

echo -e "${BLUE}[INFO]${NC} Aktiviere virtuelle Umgebung..."
source /home/robotics/Desktop/fe25_venv/bin/activate

echo -e "${BLUE}[INFO]${NC} Starte sensor-test.py..."
sensor_output=$(echo "test" | python3 sensor-test.py 2>&1)
sensor_status=$?

echo -e "${GREEN}[AUSGABE sensor-test.py]${NC}"
echo "$sensor_output"

if [ $sensor_status -ne 0 ]; then
  echo -e "${RED}[FEHLER] sensor-test.py fehlgeschlagen mit Status $sensor_status${NC}"
  exit $sensor_status
fi

echo -e "${BLUE}[INFO]${NC} Starte controller-test.py..."
python3 controller-test.py
controller_status=$?

if [ $controller_status -ne 0 ]; then
  echo -e "${RED}[FEHLER] controller-test.py fehlgeschlagen mit Status $controller_status${NC}"
  exit $controller_status
fi

echo -e "${GREEN}[FERTIG] Alles erfolgreich abgeschlossen.${NC}"
exit 0