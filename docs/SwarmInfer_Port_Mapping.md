# SwarmInfer: Port-to-Board Mapping

**Generated:** 2026-04-05
**Method:** Auto-detected by flashing get_mac firmware to each port

## Port Registry

| USB Port | MAC Address | Board | Role |
|----------|-------------|-------|------|
| `/dev/cu.usbmodem212301` | B8:F8:62:E2:D0:8C | Board 0 | **Coordinator** |
| `/dev/cu.usbmodem212401` | B8:F8:62:E2:D1:98 | Board 1 | **Worker 1** |
| `/dev/cu.usbmodem212201` | B8:F8:62:E2:CD:E4 | Board 2 | **Worker 2** |
| `/dev/cu.usbmodem2121401` | B8:F8:62:E2:DA:28 | Board 3 | **Worker 3** |
| `/dev/cu.usbmodem2121301` | B8:F8:62:E2:D0:DC | Board 4 | **Worker 4** |
| `/dev/cu.usbmodem2121201` | B8:F8:62:E2:C7:30 | Board 5 | **Monitor/Spare** |

## Quick Flash Commands

```bash
# Activate ESP-IDF first:
source "$IDF_PATH/export.sh"

# Flash coordinator (Board 0):
idf.py -p /dev/cu.usbmodem212301 flash

# Flash workers (Boards 1-4):
idf.py -p /dev/cu.usbmodem212401 flash    # Worker 1
idf.py -p /dev/cu.usbmodem212201 flash    # Worker 2
idf.py -p /dev/cu.usbmodem2121401 flash   # Worker 3
idf.py -p /dev/cu.usbmodem2121301 flash   # Worker 4

# Flash monitor/spare (Board 5):
idf.py -p /dev/cu.usbmodem2121201 flash   # Monitor

# Monitor coordinator:
idf.py -p /dev/cu.usbmodem212301 monitor

# Kill stuck serial processes:
kill $(lsof -t /dev/cu.usbmodem212301) 2>/dev/null
```

## Notes

- Port names may change if boards are reconnected to different USB hub ports
- Always run `ls /dev/cu.usb*` to verify current ports
- If a port is busy: `kill $(lsof -t /dev/cu.usbmodemXXXXX) 2>/dev/null`
- Flash and monitor must be run as **separate** commands (not `idf.py -p PORT flash monitor`)
