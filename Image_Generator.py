from gradio_client import Client
from PIL import Image
import requests
from io import BytesIO
import time
import threading
import itertools
import sys
import os

# ──────────────────────────────
# 🎨 ASCII-BANNER
# ──────────────────────────────
banner = r"""
██████╗ ██████╗ ███████╗███████╗███████╗██╗   ██╗██╗   ██╗██╗███╗   ███╗ █████╗  ██████╗ ██╗███████╗
██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║   ██║██║   ██║██║████╗ ████║██╔══██╗██╔════╝ ██║██╔════╝
██████╔╝██████╔╝███████╗███████╗█████╗  ██║   ██║██║   ██║██║██╔████╔██║███████║██║  ███╗██║███████╗
██╔═══╝ ██╔══██╗╚════██║╚════██║██╔══╝  ╚██╗ ██╔╝██║   ██║██║██║╚██╔╝██║██╔══██║██║   ██║██║╚════██║
██║     ██║  ██║███████║███████║███████╗  ╚████╔╝ ╚██████╔╝██║██║ ╚═╝ ██║██║  ██║╚██████╔╝██║███████║
╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═══╝   ╚═════╝ ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝
"""
print(banner)
print("🧠 coded by BreckzTv\n")

# ──────────────────────────────
# MODEL EINSTELLUNG
# ──────────────────────────────
client = Client("black-forest-labs/FLUX.1-schnell")

def ladeanimation(stop_event):
    """Rotierende Ladeanimation"""
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r⏳ Generiere Bild... bitte warten... {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r✅ Generierung abgeschlossen!            \n')

while True:
    prompt = input("🖋️ Gib deinen Prompt ein (oder 'exit' zum Beenden): ")
    if prompt.lower() == "exit":
        print("👋 Programm beendet.")
        break

    # Ladeanimation starten
    stop_event = threading.Event()
    t = threading.Thread(target=ladeanimation, args=(stop_event,))
    t.start()

    # Anfrage an Modell
    result = client.predict(
        prompt=prompt,
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1024,
        num_inference_steps=4,
        api_name="/infer"
    )

    # Ladeanimation stoppen
    stop_event.set()
    t.join()

    # Ergebnis analysieren
    result_item = result[0] if isinstance(result, (list, tuple)) else result
    file_path = result_item[0] if isinstance(result_item, tuple) else result_item

    # Wenn lokale Datei vorhanden → direkt öffnen
    if os.path.exists(file_path):
        image = Image.open(file_path)
    else:
        response = requests.get(file_path)
        image = Image.open(BytesIO(response.content))

    # Bild anzeigen
    image.show()

    # Zeitstempel für eindeutigen Dateinamen
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = f"flux_output_{timestamp}.png"

    # Bild speichern
    image.save(output_path)
    print(f"💾 Bild gespeichert als {output_path}\n")
