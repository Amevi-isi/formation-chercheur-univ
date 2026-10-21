import socket

import uvicorn


if __name__ == "__main__":
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        local_ip = "adresse-de-la-machine"
    print("\nLaboratoire IA")
    print("Sur cet ordinateur : http://127.0.0.1:8000")
    print(f"Sur le même réseau Wi-Fi : http://{local_ip}:8000\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
