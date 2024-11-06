#!/bin/bash

pip install --no-cache-dir -r requirements.txt
uvicorn app.app:app --host 127.0.0.1 --port 80