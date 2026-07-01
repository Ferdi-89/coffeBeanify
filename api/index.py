import sys
import os

# Tambahkan direktori root dan backend ke python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend'))

# Import Flask app dari main_flask
from backend.main_flask import app

# Vercel mencari variabel bernama 'app'
app = app
