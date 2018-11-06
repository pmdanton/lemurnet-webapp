from flask import Flask, render_template, url_for
from flask_sslify import SSLify

app = Flask(__name__)
sslify = SSLify(app, permanent=True)

@app.route('/')
def index():
    return render_template('index.html')

if(__name__ == "__main__"):
    app.run()
