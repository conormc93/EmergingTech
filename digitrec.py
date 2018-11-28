from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


# Main method.
if __name__ == "__main__":
    # Run the app.
    app.run()
