from flask import Flask,render_template,request
from common import get_tensor, get_output
import os

app =  Flask(__name__)

@app.route("/",methods = ['GET','POST'])
def hello():
    if request.method == 'GET':
        return render_template("index.html",value="hello")
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            return '<h3>File Not Uploaded</h3>'
        image = file.read()
        prediction = get_output(image_bytes=image)
        return render_template('result.html', flower=prediction)

if __name__ == "__main__":
    app.run(debug=True,port=(os.getenv('PORT',8000)))
