from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
#from werkzeug.datastructures import  FileStorage
import errno
import os
from datetime import datetime
from pyresparser import ResumeParser

app = Flask(__name__, template_folder='template')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file11():
   if request.method == 'POST':
      #f = request.files['file']
      files = request.files.getlist("file")
      mydir = os.path.join(
        os.getcwd(), 'resume', 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
      resp = []
      for f in files: 
        os.makedirs(mydir, mode=0o777, exist_ok=True)
        fpp = os.path.join(mydir, secure_filename(f.filename))
        f.save(fpp)
        resp.append(ResumeParser(fpp).get_extracted_data())
      return resp
   else:
      return render_template('upload.html')

@app.route('/storage', defaults={'req_path': 'resume'})
@app.route('/storage<path:req_path>')
def dir_listing(req_path):
    BASE_DIR = os.getcwd()
    abs_path = os.path.join(BASE_DIR, req_path)
    if not os.path.exists(abs_path):
        return abort(404)
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    #files = os.listdir(abs_path)
    files = []
    with os.scandir(abs_path) as it:
        for entry in it:
            if entry.is_file():
                print('reached..');
                files.append({ filepath : entry.path, filesize : entry.stat().st_size })
    return render_template('files.html', len = len(files), files=files)