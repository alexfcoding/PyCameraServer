# python localFiles.py -i 192.168.0.12 -o 8000

import os
import argparse
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import sys
import subprocess

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('start_analysis',prt=8001, filee=filename))
            #return f"Файл {filename} загружен. Запускаю сервер обработки..."
			#return redirect(url_for('video_feed',prt=8000))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Выберите видеофайл для загрузки</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Загрузить>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route("/<prt>,<filee>")
def start_analysis(prt,filee):
    # return Response(generate(),
    # 	mimetype = "multipart/x-mixed-replace; boundary=frame")
	
	#os.system(f"python localFiles.py -i 192.168.0.12 -o {prt} -s {filee}")
	subprocess.Popen([f'python', 'localFiles.py', '-i', "192.168.0.12", '-o', str(prt), '-s', str(filee)])

	#return f"Обработка доступна по адресу: http://192.168.0.12:{prt}"
	return redirect(f"http://178.140.230.247:{prt}")
	
	#return os.system(f"python localFiles.py -i 192.168.0.12 -o {prt} -s {filee}")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
