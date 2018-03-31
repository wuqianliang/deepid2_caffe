# coding:utf-8
import pymysql as mdb
import time
import os
import base64
import face_verify_statics_sphereface as fv
import cv2
import numpy as np
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import threading

imgFiles = []
rets = {}

feature_arr = {}
threshold = 0.4


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return "hello"

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        starttime = time.time()
        filenum = request.values['filenum']
        #print range(int(filenum))
	filelist = []
        global rets
        for i in range(int(filenum)):
            f = request.files['file'+str(i)]
	    basepath = os.path.dirname(__file__)  # 当前文件所在路径
            upload_path = os.path.join(basepath, 'uploads',secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
            f.save(upload_path)
	    filelist.append(upload_path)
	    #print 'file type:',type(f) 
            #processIMG(file)



	for filename in filelist:
	    processIMG(filename)
	    #print filename
        if len(rets) > 0:
	    updatecache(rets)
	else:
	 
            conn = mdb.connect(host='localhost', user='root', passwd='123456', db='facerec', charset='utf8')
    	    cursor = conn.cursor()
    	    cursor.execute("delete from info")
            conn.commit()
	rets = {}


        #print time.time() - starttime
        #return redirect(url_for('upload'))

    return "HTTP/1.1 200 OK\r\n\r\n"

def processIMG(img):
    
    feat = fv.get_feature_crop_align(img)
	
    #print feat.shape

    if feat is not None:
	for k in feature_arr:
	    f1 = np.array(feat[0])
	    f2=np.array(feature_arr[k][0])
	
	    lx = np.sqrt(f1.dot(f1))
	    ly = np.sqrt(f2.dot(f2))
	    cos =f1.dot(f2)/(lx*ly)
    	    print 'ID:',k,',Similar:',str(abs(cos))
	    #if abs(cos/0.5) > threshold:
	    if abs(cos) > threshold:
	    	rets[k] = str(abs(cos))#str(abs(cos/0.5)*100)

def file_exists(filename):
    try:
        with open(filename) as f:
            return True
    except IOError:
        return False

def fun_getimage():
    start = time.time()
    while True:
        if time.time() - start < 6:
            continue
        start = time.time()
        conn = mdb.connect(host='localhost', user='root', passwd='123456', db='facerec', charset='utf8')
        cursor = conn.cursor()
        aa = cursor.execute("select IDNumber,photo from idinfo")
        infos = cursor.fetchmany(aa)
        cursor.close()
        #print 'read database'
        for info in infos:
            imgfile = "./raw-images/" +  info[0] + ".jpg"
            if imgfile not in imgFiles:
                imgFiles.append(imgfile)
            if file_exists(imgfile):
                continue
            
            #print 'Add person:', imgfile
            fout = open(imgfile, 'wb')
            fout.write(base64.b64decode(info[1]))

            fout.close()
	
        #print time.time() - start

        conn.close()
    
        for imgname in imgFiles:

	    feat = fv.get_feature_crop_align(imgname)
	    if feat is not None:
	        feature_arr[imgname.split("/")[-1].split(".")[0]]=feat
#	break


locateinfo = "南京市长江大桥"


def updatecache(persons):
    #conn = mdb.connect(host='192.168.0.214', user='root', passwd='123456', db='FaceRec', charset='utf8')
    conn = mdb.connect(host='localhost', user='root', passwd='123456', db='facerec', charset='utf8')
    cursor = conn.cursor()
    cursor.execute("delete from info")
    conn.commit()

    for key in persons:
        sqltext = "insert into info(IDNumber,LocateInfo,Similarity) values(%s,%s,%s)"
        cursor.execute(sqltext, (key, locateinfo, str(persons[key])))
        conn.commit()

    cursor.close()
    conn.close()



if __name__ == "__main__":
    t = threading.Thread(target=fun_getimage)
    #t.setDaemon(True)
    t.start()
    app.run(host='', port=8008)


    
    
	    
    
   























    
