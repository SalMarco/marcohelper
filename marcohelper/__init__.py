import os
import sys
import csv
import argparse
import logging
import traceback
from datetime import datetime
import time
from pprint import pprint as pp
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import magic
import gzip

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)

SUB_DEV=10

OUT_DIR="data"
DEF_NAME=".csv"

REMAP_DICT={'F':np.array([1]),'M':np.array([0]),'f':np.array([1]),'m':np.array([0])}
REVERSE_DICT={"1":'F',"0":"M"}

class Helper:
    """Class with some handy stuff"""

    def __init__(self):
        x=0


    def prepareProgress(**kargs):
        fi=kargs["fi"]
        next(fi)
        num_lines = sum(1 for line in fi)
        subdiv=num_lines // SUB_DEV
        fi.seek(0)
        next(fi)
        return fi, subdiv, num_lines

    def printProgress(**kargs):
        subdiv=kargs['subdiv']
        cur_line=kargs['cur_line']
        resto = cur_line % subdiv
        num= cur_line // subdiv
        if resto ==0:
            logger.info("Progress: [ %s%s ] "%("#"*num,"."*(10-num)))

    def printSimpleProgress(**kargs):
        if kargs["num"] % kargs["div"] == 0:
            logger.info("Progress: %s"%kargs["num"])


    def outfileName(**kargs):
        ret=dict()
        if not kargs["fo"]:
            fi=kargs['fi']
        else:
            fi=kargs['fo']
        if not fi:
            name=DEF_NAME
            path="%s/%s"%(os.path.dirname(os.path.realpath(__file__)),OUT_DIR)
        else:
            fi=fi.strip(".gz")
            name=os.path.basename(fi)
            path=os.path.dirname(fi)
        if "overwrite" in kargs and kargs["overwrite"]:
            name=DEF_NAME
        if "add_date" in kargs and kargs["add_date"]:
            date=datetime.now().strftime("%Y%m%d")
            split_name=name.split(".")
            if len(split_name[0]):
                name="%s_%s.%s"%(split_name[0],date,split_name[1])
            else:
                name="%s.%s"%(date,split_name[1])
        for k,pref in kargs['prefix_dict'].items():
            cur_name="%s_%s"%(pref,name)
            ret[k]=os.path.join(path,cur_name)
        return ret

    def simpleOutFileName(**kargs):
        fi = kargs['fi']
        path=os.path.dirname(fi)
        return os.path.join(path,kargs['def_name'])

    def personalLogger(**kargs):
        date=datetime.now().strftime("%Y%m%d_%H%M")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logging.basicConfig(format=formatter, level=logging.INFO)
        log_file="data/%s_%s.log"%(os.path.basename(kargs["script"]).split(".")[0],date)
        logger = logging.getLogger()
        if  'use_logfile' in kargs and kargs["use_logfile"]:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def splitDataset(sep,inFile,use_hashu):
        dataframe = read_csv(inFile, header=1, sep=sep)
        dataset = dataframe.values
        leng=dataset.shape[1]
        # split into input (X) and output (Y) variables
        if not use_hashu:
            X = dataset[:,2:].astype(np.uint8)
            Y = dataset[:,1].astype(str)
            label = dataset[:,0].astype(str)
            hashu = None
            logger.info("LOADED DATA FROM %s. SHAPES ARE: DATASET %s, X %s, Y %s, code %s",inFile,str(dataset.shape),str(X.shape),str(Y.shape),str(label.shape))
        else:
            print(dataset)
            X = dataset[:,3:].astype(np.uint8)
            Y = dataset[:,2].astype(str)
            hashu = dataset[:,1].astype(str)
            label = dataset[:,0].astype(str)
            logger.info("LOADED DATA FROM %s. SHAPES ARE: DATASET %s, X %s, Y %s, code %s, hashu %s",inFile,str(dataset.shape),str(X.shape),str(Y.shape),str(label.shape),str(hashu.shape))
        return X, Y, label, hashu, leng

    @staticmethod
    def LoadDataset(**kargs):
        logger.info("CREATION OF THE ARRAY FOR THE MODEL")
        inFile=kargs['fi']
        use_hashu = False
        if 'use_hashu' in kargs:
            use_hashu = kargs['use_hashu']
            logger.info('USED HASHU LABEL: %s',use_hashu)
        try:
            X, Y, label, hashu, leng = Helper.splitDataset(",",inFile,use_hashu)
        except:
            X, Y, label, hashu, leng = Helper.splitDataset(";",inFile,use_hashu)
        logger.info("ENCODING CLASS FOR Y")
        # le = LabelEncoder()
        # le.fit(Y)
        # logger.info("CLASSES: %s",str(le.classes_))
        for s in ["F","M",'f','m']:
            Y[Y==s]=REMAP_DICT[s]
        # encoded_Y = le.transform(Y)
        #return X, Y, label, leng, le
        return X, Y, label, hashu, leng, REVERSE_DICT


    @staticmethod
    def readHeader(**kargs):
        logger.info("READING HEADER FROM FILE ")
        with open(kargs["fi"],'r') as inp:
             header=inp.readline().split(';')
        num_col=len(header)
        logging.info("USING %i FEATURES"%(num_col-2))
        return header

    @staticmethod
    def checkFileType(**kargs):
        inFile=kargs['fi']
        ext=kargs['ext']
        file_type=magic.from_file(inFile,mime=True)
        if file_type=="application/gzip":
            fi = gzip.open(inFile,'%st'%ext)
        else:
            fi = open(inFile,ext)
        logger.info('FILE TYPE FOR %s IS %s',inFile,file_type)
        return fi

    @staticmethod
    def checkModelName(**kargs):
        name=os.path.basename(kargs['fi'])
        mod_type_f=name.split('_')[kargs['pos']]
        mod_type_h=os.path.splitext(mod_type_f)[0]
        mod_type=os.path.splitext(mod_type_h)[0]
        conf4type=kargs['types'][mod_type]
        logger.info("DETECTED MODEL TYPE: %s FROM WEIGHTS FILE %s: USING CONF %s",mod_type,name,conf4type)
        #return kargs['mod'].conf4type
        # return getattr(__import__(kargs['conf_file'], fromlist=[conf4type]), conf4type)
        return mod_type



# try:
#     ...
#     except Exception as e:
#       logger.error("msg:'%s', traceback:%s"%(str(e),traceback.format_exc()))
