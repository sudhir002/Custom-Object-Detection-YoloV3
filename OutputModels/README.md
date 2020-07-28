# Wrist-Detection-TF-2.0-YoloV3

• Requirments:___
  Tensorflow 2.0
  OpenCv 4
  keras
  
• Train Your Data
  1.  Download pre trained YoloV3 mode : https://pjreddie.com/media/files/yolov3.weights
  2.  Run Convert.py to convert this weight to tf weight
  3.  Test your converted model working with - Run Detect_Video.py
  
  • Annotate your object with some annotation tool like : labelImg, imglab etc
  
    From here you will get XML file
    
      4.  Run XmlToTfRecord.py to generate trrecord file
          In his file keep split type train and test
      5.  Finally change some variables in Train.py file according to ou and run it.
      6.  For etection on camera Run Detect_video.py with your trained model.
    
    
    
    
   Feel free to ask something,
 
