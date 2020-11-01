#/usr/bin/env python

import os
import os.path
import sys
import string
import datetime, threading, time

## Shared variables
currentMutant = "";
currentPath = "";

## Paths
Path = os.path.abspath(os.path.join(os.getcwd(), os.pardir));
databasePath = Path+"/Database";
datasetPath = Path+"/Dataset";
dumpPath = Path+"/Dump";

## Package names
apps = ["aMetro","OpenCamera","A2DP_Volume","Sensorium","jamendo","Openbmap","Ushahidi","GTalk","DSub"];

packages = ["org.ametro","net.sourceforge.opencamera","a2dp.Vol","at.univie.sensorium","com.teleca.jamendo","org.openbmap","com.ushahidi.android.app","com.googlecode.gtalksms","github.daneren2005.dsub"];

testPackages = ["org.ametro.test", "net.sourceforge.opencamera.test", "a2dp.Vol.test", "at.univie.sensorium.test", "com.teleca.jamendo.test", "org.openbmap.activities", "com.ushahidi.android.app.test", "com.googlecode.gtalksms.test", "github.daneren2005.dsub.activity"];

class Threading(object):

	def __init__(self,interval,path,mutant):
		self.interval = interval;
		self.path = path;
		self.mutant = mutant;
		self.iteration = 0;
		thread = threading.Thread(target=self.run, args=())
		# Daemonize thread
        	thread.daemon = True                            
		# Start the execution
        	thread.start()                                  

    	def run(self):
		# Since the thread runs in the background, we set a limit of 127 so that it does not overwrite the after execution dump
       		while True and self.iteration < 127:
			self.iteration = self.iteration+1;
			dumpThread = threading.Thread(target=self.dumper);
			dumpThread.start()
			time.sleep(self.interval);

	def dumper(self):
		dump(self.iteration,self.path,self.mutant);


def startApp(path,mutant):
	os.system("adb install %s" % (path+"/"+ mutant));
	print "Starting the app: "+ mutant;

def stopApp(mutant):
	package = getPackage(mutant);
	os.system("adb shell pm uninstall -k %s" % (package));

def startLog(path):
	os.system("adb logcat Xposed:V *:S >> %s/log.txt &" % (path));

def stopLog():
	os.system("pkill adb logcat");

def getPackage(mutant):
	package = "";
	for app in apps:
		if app in mutant:
			index = apps.index(app);
			package = packages[index];
	return package;

def getTestPackage(mutant):
	package = "";
	for app in apps:
		if app in mutant:
			index = apps.index(app);
			package = testPackages[index];
	return package;

def executeTest(mutant,test,hardware,category,testType,datapoint):
	print "*******";
	currPath = "/"+hardware+"/"+category+"/"+testType+"/"+datapoint
	if not os.path.exists(dumpPath+currPath):
		os.mkdir(dumpPath+currPath);
	
	startApp(databasePath+currPath,mutant);
	startLog(dumpPath+currPath);

	# Dump the state before 
	dump(0,currPath,mutant);

	# Execute test
	testPackage = getTestPackage(mutant);
	command = testPackage+"."+test+" "+testPackage+"/android.test.InstrumentationTestRunner";
	Threading(0.1,currPath,mutant);
	os.system("adb shell am instrument -w -e class %s" % (command));
	
	
	# Dump the state after
	dump(128,currPath,mutant);

	stopLog();
	stopApp(mutant);
	print "*******";

def dump(count,path,mutant):
	path = dumpPath+path;
	path = path+"/"+str(count)+".txt";
	
	# dumpsys commands
	c1 = "adb shell dumpsys battery >> "+path;
	c2 = "adb shell dumpsys power  >> "+path;
	c3 = "adb shell dumpsys bluetooth_manager >> "+path;
	c4 = "adb shell dumpsys display >> "+path;
	c5 = "adb shell dumpsys location >> "+path;
	c6 = "adb shell dumpsys wifi >> "+path;
	c7 = "adb shell dumpsys connectivity >> "+path;
	c8 = "adb shell dumpsys telephony.registry  >> "+path;
	
	# ps commands
	c9 = "adb shell ps | grep "+getPackage(mutant);

	command = c1+" && "+c2+" && "+c3+" && "+c4+" && "+c5+" && "+c6+" && "+c7+" && "+c8+" && "+c9;
	os.system(command);
	

def main(argv):
	# install all the test apps (uncomment the following three lines before release)
	# muDroidTests = [f for f in os.listdir(Path+"/muDroid") if "Test" in f]
	# for muDroidTest in muDroidTests:
		# os.system("adb install %s" % (Path+"/muDroid/"+muDroidTest));

	hardwares = [f for f in os.listdir(databasePath) if not f.startswith('.')]
	for hardware in hardwares:
		print hardware;
		categories = os.listdir(databasePath+"/"+hardware);
		categories = [f for f in os.listdir(databasePath+"/"+hardware) if not f.startswith('.')]
		for category in categories:
			print category;
			testTypes = [f for f in os.listdir(databasePath+"/"+hardware+"/"+category) if not f.startswith('.')]
			for testType in testTypes:
				datapoints = [f for f in os.listdir(databasePath+"/"+hardware+"/"+category+"/"+testType) if not f.startswith('.')]
				for datapoint in datapoints:
					print datapoint;
					mutant = [f for f in os.listdir(databasePath+"/"+hardware+"/"+category+"/"+testType+"/"+datapoint) if f.endswith('.apk')];
					mutant = mutant[0];
					test = [f for f in os.listdir(databasePath+"/"+hardware+"/"+category+"/"+testType+"/"+datapoint) if f.endswith('.java')];
					test = test[0].split(".java")[0];
					executeTest(mutant,test,hardware,category,testType,datapoint);

if __name__ == "__main__": main(sys.argv[1:])
