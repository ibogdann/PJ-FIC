#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#include <opencv2\highgui.h>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2\cv.h>
#include "opencv2/opencv.hpp"
#include<string.h>    //strlen
#include<sys/socket.h>    //socket
#include<arpa/inet.h> //inet_addr
#include<netdb.h> //hostent
#include <unistd.h>

using namespace std;
using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
/*int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;*/

int H_MIN = 29;
int H_MAX = 256;
int S_MIN = 65;
int S_MAX = 256;
int V_MIN = 165;
int V_MAX = 256;
// 145 H value and 35 S value for pink only
// 21 H and 65 S and 165 V for pink and yellow

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const std::string windowName = "Original Image";
const std::string windowName1 = "HSV Image";
const std::string windowName2 = "Thresholded Image";
const std::string windowName3 = "After Morphological Operations";
const std::string trackbarWindowName = "Trackbars";


void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}

string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars


	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf(TrackbarName, "H_MIN", H_MIN);
	sprintf(TrackbarName, "H_MAX", H_MAX);
	sprintf(TrackbarName, "S_MIN", S_MIN);
	sprintf(TrackbarName, "S_MAX", S_MAX);
	sprintf(TrackbarName, "V_MIN", V_MIN);
	sprintf(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25 < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25 > 0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25 < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
	//cout << "x,y: " << x << ", " << y;

}
void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area > MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				//cout << x << "," << y;
				drawObject(x, y, cameraFeed);

			}


		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

void  point(Mat &HSV,Mat &threshold,bool useMorphOps){

	inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MIN),threshold);
	if(useMorphOps)
		morphOps(threshold);

}
int xold,xnew,yold,ynew,xtarget,ytarget,turned;  //o-old; n-new; t-target
char c[10] = "sfbrl";
int sock;
struct sockaddr_in server;
char message[1000] , server_reply[2000];
void win_game(){
 for (int i = 0; i < 3; i++)
	{
		sprintf(message,"%c",c[1]);
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  
        	}
		sleep(1);
   }
   
   while(1)
   {
     sprintf(message,"%c",c[1]);
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  
        	}
         sprintf(message,"%c",c[3]);
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  
        	}
		//sleep(1);
   sprintf(message,"%c",c[3]);
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  
        	}
		sleep(1);
   
   }
}

int send_move(char *c)
{
  sprintf(message,"%c",c);
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  return 1;
        	}
		sleep(1);
   sprintf(message,"%c",'s');
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  return 1;
        	}
		sleep(1);
}

int main(int argc, char* argv[])
{

  

	//some boolean variables for different functionality within this
	//program
	bool trackObjects = true;
	bool trackObjects2 =  true;
	bool useMorphOps = true;

	Point p;
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	//x and y values for the location of the object
	int x = 0, y = 0;
	//create slider bars for HSV filtering
	createTrackbars();
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	capture.open("rtmp://172.16.254.99/live/nimic");
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop


//first of all take initial coords
    //store image to matrix
		capture.read(cameraFeed);
		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		//filter HSV image between values and store filtered image to
		//threshold matri
		inRange(HSV, Scalar(19, 110, 0), Scalar(166,236,256), threshold);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if (useMorphOps)
			morphOps(threshold);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		//point(HSV,threshold,useMorphOps);
		if (trackObjects){
			trackFilteredObject(x, y, threshold, cameraFeed);
		}
   xold=x;
   yold=y;
boolean pink = true;
	while (1) {
    xold=xnew;
    yold=ynew;
		//store image to matrix
		capture.read(cameraFeed);
    if(cameraFeed.empty())continue;
		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		//filter HSV image between values and store filtered image to
		//threshold matri
		//inRange(HSV, Scalar(19, 110, 0), Scalar(166,236,256), threshold);
		inRange(HSV, Scalar(H_MIN, H_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		
		/*
		if (pink){
			inRange(HSV, Scalar(144, 35, 0), Scalar(H_MAX, S_MAX, V_MAX), threshold);
			pink = false;
		}else{
			pink = true;
			inRange(HSV, Scalar(21, 65, 165), Scalar(H_MAX, S_MAX, V_MAX), threshold);
		}
*/
		if (useMorphOps)
			morphOps(threshold);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		//point(HSV,threshold,useMorphOps);
		if (trackObjects){
			trackFilteredObject(x, y, threshold, cameraFeed);
		}
    xnew=x;
    ynew=y;
    printf("%d,%d\n",x,y);
		/*inRange(HSV,Scalar(19,110,0),Scalar(256,236,256),threshold);
		if(useMorphOps)
			morphOps(threshold);
		if(trackObjects2)
			trackFilteredObject(x,y,threshold,cameraFeed);
    xtarget=x;
    ytarget=y;
    printf("%d,%d\n\n\n",x,y); 
		//show frames*/
		imshow(windowName2, threshold);
		imshow(windowName, cameraFeed);
		//imshow(windowName1, HSV);
		setMouseCallback("Original Image", on_mouse, &p);
		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
    if(xnew!=xtarget)
    {
      turned=0;
      //if(xnew<xtarget)
        //send_move("f");
      //else
        //send_move("b");
    }
    else
    {
      turned=1;
    }
		waitKey(30);
	}
	return 0;
}
/*
int main(int argc , char *argv[])
{
    
     
    //Create socket
    sock = socket(AF_INET , SOCK_STREAM , 0);
    if (sock == -1)
    {
        printf("Could not create socket");
    }
    puts("Socket created");
     
    server.sin_addr.s_addr = inet_addr("193.226.12.217");
    server.sin_family = AF_INET;
    server.sin_port = htons( 20232 );
 
    //Connect to remote server
    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0)
    {
        perror("connect failed. Error");
        return 1;
    }
     
    puts("Connected\n");
     
     
    //keep communicating with server
	  int ok=1;
    while(ok==1)
    {
        printf("Enter message : ");
        scanf("%s" , message);
      //  for (int i = 0; i < strlen(c); i++)
    //{
		//cout << c[i];
		//if((c[i]=='f')||(c[i]=='s')||(c[i]=='r')||(c[i]=='l'))
		//{
		sprintf(message,"%c",c[i]);
		if( send(sock , message , strlen(message) , 0) < 0)
        	{
            	  puts("Send failed");
            	  return 1;
        	}
		sleep(1);
		}
	}
        //Send some data
        ok=0;
    }
    
    //close(sock);
    return 0;
}
*/
