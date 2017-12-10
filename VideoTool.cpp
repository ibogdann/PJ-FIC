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
#include <math.h>

using namespace std;
using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

/*int H_MIN = 29;
int H_MAX = 256;
int S_MIN = 65;
int S_MAX = 256;
int V_MIN = 165;
int V_MAX = 256;*/
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
int xold,xnew,yold,ynew,xtarget,ytarget,turned,last_move;
char c[10] = "sfbrl";
int sock;
struct sockaddr_in server;
char message[1000] , server_reply[2000];


int send_move(char &c,int slp)
{
    sprintf(message,"%c",c);
    if( send(sock , message , strlen(message) , 0) < 0)
       	{
           	  puts("Send failed");
           	  return 1;
       	}
	sleep(slp);
}

void win_game()
{
    //backup mode: just go in loop and hope that opponent gets out by trying to hit you
    int i=0;
    while(i<11)
    {
        send_move(c[1],2);
        send_move(c[3],2);
        send_move(c[0],1);
        i++;
    }
}


void battle()
{
    //engage battle mode: go on OX untill same coords then attack
    int dif,dir;
    if((abs(xtarget-xnew)<10)&&(abs(ytarget-ynew)<10))
    {
        send_move(c[last_move],5);
        send_move(c[0],1);
    }
    if(abs(xtarget-xnew)>10)
    {
        if(turned==1)
        {
            turned=0;
            send_move(c[4],4);
        }
        dif=abs(xtarget-xnew);
        dif/=10;
        if(xtarget>xnew)
        {
            send_move(c[1],dif);
            send_move(c[0],1);
        }
        else
        {
            send_move(c[2],dif);
            send_move(c[0],1);
        }
    }
    else
    {
        if(turned==0)
        {
            turned=1;
            send_move(c[3],4);
        }
        else
        {
            dif=abs(ytarget-ynew);
            dif/=20;
            if(ytarget>ynew)
            {
                send_move(c[1],dif);
                last_move=1;
                send_move(c[0],1);
            }
            else
            {
                send_move(c[2],dif);
                last_move=2;
                send_move(c[0],1);
            }
        }
    }
}

int calibrate()
{
    if(xold==0)
    {
        xold=xnew;
        yold=ynew;
        //last_move=1;
        send_move(c[1],1);
        send_move(c[0],1);
    }
    else
    {
        send_move(c[2],1);
        send_move(c[0],1);
        if(yold!=ynew)
        {
            if(xold==xnew) //it is facing up or down, need to do 90 degree turn
            {
                send_move(c[3],4);
                send_move(c[3],0);
            }
            else
            {
                send_move(c[3],1); //keep moving right by little untill reach moving up/down or left/right
                send_move(c[0],1);
                send_move(c[1],1);
                send_move(c[0],1);
            }
        }
        else
        {
            if(xnew<xold) //it is facing 0, from inf to 0 need to do a 180 degree turn
            {
                send_move(c[3],8);
                send_move(c[0],1);
            }
            return 1;
        }

    }
    return 0;
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

	//Create socket
    sock = socket(AF_INET , SOCK_STREAM , 0);
    if (sock == -1)
    {
        printf("Could not create socket!");
        return 1;
    }
    //puts("Socket created"); //for verification

    //Connect to robot
    server.sin_addr.s_addr = inet_addr("193.226.12.217");
    server.sin_family = AF_INET;
    server.sin_port = htons( 20232 );
    //Connect to remote server
    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0)
    {
        perror("connect failed. Error");
        return 1;
    }
    //puts("Connected\n"); for verification

    int ok=1;
    while(ok==1)
    {
        //store image to matrix
        capture.read(cameraFeed);
        if(cameraFeed.empty())continue;
        //convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		//filter HSV image between values and store filtered image to
		//threshold matrix

		//FIRST is OUR Robot
		inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold); //to be changed with values to identify our robot
		if (useMorphOps)
			morphOps(threshold);
		if (trackObjects){
			trackFilteredObject(x, y, threshold, cameraFeed);
		}
        xnew=x;
        ynew=y;

        //SECOND is target Robot
		inRange(HSV, Scalar(1, 115, 150), Scalar(197, 197, 256), threshold); //to be changed with values to identify target robot
		if(useMorphOps)
			morphOps(threshold);
		if(trackObjects2)
			trackFilteredObject(x,y,threshold,cameraFeed);
        xtarget=x;
        ytarget=y;

        if((xtarget==0)||(xnew==0))ok=0; //one of the robots have exited the camera feed (ring) => battle has finished
        else
        {
            //Battle mode: go on OX until differences are almoust 0 and then try to hit
            battle();

            //Don't care about anything anymore just enter defence mode to win the battle
            //win_game();

            //Calibrate to have the robot as we want, facing OX from 0 -> inf
            //while(calibrate()==0)calibrate();
        }

		//show frames
		imshow(windowName2, threshold);
		imshow(windowName, cameraFeed);
		//imshow(windowName1, HSV);
		setMouseCallback("Original Image", on_mouse, &p);
		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
        waitKey(30);
    }

	close(sock); //close socket
	return 0;
}
