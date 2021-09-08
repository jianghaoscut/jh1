#include <opencv2/opencv.hpp>
using namespace cv;
#include <iostream>
using namespace std;
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

Mat hsv_img; 
Mat threshold_img;
Mat mask;

 int LowH =0;
 int LowS=120;
 int LowV =245;

 int HighH = 30;
 int HighS = 255; 
 int HighV =255;

void draw_keypoint(Mat src, Mat dst)
{
	vector< RotatedRect> vc;
    vector< RotatedRect> vRec;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        // 求轮廓面积
        float Contour_Area = contourArea(contours[i]);

        // 去除较小轮廓&fitEllipse的限制条件
        if (Contour_Area < 15 || contours[i].size() <= 10)

            continue;

        // 用椭圆拟合区域得到外接矩形
        RotatedRect Rec = fitEllipse( contours[i]);
       
        if (Rec.angle > 20 )
            continue;

        // 长宽比和轮廓面积比限制
        if (Rec.size.width / Rec.size.height > 1.5
                || Contour_Area / Rec.size.area() < 0.5)
            continue;
       
         vc.push_back(Rec);
	}
/*	for(int u=0 ; u < vRec.size() ; u++ )
	{
	drawContours(src,vRec,u,Scalar(255,255,255),2,LINE_8,hierarchy,0,Point(0,0));
	}
	imshow("2",src);*/
	for (size_t i = 0; i < vc.size(); i++)
    {
        for (size_t j = i + 1; (j < vc.size()); j++)
        {
            //判断是否为相同灯条
            float Contour_angle = abs(vc[i].angle - vc[j].angle); //角度差
            if (Contour_angle >= 25)
                continue;
            //长度差比率
            float Contour_Len1 = abs(vc[i].size.height - vc[j].size.height) / max(vc[i].size.height, vc[j].size.height);
            //宽度差比率
            float Contour_Len2 = abs(vc[i].size.width - vc[j].size.width) / max(vc[i].size.width, vc[j].size.width);
            if (Contour_Len1 > 0.35 || Contour_Len2 > 0.35)
                continue;


            RotatedRect ZJB;
            ZJB.center.x = (vc[i].center.x + vc[j].center.x) / 2.; //x坐标
            ZJB.center.y = (vc[i].center.y + vc[j].center.y) / 2.; //y坐标
            ZJB.angle = (vc[i].angle + vc[j].angle) / 2.; //角度
            float nh, nw, yDiff, xDiff;
            nh = (vc[i].size.height + vc[j].size.height) / 2; //高度
            // 宽度
            nw = sqrt((vc[i].center.x - vc[j].center.x) * (vc[i].center.x - vc[j].center.x) + (vc[i].center.y - vc[j].center.y) * (vc[i].center.y - vc[j].center.y));
            float ratio = nw / nh; //匹配到的装甲板的长宽比
            xDiff = abs(vc[i].center.x - vc[j].center.x) / nh; //x差比率
            yDiff = abs(vc[i].center.y - vc[j].center.y) / nh; //y差比率
            if (ratio < 1.0 || ratio > 5.0 || xDiff < 0.5 || yDiff > 2.0)
                continue;
            ZJB.size.height = nh;
            ZJB.size.width = nw;
            vRec.push_back(ZJB);
            Point2f point1;
            Point2f point2;
            point1.x=vc[i].center.x;point1.y=vc[i].center.y+30;
            point2.x=vc[j].center.x;point2.y=vc[j].center.y-30;

			rectangle(dst, point1,point2, (0, 120, 255), 2);//将装甲板框起来
			circle(dst,ZJB.center,10,CV_RGB(0,120,255),-1);//在装甲板中心画一个圆
		}
	}

}
int main()
{
	VideoCapture capture("2.avi");
	Mat frame;

	if (!capture.isOpened())
	{
		cout << " cannot open the vedio" << endl;
		return -1;
	}

	capture>>frame;
	while (capture.read(frame))
	{
		cvtColor(frame, hsv_img, COLOR_BGR2HSV);
		vector<Mat> hsv_split;
		split(hsv_img,hsv_split);
		equalizeHist(hsv_split[2],hsv_split[2]);
		merge(hsv_split,hsv_img);
		inRange(hsv_img, Scalar(LowH,LowS,LowV), Scalar(HighH,HighS,HighV),threshold_img);
		
		Canny(threshold_img,threshold_img, 3, 9,3);
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(threshold_img, threshold_img, element);

		draw_keypoint(threshold_img, frame);
		imshow("1", frame);
		if (waitKey(30) == 32)
		{
			if (waitKey(0) == 32)
			{
				break;
			}
		};
	}
	capture.release();
	return 0;
}
