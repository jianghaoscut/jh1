#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include<chrono>
#include<string>
using namespace cv;
using namespace cv::ml;
using namespace std;

//获取点间距离
double getDistance(Point A,Point B)
{
    double dis;
    dis=pow((A.x-B.x),2)+pow((A.y-B.y),2);
    return sqrt(dis);
}
//标准化并计算hog
vector<float> stander(Mat im)
{

    if(im.empty()==1)
    {
        cout<<"filed open"<<endl;
    }
    resize(im,im,Size(48,48));

    vector<float> result;

    HOGDescriptor hog(Size(48,48),Size(16,16),Size(8,8),Size(8,8),9,1,-1,
                      HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);           //初始化HOG描述符
    hog.compute(im,result);
    return result;
}

//最小二乘法画圆
/*尝试用          hog+最小二乘法      画圆确定中心点
 * 但是因为镜头在运动，无法保证收集到的圆点为统一个圆的元素放弃了这个方法，
 * 最后通过drawContours画出大概中心R的轮廓*/

static bool CircleInfo2(std::vector<cv::Point2f>& pts, cv::Point2f& center, float& radius)
{
    center = cv::Point2d(0, 0);
    radius = 0.0;
    if (pts.size() < 3) return false;;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumX3 = 0.0;
    double sumY3 = 0.0;
    double sumXY = 0.0;
    double sumX1Y2 = 0.0;
    double sumX2Y1 = 0.0;
    const double N = (double)pts.size();
    for (int i = 0; i < pts.size(); ++i)
    {
        double x = pts.at(i).x;
        double y = pts.at(i).y;
        double x2 = x * x;
        double y2 = y * y;
        double x3 = x2 *x;
        double y3 = y2 *y;
        double xy = x * y;
        double x1y2 = x * y2;
        double x2y1 = x2 * y;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumY2 += y2;
        sumX3 += x3;
        sumY3 += y3;
        sumXY += xy;
        sumX1Y2 += x1y2;
        sumX2Y1 += x2y1;
    }
    double C = N * sumX2 - sumX * sumX;
    double D = N * sumXY - sumX * sumY;
    double E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX;
    double G = N * sumY2 - sumY * sumY;
    double H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY;

    double denominator = C * G - D * D;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double a = (H * D - E * G) / (denominator);
    denominator = D * D - G * C;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double b = (H * C - E * D) / (denominator);
    double c = -(a * sumX + b * sumY + sumX2 + sumY2) / N;

    center.x = a / (-2);
    center.y = b / (-2);
    radius = std::sqrt(a * a + b * b - 4 * c) / 2;
    return true;
}

//模板匹配
/*封装之后的模板匹配
 * 本来想通过获取字符串来做后来发现
 * 通过switch的方式可以比较方便的实现特殊参数的调用*/

double TemplateMatch(Mat image, Mat tepl, Point &point, int method)
{
//匹配结果图像大小		
    int result_cols =  image.cols - tepl.cols + 1;
    int result_rows = image.rows - tepl.rows + 1;

//匹配图像的定义
    Mat result = Mat( result_cols, result_rows, CV_32FC1 );
    matchTemplate( image, tepl, result, method );

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
     switch(method)
    {
    case TM_SQDIFF:
    case TM_SQDIFF_NORMED:
        point = minLoc;
        return minVal;

    default:
        point = maxLoc;
        return maxVal;  
	}
}

int main(int argc, char *argv[])
{
    VideoCapture cap;
    cap.open("1.avi");

    Mat srcImage;
    cap >> srcImage;
    // 画拟合圆
    Mat drawcircle=Mat(srcImage.rows,srcImage.cols, CV_8UC3, Scalar(0, 0, 0));

//	准备模板匹配
    Mat templ[9];
    for(int i=1;i<=8;i++)
    {
        templ[i]=imread("template/template"+to_string(i)+".jpg",IMREAD_GRAYSCALE);
    }
    vector<Point2f> cirV;

    Point2f cc=Point2f(0,0);
    while(true)
    {
        cap >> srcImage;

        if(!cap.isOpened())
        {
            cout << "Capture failed." << endl;
            continue;
        }
    
        //分割颜色通道
        vector<Mat> imgChannels;
        split(srcImage,imgChannels);
        Mat midImage2=imgChannels.at(0)-imgChannels.at(2);
		//二值化处理
        threshold(midImage2,midImage2,100,255,THRESH_BINARY);
        imshow("midImage2",midImage2);

        //形态学处理
        Mat element=getStructuringElement(MORPH_RECT,Size(5,5),Point(-1,-1));
       
        dilate(midImage2,midImage2,element,Point(-1,-1));
      //  morphologyEx(midImage2,midImage2, MORPH_OPEN, element,Point(-1,-1),2);
        
        element=getStructuringElement(MORPH_RECT,Size(7,7),Point(-1,-1));
        morphologyEx(midImage2,midImage2, MORPH_CLOSE, element,Point(-1,-1));
	   	imshow("dilate",midImage2);
        
	    //处理完图像，准备提取轮廓点	
        vector<vector<Point>> contours2;
        vector<Vec4i> hierarchy2;
        findContours(midImage2,contours2,hierarchy2,RETR_TREE,CHAIN_APPROX_SIMPLE);

        RotatedRect rect_tmp2;

        //遍历轮廓
        if(hierarchy2.size())
            for(int i=0;i>=0;i=hierarchy2[i][0])
            {
                rect_tmp2=minAreaRect(contours2[i]);
                Point2f P[4];
                rect_tmp2.points(P);

                Point2f srcRect[4];
                Point2f dstRect[4];

                double width;
                double height;

                //矫正提取的叶片的宽高,得到一个不扭曲的长方形
                width=getDistance(P[0],P[1]);
                height=getDistance(P[1],P[2]);
                if(width>height)
                {
                    srcRect[0]=P[0];
                    srcRect[1]=P[1];
                    srcRect[2]=P[2];
                    srcRect[3]=P[3];
                }
                else
                {
                    swap(width,height);
                    srcRect[0]=P[1];
                    srcRect[1]=P[2];
                    srcRect[2]=P[3];
                    srcRect[3]=P[0];
                }

                Scalar color( 0, 120, 0);
                drawContours(srcImage, contours2, i, color, 4, 4, hierarchy2);
                //通过面积筛选
                double area=height*width;
                if(area>5000){

                    cout <<hierarchy2[i]<<endl;
                        
                    dstRect[0]=Point2f(0,0);
                    dstRect[1]=Point2f(width,0);
                    dstRect[2]=Point2f(width,height);
                    dstRect[3]=Point2f(0,height);
                    // 应用透视变换，矫正成规则矩形，进行模板匹配
                    Mat transform = getPerspectiveTransform(srcRect,dstRect);
                    Mat perspectMat;
                    warpPerspective(midImage2,perspectMat,transform,midImage2.size());

                    imshow("warpdst",perspectMat);
                    // 进行模板匹配
                    Mat testim;
                    testim = perspectMat(Rect(0,0,width,height));

                    Point matchLoc;
                    double value;
                    Mat tmp1;
                    resize(testim,tmp1,Size(42,20));

                    vector<double> Vvalue1;//识别待打击的叶片
                    vector<double> Vvalue2;//识别已经打击过的叶片
                    for(int j=1;j<=6;j++)
                    {
                        value = TemplateMatch(tmp1, templ[j], matchLoc, TM_CCORR);
                        Vvalue1.push_back(value);
                    }
                    for(int j=7;j<=8;j++)
                    { 
						//重新封装的函数中，返回了最大点或最小点值作为匹配值	
                        value = TemplateMatch(tmp1, templ[j], matchLoc, TM_CCORR);
                        Vvalue2.push_back(value);
                    }
                    int maxv1=0,maxv2=0;

                    for(int t1=0;t1<6;t1++)
                    {
                        if(Vvalue1[t1]>Vvalue1[maxv1])
                        {
                            maxv1=t1;
                        }
                    }
                    for(int t2=0;t2<2;t2++)
                    {
                        if(Vvalue2[t2]>Vvalue2[maxv2])
                        {
                            maxv2=t2;
                        }
                    }

                    cout<<Vvalue1[maxv1]<<endl;
                    cout<<Vvalue2[maxv2]<<endl;

                    
                    //预测是否是要打击的扇叶
                    if(Vvalue1[maxv1]>Vvalue2[maxv2]&&Vvalue1[maxv1]>0.6)

                    {
                        //查找装甲板
                        if(hierarchy2[i][2]>=0)
                        {
                            RotatedRect rect_tmp=minAreaRect(contours2[hierarchy2[i][2]]);
                            Point2f Pnt[4];
                            rect_tmp.points(Pnt);
                            const float maxHWRatio=0.7153846;
                            const float maxArea=2000;
                            const float minArea=500;

                            float width=rect_tmp.size.width;
                            float height=rect_tmp.size.height;
                            if(height>width)
                                swap(height,width);
                            float area=width*height;
                            
						    //根据得分预测是否打击	
                            if(height/width>maxHWRatio||area>maxArea ||area<minArea)
							{
                                 continue;
                            }
                            Point centerP=rect_tmp.center;
                            //打击点
                            circle(srcImage,centerP,1,Scalar(0,255,0),2);

                            circle(drawcircle,centerP,1,Scalar(0,0,255),1);
              
	//		 --------------------------------------------------------------------------------------------------------------
	 /*				 
                            if(cirV.size()<30)
                            {
                                cirV.push_back(centerP);
                            }
                            else
                            {
                                float R;
                                //得到拟合的圆心
                          //      CircleInfo2(cirV,cc,R);
                          //      circle(drawcircle,cc,1,Scalar(255,0,0),2);

                                cirV.erase(cirV.begin());
                            }
                            if(cc.x!=0&&cc.y!=0){
                                Mat rot_mat=getRotationMatrix2D(cc,30,1);

                               //将打击点围绕圆心旋转一定角度得到的预测打击点
                                float sinA=rot_mat.at<double>(0,1);//sin(60);
                                float cosA=rot_mat.at<double>(0,0);//cos(60);
                                float xx=-(cc.x-centerP.x);
                                float yy=-(cc.y-centerP.y);
                                Point2f resPoint=Point2f(cc.x+cosA*xx-sinA*yy,cc.y+sinA*xx+cosA*yy);
                                circle(srcImage,resPoint,1,Scalar(0,255,0),10);
                            }
*/
                            for(int j=0;j<4;++j)
                            {
                                line(srcImage,Pnt[j],Pnt[(j+1)%4],Scalar(0,255,255),2);
                            }
                        }
                    }
               }
            }

  //      imshow("circle",drawcircle);

        imshow("Result",srcImage);
        if (waitKey(30)==32)
		{
				if(waitKey(0)==32)
				{ 
						break;
				}
		}

    }
    return 0;
}
