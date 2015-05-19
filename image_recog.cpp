#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;

//Store all Necessary data of a Pixel Group
class pixGroup
{
public:
	int num, maxPix, maxX, maxY, area, mean;
	long xpos, ypos, perimtr;
	double oriA, oriB, oriC, ori, proj;
	void clear(void);
};
void pixGroup::clear(void)
{
	//Initialize all variables
	maxPix = -1;
	maxX = 0;
	maxY = 0;
	area = 0;
	xpos = 0;
	ypos = 0;
	oriA = 0;
	oriB = 0;
	oriC = 0;
	ori = 0;
	perimtr = 0;
	proj = 0;
}

//Operation Functions
bool ThresFunc(void);
bool kmcFunc(int);
bool RegGrwFunc(int);
bool GradEdgeFunc(void);
bool FisherLearner(void);
bool CleanGroups(void);
bool FisherDeterm(void);
bool EDeterm(void);
bool DescSeperator(void);
bool CalcProjections(void);
bool FindEucDist(void);
bool OtherSeperator(void);

void print2dMatrix(float [3][3]);
void print1dMatrix(float [3]);

//Global Variables
Mat image, gimage, outimage1, outimage2;
ofstream fOut, fDet;
int noOfPix, minval = 1000, maxval = 0, reqTask = 0;
double FD[3], FDthres, ED[4];
int histog[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
String ipImage, timage1, timage2, descfile;
bool skip = false;
vector <pixGroup> grpList, grpList1, grpList2;
int *timage;
vector <int> imagePall;

int main( int argc, char** argv )
{
    if( argc < 3)
    {
     cout <<"----Error: Incomplete Command\n\tUsage (Image Processing)\t: \"Command\" \"Input Image File\" \"Operation\" \"[Other Operation Specifics]\"" << endl;
	 cout <<"\tUsage (Pattern Learning)\t: \"Command\" \"train\" \"Image File 1\" \"Image File 2\"" << endl;
	 cout <<"\tUsage (Pattern Recognition)\t: \"Command\" \"sep\" \"Image File 1\" \"Pattern File\"" << endl;
	 return -1;
    }
	String oper;
	oper.assign(argv[1]);
	//Output file to record Stats
	String outfile;
	outfile.assign(argv[1]);
	outfile.append("_");
	outfile.append(argv[2]);
	outfile.append("_stats.csv");
	cout << "Function: " << oper <<std::endl;
	if((oper.compare("trainf") != 0) && (oper.compare("sep") != 0))
	{
		// Read the file
		image = imread(argv[1], IMREAD_COLOR); 
	    gimage = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);  //Grey Scale image format used for all funcs

		// Check for invalid input
		if(! image.data || ! gimage.data ) 
		{
	        cout << "Could not open or find the image" << std::endl ;
		    return -1;
		}

		//Used every where
		noOfPix = gimage.cols*gimage.rows;

		//Get Basic Image Properties
		for(int i = 0; i < noOfPix; i=i+1)
		{
			//cout<<int(image.data[noOfPix*image.rows+i])<<" ";
			if((int(gimage.data[i])) < minval)
				minval = int(gimage.data[i]);
			if(int(gimage.data[i]) > maxval)
				maxval = int(gimage.data[i]);
		}

		fOut.open(outfile);
		fOut <<"Start of Report\nCommand: "<<argv[0]<<" "<<argv[1]<<" "<<argv[2]<<"\n";
		fOut<<"Image Stats\n";
		fOut <<"Image Size,"<<gimage.rows<<","<<gimage.cols<<"\n";
		fOut <<"Max Pix Value, "<<maxval<<"\nMin Pixel Value, "<<minval<<"\n";

		ipImage.assign(argv[1]);
		oper.assign(argv[2]);
	}
	if(oper.compare("thrs") == 0)
	{
		cout << "Thresholding in Progress...";
		ThresFunc();
		cout <<"Thresholding Complete\n";
		//Group the resultant image (gimage) and label the groups using Region Growing Labeling
		reqTask = 1;
		RegGrwFunc(100);
	}
	else if(oper.compare("kmc") == 0)
	{
		cout << "k-Means Clustering in Progress...";
		int km = 2;
		if(argc == 4)
		{
			String kmarg;
			kmarg.assign(argv[3]);
			km = atoi(argv[3]);
			if(km < 2)
			{
				cout << "\n\tNo. of Means for K-means method was \""<<km<<"\". The default value 2 was chosen\n";
				km = 2;
			}
		}
		cout<<"\n\tNo of Means used: "<<km<<"\n";
		kmcFunc(km);
		cout <<"k-means Clustering Complete\n";

		//Group the resultant image (gimage) and label the groups using Region Growing Labeling
		reqTask = 2;
		RegGrwFunc(100);
	}
	else if(oper.compare("rgrw") == 0)
	{
		cout << "Region Growing Clustering Method - using Eight Neighbour rule - in Progress... " <<oper << std::endl;
		int grpCond = 50;
		if(argc == 4)
		{
			grpCond = atoi(argv[3]);
			if(grpCond < 50)
			{
				cout <<"\n\tThe selected Grouping condition \""<<grpCond<<"\" may segment the image into a large number of disjoint regions. This value is defaluted to 50.\n";
				grpCond = 50;
			}
		}
		reqTask = 3;
		RegGrwFunc(grpCond);
	}
	else if(oper.compare("ged") == 0)
	{
		cout << "Gradient Edge Detection in Progress..." << std::endl;
		GradEdgeFunc();
		cout << "Complete" << std::endl;
	}
	else if(oper.compare("trainf") == 0)
	{
		cout << "Reading Training data sets..." << std::endl;
		timage1.assign(argv[2]);
		timage2.assign(argv[3]);
		//Training function
		FisherLearner();
		cout << "Training Complete" << std::endl;
	}
	else if(oper.compare("traine") == 0)
	{
		cout << "Reading Training data sets..." << std::endl;
		timage1.assign(argv[2]);
		timage2.assign(argv[3]);
		//Training function
		FisherLearner();
		cout << "Training Complete" << std::endl;
	}
	else if(oper.compare("sep") == 0)
	{
		skip = true;
		cout << "Classifying objects in the image using the training data..." << std::endl;
		//Clasifying function
		descfile.assign(argv[3]);
		timage1.assign(argv[2]);
		DescSeperator();
		cout << "Classfication Complete" << std::endl;
	}
	else
	{
		cout << "Undefined operation \"" <<oper<<"\"\n";
		cout << "Valid operations:\n\t\"thrs\"\t- Segmentation using Thresholding\n\t\"kmc\"\t- K-means Clustering\n\t\"rgrw\"\t- Region Growing\n\t\"ged\"\t- Gradient Based Edge detection"<< std::endl;
        return -1;
	}
	if(!skip)
	{
		//Display the Input Image
		namedWindow( "Input Image", WINDOW_AUTOSIZE ); 
	    imshow( "Input Image", image ); // Show Input image 

		namedWindow( "Output Image", WINDOW_AUTOSIZE ); 
	    imshow( "Output Image", gimage ); // Show Output image
		outfile.assign("out_");
		outfile.append(argv[2]);
		outfile.append("_");
		outfile.append(argv[1]);
		imwrite(outfile,gimage);
		fOut.close();
	    waitKey(0); // Wait for a keystroke to close the windows
	}
    return 0;
}

bool ThresFunc(void)
{
	// Thresholding Function
	int gradnt = (maxval - minval)/20;
	if(!skip)
	{
		fOut << "\nOperation: Thresholding "<<std::endl;
		fOut << "Gradiant for Histogram, "<< gradnt <<"\n";
	}
	
	//Create Histogram
	int noOfPix = gimage.cols*gimage.rows;
	for(int i = 0; i < noOfPix; i++)
	{
		for(int k = 0; k < 19; k++)
		{
			if(gimage.data[i] <= (minval + (k+1)*gradnt))
			{
				histog[k] = histog[k]+1;
				break;
			}
		}
	}
	if(!skip)
	{
		cout<<"\nHistogram\n";
		fOut<<"Histogram:,";
		for(int i = 0; i < 20; i++)
		{
			cout<<histog[i]<<", ";
			fOut<<histog[i]<<", ";
		}
		cout<<"\n";
	}
	//Find valley
	bool peak = false;
	int thres = minval;
	int currmaxslope = 0;
	int currpeak = 0;
	for(int i = 0; i < 19; i++)
	{
		if(peak)
		{
			if(histog[i] < histog[i+1])
			{
				if((currpeak - histog[i]) > currmaxslope)
				{
					thres = minval + gradnt*i;
					currmaxslope = currpeak - histog[i];
				}
				peak = false;
			}
		}
		else
		{
			if(histog[i] > histog[i+1])
			{
				peak = true;
				currpeak = histog[i];
			}
		}
	}
	if(!skip)	
	{
		cout<<"\nThreshold Selected = "<<thres;
		fOut<<"\nThreshold Selected ,"<<thres<<"\n\n";
	}
	cout<<"\nThreshold Selected = "<<thres;

	pixGroup grp1, grp2;
	grp1.clear();
	grp2.clear();
	int i = 0;
	for(int row = 0; row < gimage.rows; row++)
		for(int col = 0; col < gimage.cols; col ++)
		{
			if(gimage.data[i] < thres)
			{
				gimage.data[i] = 0;
				grp1.xpos += col;
				grp1.ypos += row;
				grp1.area++;
			}
			else
			{
				gimage.data[i] = 250;
				grp2.xpos += col;
				grp2.ypos += row;
				grp2.area++;
			}
			i++;
		}
	return true;
}
bool kmcFunc(int km)
{
	int *thresh = new(int[km - 1]);
	int *kmeans = new(int[km]);
	int *kmeanN = new(int[km]);
	int *kmPixl = new(int[km]);
	for(int i = 0; i<km; i++)
		kmeans[i] = 0;
	//Load Initial Thresholds
	fOut<<"\nInitial Thresholds,";
	int gradnt = (maxval - minval)/km;
	for(int i = 0; i < km - 1; i++)
	{
		kmPixl[i] = 0;
		kmeans[i] = 0;
		kmeanN[i] = 0;
		thresh[i] = minval + gradnt*(i+1);
		fOut <<thresh[i]<<", ";
	}
	kmPixl[km -1] = 0;
	kmeans[km -1] = 0;
	kmeanN[km -1] = 0;
	bool cont = true;
	int count = 0;

	//Group pixels
	while(cont && (count < 10))
	{
		//Find the new Means
		cont = false;
		int mean = 0;
		int k = 0, l = 0;
		fOut << "\n\n";
		fOut<<"Iteration["<<count<<"],";
		for(int i = 0; i < noOfPix; i++)
		{
			for( k = 0; k < km; k++)
			{
				if( k == km - 1)
				{
					kmeanN[k] += gimage.data[i];
					kmPixl[k]++;
				}
				else if(gimage.data[i] < thresh[k])
				{
					kmeanN[k] += gimage.data[i];
					kmPixl[k]++;
					break;
				}
			}
		}

		//Find new means and Thresholds
		for(int i = 0; i < km ; i++)
		{
			if(kmeanN[i] > 0)
			{
				kmeanN[i] = kmeanN[i]/kmPixl[i];
			}
			if(kmeans[i] != kmeanN[i])
			{
				kmeans[i] = kmeanN[i];
				cont = true;
			}
			fOut << "Group["<<i<<"], Pixels =,"<<kmPixl[i]<<", Mean =,"<<kmeanN[i]<<",\t";
			kmeanN[i] = 0;
			kmPixl[i] = 0;
		}
		fOut<<"\n Thresholds,";
		for(int i = 0; i < km - 1; i++)
		{
			thresh[i] = (kmeans[i] + kmeans[i + 1])/2;
			fOut<<thresh[i]<<",";
		}
	
		count++;
	}
	fOut<<",Final\n";
	//Apply Threshold
	int k = 0;
	int *pcount = new(int[km]);
	for(int  i = 0; i < km; i++)
		pcount[i] = 0;
	for(int i = 0; i < noOfPix; i++)
	{
		for( k = 0; k < km; k++)
		{
			if( k == km - 1)
			{
				gimage.data[i] = 255;
				pcount[k]++;
			}
			else if(gimage.data[i] < thresh[k])
			{
				gimage.data[i] = gradnt*k;
				pcount[k]++;
				break;
			}
		}
	}
	return true;
}

bool RegGrwFunc(int grpCond)
{
	if(reqTask == 1 || reqTask == 2)
	{
		cout <<"\n\tGrouping Image pixels and labeling using Region Growing Method...\n";
		fOut <<"\nGrouping Image pixels and labeling using Region Growing Method\n";
	}
	timage = new(int[noOfPix]);
	pixGroup currGrp;

	//Preset values
	for(int i = 0; i < noOfPix; i++)
	{
		timage[i] = 0;
		imagePall.push_back(timage[i]);
	}
	
	int markedPix = 0;
	currGrp.num = 0;
	int grpCount = 0;
	bool cont = true;
	int temppos = 0;

	while(markedPix < noOfPix)
	{
		currGrp.clear();
		currGrp.num++;
		grpCount++;
		
		//Find a new Max Pix for this Group
		for(int i = 0; i < noOfPix ; i++)
		{
			if(timage[i] == 0)
			{
				if(gimage.data[i] > currGrp.maxPix)
				{
					currGrp.maxPix = gimage.data[i];
					currGrp.maxX = i/gimage.rows;
					currGrp.maxY = i%gimage.rows;
					temppos = i;
				}
			}
		}
		timage[temppos] = grpCount;
		imagePall[temppos]= grpCount;

		markedPix++;
		currGrp.area++;
		currGrp.xpos += currGrp.maxX;
		currGrp.ypos += currGrp.maxY;
		//Start grouping until no more changes
		while(1)
		{
			bool stats = true;
			int i = 0;
			//cout<<"\rCurr Count: "<<markedPix;
			cont = false;
			for(int row = 0; row <gimage.rows; row++)
				for(int col = 0; col <gimage.cols; col++)
				{
					stats = false;
					if(row == 0 && col == 0)
					{
						//Top Left Corner
						if(timage[i] == 0)
						{
							if(timage[i+1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(row == 0 && col == gimage.cols - 1)
					{
						//Top Right Corner
						if(timage[i] == 0)
						{
							if(timage[i-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(row == 0)
					{
						//Rest of Top Row
						if(timage[i] == 0)
						{
							if(timage[i+1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(row == gimage.cols - 1 && col == 0)
					{
						//Bottom Left Corner
						if(timage[i] == 0)
						{
							if(timage[i+1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(row == gimage.cols - 1 && col == gimage.cols - 1)
					{
						//Botton Right Corner
						if(timage[i] == 0)
						{
							if(timage[i-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(row == gimage.cols - 1)
					{
						//Rest of Botton Row
						if(timage[i] == 0)
						{
							if(timage[i+1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(col == 0)
					{
						//Rest of Left Edge
						if(timage[i] == 0)
						{
							if(timage[i+1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i-gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else if(col == gimage.cols - 1)
					{
						//Rest of Right Edge
						if(timage[i] == 0)
						{
							if(timage[i-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i-gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i-gimage.cols-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					else
					{
						//Other Points
						if(timage[i] == 0)
						{
							if(timage[i+1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i-1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i - gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols - 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
							else if(timage[i + gimage.cols + 1] == currGrp.num)
							{
								if(abs(currGrp.maxPix - gimage.data[i]) <= grpCond)
								{
									stats = true;
								}
							}
						}
					}
					if(stats)
					{
						currGrp.area++;
						currGrp.xpos += col;
						currGrp.ypos += row;
						cont = true;
						timage[i] = currGrp.num;
						imagePall[i] = currGrp.num;
						markedPix++;
					}
					i++;
				}
			if(!cont)
				break;
		}
		//Move this List to the current group
		grpList.push_back(currGrp);
	}
	cout << "\n\tGrouping Complete  Total No of Groups:"<<grpCount <<","<<grpList.size();
	fOut << "\nNo of Groups Formed,"<<grpCount;
	//Grouping Complete, Calculating the Image Details
	for(int i = 0; i < grpCount; i++)
	{
		if(grpList[i].area > 0)
		{
			grpList[i].xpos = grpList[i].xpos/grpList[i].area;
			grpList[i].ypos = grpList[i].ypos/grpList[i].area;
		}
	}

	//Calculate Orientation and Perimeter
	int i = 0;
	for(int row = 0; row < gimage.rows; row++)
		for(int col = 0; col < gimage.cols; col++)
		{
			//Orientation
			grpList[timage[i]-1].oriA = (grpList[timage[i]-1].maxX - col)*(grpList[timage[i]-1].maxX - col);
			grpList[timage[i]-1].oriB = (grpList[timage[i]-1].maxY - row)*(grpList[timage[i]-1].maxY - row);
			grpList[timage[i]-1].oriC = (grpList[timage[i]-1].maxX - col)*(grpList[timage[i]-1].maxY - row);
			//Perimeter
			if(row == 0 || col == 0)
			{
				grpList[timage[i]-1].perimtr++;
			}
			else
			{
				if(timage[i] != timage[i-1])
				{
					grpList[timage[i]-1].perimtr++;
					if(timage[i-1] > 0)
						grpList[timage[i-1]-1].perimtr++;
				}
				if(timage[i] != timage[i - gimage.cols])
				{
					grpList[timage[i]-1].perimtr++;
					if(timage[i-gimage.cols] > 0)
						grpList[timage[i-gimage.cols]-1].perimtr++;
				}
			}
			i++;
		}
	//Calculating Orientation and Print Results
	fOut<<"\nGroup Name/Number,Area,Position - m,Position - n,Orientation(Rads),Perimeter\n";
	for(i = 0; i < grpCount; i++)
	{
		if(grpList[i].area > 0)
		{
			grpList[i].ori = atan(grpList[i].oriC/(grpList[i].oriA - grpList[i].oriB));
			fOut<<"\nGroup["<<i+1<<"],"<<grpList[i].area<<","<<grpList[i].xpos<<","<<grpList[i].ypos<<","<<grpList[i].ori<<","<<grpList[i].perimtr;
		}
	}
	//Skip the following, it operation is Thresholding or k-Means Clustering
	if(reqTask != 3)
		return true;
	//Set the colors in the image to represent groups
	gimage = imread(ipImage, IMREAD_COLOR);

	i = 0;
	for(int row = 0; row < gimage.rows; row++)
		for (int col = 0; col < gimage.cols; col++)
		{
			gimage.at<cv::Vec3b>(row, col)[0] = timage[i]*255/grpCount;
			gimage.at<cv::Vec3b>(row, col)[1] = (timage[i]*255/grpCount + 50)%255;
			gimage.at<cv::Vec3b>(row, col)[2] = (timage[i]*255/grpCount + 100)%255;
			i++;
		}
	
	return true;
}

bool GradEdgeFunc(void)
{
	//Store the Dfx and Dfy of the images
	int *dfx = new(int[noOfPix]);
	int *dfy = new(int[noOfPix]);
	int *dfmag = new(int[noOfPix]);
	int i = 0;
	long thresh = 0;
	for(int row = 0; row < gimage.rows;row++)
		for(int col = 0; col < gimage.cols;col++)
		{
			if(row == 0 && col == 0)
			{
				dfx[i] = gimage.data[i];
				dfy[i] = gimage.data[i];
			}
			else if(row == 0 && col == gimage.cols - 1)
			{
				dfx[i] = gimage.data[i] - gimage.data[i-1];
				dfy[i] = gimage.data[i];
			}
			else if(row == 0)
			{
				dfx[i] = gimage.data[i] - gimage.data[i-1];
				dfy[i] = gimage.data[i];
			}
			else if(row == gimage.rows-1 && col == 0)
			{
				dfx[i] = gimage.data[i];
				dfy[i] = gimage.data[i] - gimage.data[i - gimage.rows];
			}
			else if(row == gimage.rows-1 && col == 0)
			{
				dfx[i] = gimage.data[i] - gimage.data[i-1];
				dfy[i] = gimage.data[i] - gimage.data[i - gimage.rows];
			}
			else if(row == gimage.rows -1)
			{
				dfx[i] = gimage.data[i] - gimage.data[i-1];
				dfy[i] = gimage.data[i] - gimage.data[i - gimage.rows];
			}
			else if(col == 0)
			{
				dfx[i] = gimage.data[i] - gimage.data[i];
				dfy[i] = gimage.data[i] - gimage.data[i - gimage.rows];
			}
			else if(col == gimage.cols-1)
			{
				dfx[i] = gimage.data[i] - gimage.data[i-1];
				dfy[i] = gimage.data[i] - gimage.data[i - gimage.rows];
			}
			else
			{
				dfx[i] = gimage.data[i] - gimage.data[i-1];
				dfy[i] = gimage.data[i] - gimage.data[i - gimage.rows];
			}
			dfmag[i] = int(sqrt((dfx[i]*dfx[i]) + (dfy[i]*dfy[i])));
			thresh += dfmag[i];
			i++;
		}
	//Edge Magnitude Matrix Ready, find the threshold - mean of the values
	thresh = thresh/noOfPix;
	//Mark the edges
	for(i = 0; i < noOfPix; i++)
	{
		if(dfmag[i] > thresh)
			gimage.data[i] = 255;
		else
			gimage.data[i] = 0;
	}
	return true;
}

bool FisherLearner(void)
{
	skip = true;
	//Load first image, find its properties
	gimage = imread(timage1, CV_LOAD_IMAGE_GRAYSCALE);  
	// Check for invalid input
	if(! gimage.data ) 
	{
        cout << "Could not open or find the image file: " << timage1 << std::endl ;
	    return false;
	}

	cout<<"File: "<<timage1<<"\n";
	//Used every where
	noOfPix = gimage.cols*gimage.rows;

	//Get Basic Image Properties
	for(int i = 0; i < noOfPix; i=i+1)
	{
		//cout<<int(image.data[noOfPix*image.rows+i])<<" ";
		if((int(gimage.data[i])) < minval)
			minval = int(gimage.data[i]);
		if(int(gimage.data[i]) > maxval)
			maxval = int(gimage.data[i]);
	}

	//Sperate object pixels using thresholding
	ThresFunc();
	namedWindow( "Thresholded1", WINDOW_AUTOSIZE ); 
    imshow( "Thresholded1", gimage ); // Show Output image
	reqTask = 1;
	//Find object properties
	RegGrwFunc(100);

	//Clean Image
	CleanGroups();
	cout <<"\tCleaning Complete\n\n";
	
	for(unsigned int i = 0; i < grpList.size(); i++)
		grpList1.push_back(grpList[i]);
	
	grpList.clear();

	//Load first image, find its properties
	gimage = imread(timage2, CV_LOAD_IMAGE_GRAYSCALE);  
	// Check for invalid input
	if(! gimage.data ) 
	{
        cout << "Could not open or find the image"  << timage2 << std::endl ;
	    return false;
	}

	//Used every where
	noOfPix = gimage.cols*gimage.rows;
	cout<<"File: "<<timage2<<"\n";
	//Get Basic Image Properties
	for(int i = 0; i < noOfPix; i=i+1)
	{
		//cout<<int(image.data[noOfPix*image.rows+i])<<" ";
		if((int(gimage.data[i])) < minval)
			minval = int(gimage.data[i]);
		if(int(gimage.data[i]) > maxval)
			maxval = int(gimage.data[i]);
	}

	//Sperate object pixels using thresholding
	ThresFunc();
	namedWindow( "Thresholded2", WINDOW_AUTOSIZE ); 
    imshow( "Thresholded2", gimage ); // Show Output image
	reqTask = 1;
	//Find object properties
	RegGrwFunc(100);

	//Clean Image
	CleanGroups();
	cout <<"\tCleaning Complete\n\n";

	for(unsigned int i = 0; i < grpList.size(); i++)
		grpList2.push_back(grpList[i]);

	grpList.clear();
	
	//Get Fisher Discriminant
	FisherDeterm();

	//Store Fisher Discriminent
	fDet.open("pattern_desc.txt");
	fDet << "fisher\n";
	fDet << FD[0] << " " << FD[1] << " " << FD[2];
	fDet.close();

	cout << "Fisher Determinant:\n\t" <<FD[0] << ", " << FD[1] << ", " << FD[2] << "\n\tSaved to file \"fisher_desc.txt\"\n";

	skip = true;
	waitKey(0); // Wait for a keystroke to close the windows
	return true;
}

bool CleanGroups(void)
{
	int maxarea = 0, maxpos = 0, count = 0;
	for(unsigned int i = 0; i < grpList.size(); i++)
	{
		if(maxarea < grpList[i].area)
		{
			maxpos = i;
			maxarea = grpList[i].area;
		}
		if (grpList[i].area < 200)
			grpList[i].area = 0;
		else
			count++;
	}
	grpList[maxpos].area = 0;
	cout << "\n\tNo of Objects found: "<<--count<<"\n";
	return true;
	//Group List has only the object pixels
}

bool FisherDeterm()
{
	float M0[3], M1[3], S1[3][3], S2[3][3], Sw[3][3], SwI[3][3], T1[3], T2[3];
	for(int i = 0; i < 3; i++)
		T1[i] = T2[i] = M0[i] = M1[i] = 0;

	int j = 0, k = 0;
	for(unsigned int i = 0; i < grpList1.size(); i++)
	{
		if(grpList1[i].area > 0)
		{
			M0[0] += grpList1[i].area;
			M0[1] += grpList1[i].perimtr;
			M0[2] += (grpList1[i].perimtr*grpList1[i].perimtr)/grpList1[i].area;
			j++;
		}
	}

	for(unsigned int i = 0; i < grpList2.size(); i++)
	{
		if(grpList2[i].area > 0)
		{
			M1[0] += grpList2[i].area;
			M1[1] += grpList2[i].perimtr;
			M1[2] += (grpList2[i].perimtr*grpList2[i].perimtr)/grpList2[i].area;
			k++;
		}
	}

	for(int i = 0; i < 3; i++)
	{
		M0[i] = M0[i]/j;
		M1[i] = M1[i]/k;
	}
	//cout <<"M0\n\t"<<M0[0]<<" " <<M0[1]<<" "<<M0[2]<<"\n";
	//cout <<"M1\n\t"<<M1[0]<<" " <<M1[1]<<" "<<M1[2]<<"\n";

	for(int m = 0; m < 3; m++)
		for(int n = 0; n < 3 ; n++)
		{
			S1[m][n] = 0;
			S2[m][n] = 0;
			Sw[m][n] = 0;
		}
	float tempstor[3];
	for(int i = 0;  i < 3; i++)
		tempstor[i] = 0;
	for(unsigned int i = 0; i < grpList1.size(); i++)
	{
		if(grpList1[i].area > 0)
		{
			
			tempstor[0] = grpList1[i].area - M0[0];
			tempstor[1] = grpList1[i].perimtr - M0[1];
			tempstor[2] = (grpList1[i].perimtr*grpList1[i].perimtr)/grpList1[i].area - M0[2];

			//cout <<"\nScatter Matrix 0: " << i <<"\t";
			//print1dMatrix(tempstor);
			for(int m = 0; m < 3; m++)
			{
				for(int n = 0; n < 3; n++)
				{
					S1[m][n] += tempstor[m]*tempstor[n];
					//cout << tempstor[m]*tempstor[n] << "  ";
				}
				//cout << "\n";
			}
		}
	}
	//cout <<"\nScatter Matrix 0\n";
	//print1dMatrix(tempstor);
	
	//cout <<"\nS1\n";
	//print2dMatrix(S1);

	for(int i = 0;  i < 3; i++)
		tempstor[i] = 0;
	for(unsigned int i = 0; i < grpList2.size(); i++)
	{
		if(grpList2[i].area > 0)
		{
			tempstor[0] = grpList2[i].area - M1[0];
			tempstor[1] = grpList2[i].perimtr - M1[1];
			tempstor[2] = (grpList2[i].perimtr*grpList2[i].perimtr)/grpList2[i].area - M1[2];

			for(int m = 0; m < 3; m++)
				for(int n = 0; n < 3; n++)
				{
					S2[m][n] += tempstor[m]*tempstor[n];
				}
		}
	}
	//cout <<"\nScatter Matrix 1\n";
	//print1dMatrix(tempstor);

	for(int m = 0; m < 3; m++)
		for(int n = 0; n < 3; n++)
		{
			S2[m][n] += tempstor[m]*tempstor[n];
		}
	//cout <<"\nS2\n";
	//print2dMatrix(S2);
	//Find Intra Group Scatter Matrix Sw
	for(int m = 0; m < 3; m++)
		for(int n = 0; n< 3; n++)
			Sw[m][n] = S1[m][n] + S2[m][n];
	//cout <<"\nSw\n";
	//print2dMatrix(Sw);
	float det = Sw[0][0]*(Sw[1][1]*Sw[2][2] - Sw[1][2]*Sw[2][1]) - Sw[0][1]*(Sw[1][0]*Sw[2][2] - Sw[1][2]*Sw[2][0]) + Sw[0][2]*(Sw[1][0]*Sw[2][1] - Sw[1][1]*Sw[2][0]);
	//cout <<"Determinent: "<< det << " \n";

	//Find inverse of Intra Group Scatter Matrix Sw
	SwI[0][0] = (Sw[1][1]*Sw[2][2] - Sw[2][1]*Sw[1][2])/det;
	SwI[0][1] = -1*(Sw[0][1]*Sw[2][2] - Sw[0][2]*Sw[2][1])/det;
	SwI[0][2] = (Sw[0][1]*Sw[1][2] - Sw[1][1]*Sw[0][2])/det;

	SwI[1][0] = -1*(Sw[1][0]*Sw[2][2] - Sw[1][2]*Sw[2][0])/det;
	SwI[1][1] = (Sw[0][0]*Sw[2][2] - Sw[2][0]*Sw[0][2])/det;
	SwI[1][2] = -1*(Sw[0][0]*Sw[1][2] - Sw[1][0]*Sw[0][2])/det;

	SwI[2][0] = (Sw[1][0]*Sw[2][1] - Sw[1][1]*Sw[2][0])/det;
	SwI[2][1] = -1*(Sw[0][0]*Sw[2][1] - Sw[0][1]*Sw[2][0])/det;
	SwI[2][2] = (Sw[0][0]*Sw[1][1] - Sw[1][0]*Sw[0][1])/det;

	//cout <<"\nSwI\n";
	//print2dMatrix(SwI);
	for(int i = 0; i < 3; i++)
	{
		FD[i] = 0;
		for(j = 0; j < 3; j++)
			FD[i] += (M1[j] - M0[j])*SwI[i][j];
	}

	return true;
}

bool EDeterm(void)
{
	skip = true;
	//Load first image, find its properties
	gimage = imread(timage1, CV_LOAD_IMAGE_GRAYSCALE);  
	// Check for invalid input
	if(! gimage.data ) 
	{
        cout << "Could not open or find the image file: " << timage1 << std::endl ;
	    return false;
	}

	//cout<<"File: "<<timage1<<"\n";
	//Used every where
	noOfPix = gimage.cols*gimage.rows;

	//Get Basic Image Properties
	for(int i = 0; i < noOfPix; i=i+1)
	{
		//cout<<int(image.data[noOfPix*image.rows+i])<<" ";
		if((int(gimage.data[i])) < minval)
			minval = int(gimage.data[i]);
		if(int(gimage.data[i]) > maxval)
			maxval = int(gimage.data[i]);
	}

	//Sperate object pixels using thresholding
	ThresFunc();
	//namedWindow( "Thresholded1", WINDOW_AUTOSIZE ); 
    //imshow( "Thresholded1", gimage ); // Show Output image
	reqTask = 1;
	//Find object properties
	RegGrwFunc(100);

	//Clean Image
	CleanGroups();
	//cout <<"\tCleaning Complete\n\n";
	
	for(unsigned int i = 0; i < grpList.size(); i++)
		grpList1.push_back(grpList[i]);
	
	grpList.clear();

	//Load first image, find its properties
	gimage = imread(timage2, CV_LOAD_IMAGE_GRAYSCALE);  
	// Check for invalid input
	if(! gimage.data ) 
	{
        cout << "Could not open or find the image"  << timage2 << std::endl ;
	    return false;
	}

	//Used every where
	noOfPix = gimage.cols*gimage.rows;
	//cout<<"File: "<<timage2<<"\n";
	//Get Basic Image Properties
	for(int i = 0; i < noOfPix; i=i+1)
	{
		//cout<<int(image.data[noOfPix*image.rows+i])<<" ";
		if((int(gimage.data[i])) < minval)
			minval = int(gimage.data[i]);
		if(int(gimage.data[i]) > maxval)
			maxval = int(gimage.data[i]);
	}

	//Sperate object pixels using thresholding
	ThresFunc();
	//namedWindow( "Thresholded2", WINDOW_AUTOSIZE ); 
    //imshow( "Thresholded2", gimage ); // Show Output image
	reqTask = 1;
	//Find object properties
	RegGrwFunc(100);

	//Clean Image
	CleanGroups();
	//cout <<"\tCleaning Complete\n\n";
	FindEucDist();

	fDet.open("pattern_desc.txt");
	fDet << "other\n";
	fDet << ED[0] << " " << ED[1] << " " << ED[2] << " " << ED[3];
	fDet.close();
}

bool DescSeperator(void)
{
	//Read Descriminant file
	ifstream DescFile;                   
    DescFile.open(descfile, ios_base::in);  // open data

    if (!DescFile)  
	{                    
        cout << "Could not open or find the Descriminant file: " << descfile << std::endl ;
	    return false;
    }

	string line;
	std::getline(DescFile, line);
	std::istringstream in(line);
	if(line.compare("other"))
	{
		OtherSeperator();
		return true;
	}
	cout <<"Line\t:"<<line;
	in >> FD[0] >> FD[1] >> FD[2];

	cout << "\nDescriminents read: " << FD[0] << ", " << FD[1] << ", " << FD[2] <<std::endl;

	//Read Image
	gimage = imread(timage1, CV_LOAD_IMAGE_GRAYSCALE);  
	// Check for invalid input
	if(! gimage.data ) 
	{
        cout << "Could not open or find the image file: " << timage1 << std::endl ;
	    return false;
	}

	cout<<"File: "<<timage1<<"\n";
	//Used every where
	noOfPix = gimage.cols*gimage.rows;

	//Get Basic Image Properties
	for(int i = 0; i < noOfPix; i=i+1)
	{
		//cout<<int(image.data[noOfPix*image.rows+i])<<" ";
		if((int(gimage.data[i])) < minval)
			minval = int(gimage.data[i]);
		if(int(gimage.data[i]) > maxval)
			maxval = int(gimage.data[i]);
	}

	//Sperate object pixels using thresholding
	ThresFunc();
	namedWindow( "Thresholded", WINDOW_AUTOSIZE ); 
    imshow( "Thresholded", gimage ); // Show Output image
	reqTask = 1;
	//Find object properties
	RegGrwFunc(70);

	//Clean Groups
	CleanGroups();
	cout <<"\tCleaning Complete\n";

	//Find Projections on Descriminant and find threshold using projections mean
	CalcProjections();
	cout <<"Separating objects using Projection Threshold: " << FDthres <<std::endl;

	//Find Seperate Objects
	outimage1 = imread(timage1, CV_LOAD_IMAGE_GRAYSCALE); 
	outimage2 = imread(timage1, CV_LOAD_IMAGE_GRAYSCALE);
	cout <<"No of Pix: " << noOfPix <<"," <<outimage1.cols*outimage1.rows <<","<<outimage2.cols*outimage2.rows  << "\nImage Pallate Size: " << imagePall.size() <<"\n";
	cout <<"Group Size: " << grpList.size()<<"\n";
	for(int i = 0; i < noOfPix; i++)
	{
		//cout <<"\r"<<imagePall[i];
		if(grpList[imagePall[i]-1].area > 0)
			if(grpList[imagePall[i]-1].proj > FDthres)
				outimage1.data[i] = 250;
			else
				outimage2.data[i] = 250;
		else
		{
			outimage1.data[i] = 250;
			outimage2.data[i] = 250;
		}
	}
	cout <<"Seperation Complete\n";
	//Display and Store images
	string outfile;
	namedWindow( "Seperated Object 1", WINDOW_AUTOSIZE ); 
    imshow( "Seperated Object 1", outimage1 ); // Show Output image
	outfile.assign("sep_OBJ_1_");
	outfile.append(timage1);
	imwrite(outfile,outimage1);

	namedWindow( "Seperated Object 2", WINDOW_AUTOSIZE ); 
    imshow( "Seperated Object 2", outimage2 ); // Show Output image
	outfile.assign("sep_OBJ_2_");
	outfile.append(timage1);
	imwrite(outfile,outimage2);
	waitKey(0); // Wait for a keystroke to close the windows
	return true;
}

bool CalcProjections(void)
{
	double denom = sqrt(FD[0]*FD[0]+FD[1]*FD[1]+FD[2]*FD[2]);
	double tempd1, compact;
	FDthres = 0;
	int count = 0;
	//cout << "\nProjections:\n";
	cout << "\nDifferentiators:\n";
	for(unsigned int i = 0; i < grpList.size(); i++)
	{
		if(grpList[i].area > 0)
		{
			compact = (grpList[i].perimtr*grpList[i].perimtr/grpList[i].area);
			tempd1 = abs(grpList[i].area*FD[0]+grpList[i].perimtr*FD[1]+compact*FD[2]);
			//cout << tempd1 <<": ";
			tempd1 = tempd1/denom;
			//grpList[i].proj = sqrt(grpList[i].area*grpList[i].area+grpList[i].perimtr*grpList[i].perimtr+compact*compact - tempd1*tempd1);
			grpList[i].proj = tempd1;
			FDthres += grpList[i].proj;
			//cout <<grpList[i].proj<<", "<<FDthres<<"\n";
			cout<<"Area: "<<grpList[i].area<<"\t Perimeter: "<<grpList[i].perimtr<<"\tCompactness: "<<compact<<", "<<"\n";
			count++;
		}
	}
	cout << "\nCount: "<<count<<"\n";
	FDthres = FDthres/count;
	/*
	cout << "\nStarting Threshold: "<<FDthres<<"\n";
	bool cont = true;
	double mean1, mean2, newThresh, count1, count2;
	count = 0;
	while((cont)&&(count++ < 10))
	{
		mean1 = 0;
		mean2 = 0;
		count1 = 0;
		count2 = 0;
		for(unsigned int i  = 0; i < grpList.size(); i++)
		{
			if(grpList[i].area > 0)
			{
				if(grpList[i].proj > FDthres)
				{
					mean1 += grpList[i].proj;
					count1++;
				}
				else
				{
					mean2 += grpList[i].proj;
					count2++;
				}
			}
		}
		mean1 = mean1/count1;
		mean2 = mean2/count2;
		newThresh = (mean1 + mean2)/2;
		if(newThresh == FDthres)
			cont = false;
		else
			FDthres = newThresh;
	}
	*/
	cout << "\nFinal Threshold: "<<FDthres<<"\n";
	//FDthres = 490;
	
	return true;
}

void print2dMatrix(float mat[3][3])
{
	for(int x = 0; x < 3; x++)
	{
		for(int y = 0; y < 3; y++)
		{
			cout <<mat[x][y] << "  ";
		}
		cout <<"\n";
	}
}
void print1dMatrix(float mat[3])
{
	for(int x = 0; x < 3; x++)
	{
			cout <<mat[x] << "  ";
	}
		cout <<"\n";
}