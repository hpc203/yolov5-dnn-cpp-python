#include "yolo.h"

YOLO::YOLO(Net_config config)
{
	cout << "Net use " << config.netname << endl;
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	strcpy_s(this->netname, config.netname.c_str());

	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);

	string modelFile = this->netname;
	modelFile += ".onnx";
	this->net = readNet(modelFile);
}

void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void YOLO::sigmoid(Mat* out, int length)
{
	float* pdata = (float*)(out->data);
	int i = 0; 
	for (i = 0; i < length; i++)
	{
		pdata[i] = 1.0 / (1 + expf(-pdata[i]));
	}
}

void YOLO::detect(Mat& frame)
{
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	
	/////generate proposals
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = this->classes.size() + 5, c = 0;
	for (n = 0; n < 3; n++)   ///尺度
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		int area = num_grid_x * num_grid_y;
		this->sigmoid(&outs[n], 3 * nout * area);
		for (q = 0; q < 3; q++)    ///anchor数
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			float* pdata = (float*)outs[n].data + q * nout * area;  
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = pdata[4 * area + i * num_grid_x + j];
					if (box_score > this->objThreshold)
					{
						float max_class_socre = 0, class_socre = 0;
						int max_class_id = 0;
						for (c = 0; c < this->classes.size(); c++) //// get max socre
						{
							class_socre = pdata[(c + 5) * area + i * num_grid_x + j];
							if (class_socre > max_class_socre)
							{
								max_class_socre = class_socre;
								max_class_id = c;
							}
						}
						
						if (max_class_socre > this->confThreshold)
						{
							float cx = (pdata[i * num_grid_x + j] * 2.f - 0.5f + j) * this->stride[n];  ///cx
							float cy = (pdata[area + i * num_grid_x + j] * 2.f - 0.5f + i) * this->stride[n];   ///cy
							float w = powf(pdata[2 * area + i * num_grid_x + j] * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(pdata[3 * area + i * num_grid_x + j] * 2.f, 2.f) * anchor_h;  ///h
							
							int left = (cx - 0.5*w)*ratiow;
							int top = (cy - 0.5*h)*ratioh;   ///坐标还原到原图上

							classIds.push_back(max_class_id);
							confidences.push_back(max_class_socre);
							boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
						}	
					}	
				}
			}
		}
	}
	
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

int main()
{
	YOLO yolo_model(yolo_nets[0]);
	string imgpath = "bus.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);
	
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}