

#include "stdafx.h"


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cv.h>
#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <XnCppWrapper.h>
#include "3dsloader.h"
#include "glut.h"
#include <vector>
#include "glaux.h"
using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


static XnStatus eResult = XN_STATUS_OK;
static xn::Context mContext;
static XnMapOutputMode mapMode;
static xn::DepthGenerator mDepthGenerator;
static xn::ImageGenerator mImageGenerator;
static IplImage img,img_valid,img_normalization,img_energy,img_depth_icp,img_doub_filter,img_rgb,img_doub_normal;
static CvMat mat;
static cv::Mat ma;
static CvMat mat_valid;
static cv::Mat ma_valid;
static CvMat mat_normalization;
static cv::Mat ma_normalzation;
static CvMat mat_energy;
static cv::Mat ma_energy;
static CvMat mat_depth_icp;
static cv::Mat ma_depth_icp;
static CvMat mat_doub_filter;
static cv::Mat ma_doub_filter;
static CvMat mat_img_rgb;
static cv::Mat ma_img_rgb;
static CvMat mat_doub_normal;
static cv::Mat ma_doub_normal;

static size_t global_worksize[]={160,120};
static size_t local_worksize[]={128,1};

static size_t sum_mat_worksize[]={120,36};
static size_t sum_vec_worksize[]={120,6};

static size_t sum_mat_lworksize[]={1,1};
static size_t sum_vec_lworksize[]={240,1};

static size_t doub_sum_mat_worksize[]={4624,36};
static size_t doub_sum_vec_worksize[]={4624,6};

static size_t doub_sum_sum_mat_worksize[]={68,36};
static size_t doub_sum_sum_vec_worksize[]={68,6};

static size_t doub_sum_sum_sum_mat_worksize[]={1,36};
static size_t doub_sum_sum_sum_vec_worksize[]={1,6};
//static size_t doub_sum_vec_worksize[]={32,6};

static size_t sum_sum_mat_worksize[]={36};
static size_t sum_sum_vec_worksize[]={6};

static size_t pre_bit_sum_mat_worksize[]={640*480,36};
static size_t pre_bit_sum_vec_worksize[]={640*480,6};

static size_t bit_sum_mat_worksize[]={131072,36};
static size_t bit_sum_vec_worksize[]={131072,6};

static size_t ire_sum_mat_worksize[]={36,524288};
static size_t ire_sum_vec_worksize[]={6,524288};

static size_t ire_sum_sum_mat_worksize[]={36,2048};
static size_t ire_sum_sum_vec_worksize[]={6,2048};

static size_t ire_sum_sum_sum_mat_worksize[]={36,256};
static size_t ire_sum_sum_sum_vec_worksize[]={6,256};

static size_t ire_sum_mat_lworksize[]={1,256};
static size_t ire_sum_vec_lworksize[]={1,256};

static size_t sub_depth_worksize[]={320,240};
static size_t sub_sub_depth_worksize[]={160,120};
static size_t mult_worksize[]={640,480};

static float *point_cloud=new float[160*120*3];
static float *normalization=new float[160*120*3];
static float *test_point=new float[160*120*6];
static float *energy=new float[160*120];
static float *depth_icp=new float[160*120];
//float *depth_icp=new float[640*480];
static float *doub_normal=new float[160*120*3];
static float *area=new float[160*120*4];

static float *filter=new float[160*120];

static float *buffer1_mat=new float[160*120*36];
static float *buffer1_vec=new float[160*120*6];
static float *buffer2_mat=new float[160*120*36];
static float *buffer2_vec=new float[160*120*6];

//static float *area_test=new float[640*480*4];

//static float *xyzuv=new float[640*480*5];
//area[320*4+240*640*4]=22;

float *ATA=new float[160*120*36];
float *ATb=new float[160*120*6];

//float *ATA_sum=new float[480*36];
//float *ATb_sum=new float[480*6];

static float *ATA_sum_sum=new float[36];
static float *ATb_sum_sum=new float[6];

static float *rotvec=new float[3];
static float *transvec=new float[3];

static float *L=new float[36];//[(1+LEN)*LEN/2];
static float *LT=new float[36];//[(1+LEN)*LEN/2];

static int point_num=0;
static unsigned int half_area=5;

static float *y=new float[5];
static float *x=new float[5];

static const XnDepthPixel*  pDepthMap;
static const XnRGB24Pixel*  pImageMap;
static int i=0;
static int key=0;
static float limit=0;

static unsigned short *result=new unsigned short[160*120];
static unsigned short *result2=new unsigned short[640*480];
static unsigned short *result_valid=new unsigned short[160*120];
static unsigned short *valid_icp=new unsigned short[160*120];
static unsigned short *result_valid_last=new unsigned short[160*120];
static float *errrr=new float[160*120];
static int *con=new int[160*120];

static float *sub_depth=new float[320*240];
static float *sub_point=new float[320*240];
static float *sub_sub_depth=new float[160*120];
static float *sub_sub_normal=new float[160*120*3];
static float *sub_sub_point=new float[160*120*3];

clock_t time_this,time_last,time_temp;
float fps=0;
int frame_sum=0;

static cl_int err;
static cl_uint num;

static cl_command_queue queue;
static cl_mem cl_a;
//	static cl_mem cl_b;
static cl_mem cl_res;
static cl_mem cl_point_num;
static cl_mem cl_point;
static cl_mem cl_point_last;
static cl_mem cl_valid;
static cl_mem cl_valid_last;
static cl_mem cl_normalization;
static cl_mem cl_normalization_last;
static cl_mem cl_test_point;
static cl_mem cl_energy;
static cl_mem cl_half_area;

static cl_mem cl_area;
static cl_mem cl_point_icp;
static cl_mem cl_valid_icp;
static cl_mem cl_normalization_icp;
static cl_mem cl_ATA;
static cl_mem cl_ATb;
static cl_mem cl_rotvec1;
static cl_mem cl_rotvec2;
static cl_mem cl_rotvec3;
static cl_mem cl_transvec1;
static cl_mem cl_transvec2;
static cl_mem cl_transvec3;
static cl_mem cl_ATA_sum;
static cl_mem cl_ATb_sum;
static cl_mem cl_ATA_sum_sum;
static cl_mem cl_ATb_sum_sum;
static cl_mem cl_depth_icp;
static cl_mem cl_testnum;
static cl_mem cl_matsum1;
static cl_mem cl_matsum2;
static cl_mem cl_vecsum1;
static cl_mem cl_vecsum2;
static cl_mem cl_xyzuv;
static cl_mem cl_doub_filter;
static cl_mem cl_doub_normal;
static cl_mem cl_buffer1_mat;
static cl_mem cl_buffer1_vec;
static cl_mem cl_buffer2_mat;
static cl_mem cl_buffer2_vec;
static cl_mem cl_limit;
static cl_mem cl_ire_sum_mat;
static cl_mem cl_ire_sum_vec;
static cl_mem cl_ire_sum_sum_mat;
static cl_mem cl_ire_sum_sum_vec;
static cl_mem cl_ire_sum_sum_sum_mat;
static cl_mem cl_ire_sum_sum_sum_vec;
static cl_mem cl_mm;
static cl_mem cl_errr;
static cl_mem cl_con;
static cl_mem cl_sub_depth;
static cl_mem cl_sub_point;
static cl_mem cl_sub_sub_depth;
static cl_mem cl_sub_sub_normal;
static cl_mem cl_sub_sub_point;
static cl_program program;
static cl_kernel addone;
static cl_kernel range_up;
static cl_kernel range_down;
static cl_kernel range_left;
static cl_kernel range_right;
static cl_kernel icp1;
static cl_kernel icp2;
static cl_kernel icp3;
static cl_kernel icp4;
static cl_kernel sum_mat;
static cl_kernel sum_vec;
static cl_kernel sum_sum_mat;
static cl_kernel sum_sum_vec;
static cl_kernel update;
static cl_kernel doub_sum_mat;
static cl_kernel doub_sum_vec;
static cl_kernel doub_sum_sum_mat;
static cl_kernel doub_sum_sum_vec;
static cl_kernel doub_sum_sum_sum_mat;
static cl_kernel doub_sum_sum_sum_vec;
static cl_kernel doub_filter;
static cl_kernel filter_normal;
static cl_kernel bit_sum_mat1;
static cl_kernel bit_sum_vec1;
static cl_kernel bit_sum_mat2;
static cl_kernel bit_sum_vec2;
static cl_kernel pre_bit_sum_mat;
static cl_kernel pre_bit_sum_vec;
static cl_kernel ire_sum_mat;
static cl_kernel ire_sum_vec;
static cl_kernel ire_sum_sum_mat;
static cl_kernel ire_sum_sum_vec;
static cl_kernel ire_sum_sum_sum_mat;
static cl_kernel ire_sum_sum_sum_vec;
static cl_kernel add_depth;
static cl_kernel sub_depth_kernel;
static cl_kernel sub_sub_depth_kernel;
static cl_kernel mult_kernel;
static xn::DepthMetaData dmdata;
static xn::ImageMetaData imdata;

static float rot_global[3],trans_global[3],rot_icp[3],trans_icp[3];
static float M[16],MM[16],MMM[16];

static float testnum[64*6]={1,2,1,1,1,1,1,3,1,1,1,1,1,4,1,1,1,1,1,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static GLuint texName;
cv::Mat  m_rgb8u( 480,640,CV_8UC3);
cv::Mat  m_ImageShow( 480,640,CV_8UC3);


////// 3ds loader//////
CLoad3DS loader;
///////////////////////


int init_texture()                    //初始化纹理，摄像头捕获的彩色图像，贴在纹理上作为背景
{
	//opengl Initialize

	//获得屏幕分辨率
	int nFullWidth=640; 
	int nFullHeight=480; 

	//初始化视频设备
	//PalCapture.Init( 0, false, 768, 576, 3, 30, true);

	//开启纹理映射和设置纹理参数
	
	glGenTextures(1, &texName);
	glBindTexture(GL_TEXTURE_2D, texName);

	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);       //线性滤波
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);       //线性滤波
	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);                    
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);

	//生成空纹理
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 320, 480, 0, GL_RGB, GL_UNSIGNED_SHORT,NULL );

	return 0;
}


void draw_text(const XnUInt8 *img)              //绘制纹理，在myDisplay()中调用
{
	glEnable(GL_TEXTURE_2D); 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	  //glShadeModel(GL_FLAT);
	glPushMatrix();
	glLoadIdentity();

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glBindTexture(GL_TEXTURE_2D, texName);
	
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 320, 480, GL_RGB, GL_UNSIGNED_SHORT, img);

	//glActiveTextureARB(0);
	//glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-2625.0586f/2,-1969.5523f/2,5000.0f/2);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(2625.0586f/2,-1969.5523f/2,5000.0f/2);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(2625.0586f/2,1969.5523f/2,5000.0f/2);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-2625.0586f/2,1969.5523f/2,5000.0f/2);
	glEnd();

	//glutSolidTeapot(100);
	glPopMatrix();
	glDisable(GL_TEXTURE_2D); 
	glFlush();
	
	//cvReleaseImage(&pTempImg);
}


cl_program load_program(cl_context context, const char* filename)
{
std::ifstream in(filename, std::ios_base::binary);
if(!in.good()) {return 0;}

// get file length
in.seekg(0, std::ios_base::end);
size_t length = in.tellg();
in.seekg(0, std::ios_base::beg);

// read program source
std::vector<char> data(length + 1);
in.read(&data[0], length);
data[length] = 0;

// create and build program 
const char* source = &data[0];
cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
if(program == 0) {
return 0;
}
cl_int err;
if((err=clBuildProgram(program, 0, 0, 0, 0, 0)) != CL_SUCCESS) {

if (err==CL_INVALID_PROGRAM)
{
	cout<<"1\n";
}
else if (err==CL_INVALID_VALUE)
{
	cout<<"2\n";
}
else if (err==CL_INVALID_DEVICE)
{
	cout<<"3\n";
}
else if (err==CL_INVALID_BINARY)
{
	cout<<"4\n";
}
else if (err==CL_INVALID_BUILD_OPTIONS)
{
	cout<<"5\n";
}
else if (err==CL_INVALID_OPERATION)
{
	cout<<"6\n";
}
else if (err==CL_BUILD_PROGRAM_FAILURE)
{
	cout<<"7\n";
}
else if (err==CL_OUT_OF_HOST_MEMORY)
{
	cout<<"8\n";
}
	return 0;
}

return program;
}

void CheckOpenNIError( XnStatus eResult, string sStatus )
{
  if( eResult != XN_STATUS_OK )
    cerr << sStatus << " Error: " << xnGetStatusString( eResult ) << endl;
}

//串行完全Chlolesky分解
int LEN=6;
int complete_cholesky_decompose(float *A,float* L,float* LT)
 {
     if(NULL==A)
         return NULL;
  
//     float *L=malloc_matrix();
//     clear_matrix(L);
  
     int i,j,k,m;
  
     float *w=new float[LEN];
	 for (int i=0;i<LEN;i++)
	 {
		 w[i]=0;
	 }
  
//     clear_vector(w);//清除向量
  
     for(i=0;i<LEN;i++){
         for(m=i;m<LEN;m++){
             w[m]=A[i*LEN+m];
         }
  
         for(k=0;k<i;k++){
             float temp=L[k*LEN+i];
             if(temp!=0){
                 for(m=i;m<LEN;m++){
                     w[m] -= temp*L[k*LEN+m];
                 }
             }
         }
  
         w[i]=sqrt(w[i]);
         for(m=i+1;m<LEN;m++){
             w[m] /=w[i];
         }
  
         for(m=i;m<LEN;m++){
             L[i*LEN+m]=w[m];
			 LT[m*LEN+i]=w[m];
         }
  
		 for (int i=0;i<LEN;i++)
		 {
			 w[i]=0;
		 }
//         clear_vector(w);
     }
  
     return 1;
 }

static int key_up=0;
static int key_down=0;
static int key_left=0;
static int key_right=0;
static int key_lift=0;
static int key_drop=0;
static int key_q=0;
static int key_a=0;
static int key_w=0;
static int key_s=0;
static int key_e=0;
static int key_d=0;
static int key_num=0;

void keypress(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		key_up=1;
		key_down=0;
		key_left=0;
		key_right=0;
		key_lift=0;
		key_drop=0;
		//cout<<"ffff";
		break;
	case GLUT_KEY_DOWN:
		key_up=0;
		key_down=1;
		key_left=0;
		key_right=0;
		key_lift=0;
		key_drop=0;
		break;
	case GLUT_KEY_LEFT:
		key_up=0;
		key_down=0;
		key_left=1;
		key_right=0;
		key_lift=0;
		key_drop=0;
		break;
	case GLUT_KEY_RIGHT:
		key_up=0;
		key_down=0;
		key_left=0;
		key_right=1;
		key_lift=0;
		key_drop=0;
		break;
	case GLUT_KEY_PAGE_UP:
		key_up=0;
		key_down=0;
		key_left=0;
		key_right=0;
		key_lift=1;
		key_drop=0;
		break;
	case GLUT_KEY_PAGE_DOWN:
		key_up=0;
		key_down=0;
		key_left=0;
		key_right=0;
		key_lift=0;
		key_drop=1;
		break;
	}

}



void normal_key(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 113://q
		key_q=1;
		key_a=0;
		key_w=0;
		key_s=0;
		key_e=0;
		key_d=0;
		break;
	case 97://a
		key_q=0;
		key_a=1;
		key_w=0;
		key_s=0;
		key_e=0;
		key_d=0;
		break;
	case 119://w
		key_q=0;
		key_a=0;
		key_w=1;
		key_s=0;
		key_e=0;
		key_d=0;
		break;
	case 115://s
		key_q=0;
		key_a=0;
		key_w=0;
		key_s=1;
		key_e=0;
		key_d=0;
		break;
	case 101://e
		key_q=0;
		key_a=0;
		key_w=0;
		key_s=0;
		key_e=1;
		key_d=0;
		break;
	case 100://d
		key_q=0;
		key_a=0;
		key_w=0;
		key_s=0;
		key_e=0;
		key_d=1;
		break;
	case 49:
		key_num=1;
		break;
	case 50:
		key_num=2;
		break;
	case 51:
		key_num=3;
		break;
	case 52:
		key_num=4;
		break;
	case 53:
		key_num=5;
		break;
	default:
		break;
	}
}


void myDisplay()
{
	
	glEnable(GL_DEPTH_TEST);//开启深度测试
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);//清理缓冲区

	draw_text(m_rgb8u.data);//绘制背景纹理
	glMatrixMode(GL_PROJECTION);//设置摄像机以投影方式渲染画面
	glLoadIdentity();//位置矩阵设为单位矩阵
	gluPerspective(43,1.33333,1,1000000);//设置投影参数(fov,近裁剪距离,远裁剪距离)
	gluLookAt(0,0,0,0,0,1000,0,-1,0);//设置摄像机观看方式(摄像机坐标,观看目标坐标,定义“上”矢量)

	glMatrixMode(GL_MODELVIEW);

	glPushMatrix();//保存当前摄像机矩阵，操作模型矩阵
	glRotatef(180.0f,0.0f,0.0f,1.0f);
	
	cout << "matrix pushed";
	
	switch (key_num)
	{
	case 1:
		{  
			cout << "number key 1 input";

			GLfloat earth_mat_ambient[]  = {0.6f, 0.1f, 0.1f, 0.2f};  
			GLfloat earth_mat_diffuse[]  = {0.6f, 0.2f, 0.2f, 0.2f};  
			GLfloat earth_mat_specular[] = {0.6f, 0.0f, 0.2f, 0.2f};  
			GLfloat earth_mat_emission[] = {0.6f, 0.0f, 0.2f, 0.0f};  
			GLfloat earth_mat_shininess  = 120.0f;  

			glMaterialfv(GL_FRONT, GL_AMBIENT,   earth_mat_ambient);  
			glMaterialfv(GL_FRONT, GL_DIFFUSE,   earth_mat_diffuse);  
			glMaterialfv(GL_FRONT, GL_SPECULAR,  earth_mat_specular);  
			glMaterialfv(GL_FRONT, GL_EMISSION,  earth_mat_emission);  
			glMaterialf (GL_FRONT, GL_SHININESS, earth_mat_shininess);  

		} 
		cout << "starting to load 3ds";
		loader.show3ds(0,0,0,0,1);
		cout << "finish to load 3ds";
		break;
	case 2:
		{  
			GLfloat earth_mat_ambient[]  = {0.2f, 0.1f, 0.5f, 0.2f};  
			GLfloat earth_mat_diffuse[]  = {0.2f, 0.1f, 0.5f, 0.2f};  
			GLfloat earth_mat_specular[] = {0.2f, 0.1f, 0.5f, 0.2f};  
			GLfloat earth_mat_emission[] = {0.2f, 0.1f, 0.5f, 0.0f};  
			GLfloat earth_mat_shininess  = 120.0f;  

			glMaterialfv(GL_FRONT, GL_AMBIENT,   earth_mat_ambient);  
			glMaterialfv(GL_FRONT, GL_DIFFUSE,   earth_mat_diffuse);  
			glMaterialfv(GL_FRONT, GL_SPECULAR,  earth_mat_specular);  
			glMaterialfv(GL_FRONT, GL_EMISSION,  earth_mat_emission);  
			glMaterialf (GL_FRONT, GL_SHININESS, earth_mat_shininess);  

		} 
		loader.show3ds(1,0,0,0,1);
		break;
	case 3:
		{  
			GLfloat earth_mat_ambient[]  = {0.7f, 0.7f, 0.0f, 0.0f};  
			GLfloat earth_mat_diffuse[]  = {0.7f, 0.7f, 0.0f, 0.0f};  
			GLfloat earth_mat_specular[] = {0.7f, 0.7f, 0.0f, 0.0f};  
			GLfloat earth_mat_emission[] = {0.5f, 0.5f, 0.3f, 0.0f};  
			GLfloat earth_mat_shininess  = 120.0f;  

			glMaterialfv(GL_FRONT, GL_AMBIENT,   earth_mat_ambient);  
			glMaterialfv(GL_FRONT, GL_DIFFUSE,   earth_mat_diffuse);  
			glMaterialfv(GL_FRONT, GL_SPECULAR,  earth_mat_specular);  
			glMaterialfv(GL_FRONT, GL_EMISSION,  earth_mat_emission);  
			glMaterialf (GL_FRONT, GL_SHININESS, earth_mat_shininess);  

		}
		loader.show3ds(2,0,0,0,1);
		break;
	case 4:
		{  
			GLfloat earth_mat_ambient[]  = {0.4f, 0.4f, 0.4f, 0.5f};  
			GLfloat earth_mat_diffuse[]  = {0.4f, 0.4f, 0.4f, 0.5f};  
			GLfloat earth_mat_specular[] = {0.5f, 0.0f, 0.98f, 0.9f};  
			GLfloat earth_mat_emission[] = {0.3f, 0.3f, 0.3f, 0.0f};  
			GLfloat earth_mat_shininess  = 60.0f;  

			glMaterialfv(GL_FRONT, GL_AMBIENT,   earth_mat_ambient);  
			glMaterialfv(GL_FRONT, GL_DIFFUSE,   earth_mat_diffuse);  
			glMaterialfv(GL_FRONT, GL_SPECULAR,  earth_mat_specular);  
			glMaterialfv(GL_FRONT, GL_EMISSION,  earth_mat_emission);  
			glMaterialf (GL_FRONT, GL_SHININESS, earth_mat_shininess);  

		}
		glTranslatef(0,100,0);
		loader.show3ds(3,0,0,0,0.01);
		break;
	case 5:
		{  
			GLfloat earth_mat_ambient[]  = {0.4f, 0.4f, 0.4f, 0.5f};  
			GLfloat earth_mat_diffuse[]  = {0.4f, 0.4f, 0.4f, 0.5f};  
			GLfloat earth_mat_specular[] = {0.5f, 0.0f, 0.98f, 0.9f};  
			GLfloat earth_mat_emission[] = {0.3f, 0.3f, 0.3f, 0.0f};  
			GLfloat earth_mat_shininess  = 90.0f;  

			glMaterialfv(GL_FRONT, GL_AMBIENT,   earth_mat_ambient);  
			glMaterialfv(GL_FRONT, GL_DIFFUSE,   earth_mat_diffuse);  
			glMaterialfv(GL_FRONT, GL_SPECULAR,  earth_mat_specular);  
			glMaterialfv(GL_FRONT, GL_EMISSION,  earth_mat_emission);  
			glMaterialf (GL_FRONT, GL_SHININESS, earth_mat_shininess);  

		}
		glTranslatef(0,60,0);
		glutSolidTeapot(100);
		break;
	default:
		break;
	}
	

	glPopMatrix();//回到摄像机矩阵

	glFlush();//更新缓冲区
	glutSwapBuffers();//交换缓冲区
}

void myIdle()//cpu空闲时函数，在cpu空闲时被调用，所有icp计算及一些其他相关计算都在这函数里进行
{

	////////////////////////////fps/////////////
	time_last=time_this;
	time_this=clock();
	if (frame_sum==10)
	{
		fps=10000.0f/(time_this-time_last);
		frame_sum=0;
		//cout<<"fps:"<<fps<<"\n";
	}
	frame_sum++;
	////////////////////////////////////////////////////////////////


	//////////////////初始化位移和旋转分量////////////////////////
	trans_icp[0]=0;
	trans_icp[1]=0;
	trans_icp[2]=0;
	rot_icp[0]=0;
	rot_icp[1]=0;
	rot_icp[2]=0;
	//trans_global[0]=0;
	//trans_global[1]=0;
	//trans_global[2]=0;
   //////////////////////////////////////////////////////////////////
	

	////////openni start/////
	eResult = mContext.WaitAndUpdateAll();//通过openni中的上下文变量，更新所有深度和彩色信息
	//cvWaitKey(100);
	if( eResult == XN_STATUS_OK )
	{
		// 5. get the depth map

		//pDepthMap = mDepthGenerator.GetDepthMap();
		pImageMap = mImageGenerator.GetRGB24ImageMap();//获取彩色图像

		// 6. Do something with depth map

		mDepthGenerator.GetMetaData(dmdata);//获取深度图像的数据部分
		mImageGenerator.GetMetaData(imdata);//获取彩色图像的数据部分
//cvWaitKey(100);
		memcpy(m_rgb8u.data,imdata.Data(),640*480*3);//拷贝图像数据到m_rgb8u中，以便于转换
		cvtColor(m_rgb8u,m_ImageShow,CV_RGB2BGR);//转换RGB存储方式为BGR，便于opencv显示
		imshow("image", m_ImageShow);//显示图像

//cvWaitKey(100);

//////openni ends/////



	/////////////////////////////////kernal/////////////////////////

	//clEnqueueWriteBuffer将一个写显存命令加入opencl执行队列中
	//clEnqueueReadBuffer将一个读显存命令加入opencl执行队列中
	//clEnqueueNDRangeKernel将一个kernal程序加入opencl执行队列中

	err=clEnqueueWriteBuffer(queue,cl_a,CL_TRUE,0,sizeof(unsigned short)*640*480,(unsigned short*)dmdata.Data(),0,0,0);
	err=clEnqueueReadBuffer(queue,cl_a,CL_TRUE,0,sizeof(unsigned short)*640*480, result2,0,0,0);
	err=clEnqueueNDRangeKernel(queue, sub_depth_kernel, 2, 0, sub_depth_worksize, 0, 0, 0, 0);
	err=clEnqueueNDRangeKernel(queue, sub_sub_depth_kernel, 2, 0, sub_sub_depth_worksize, 0, 0, 0, 0);
	err=clEnqueueReadBuffer(queue,cl_point,CL_TRUE,0,sizeof(float)*160*120*3, sub_sub_point,0,0,0);
	err=clEnqueueReadBuffer(queue,cl_normalization,CL_TRUE,0,sizeof(float)*160*120*3, sub_sub_normal,0,0,0);
	err=clEnqueueReadBuffer(queue,cl_valid,CL_TRUE,0,sizeof(unsigned short)*160*120,result_valid,0,0,0);
	err=clEnqueueReadBuffer(queue,cl_valid_last,CL_TRUE,0,sizeof(unsigned short)*160*120,result_valid_last,0,0,0);

	//////////////////////////统计相邻两帧深度图中，有效点的并集的总点数，for test////////////////////
	int v_sum=0;
	for (int i=0;i<160;i++)
	{
		for (int j=0;j<120;j++)
		{
			if ((result_valid[i+160*j]>0)&(result_valid_last[i+160*j]>0))
			{
				v_sum++;
			}
		}
	}
	//cout<<"valid_sum:"<<v_sum<<"\n";
	///////////////////////////////////////////////////////////////////////////////////////////////////////


	glMatrixMode(GL_MODELVIEW);//

	//glutKeyboardFunc(keypress);
	//glutSpecialFunc(&keypress);
	//glutSpecialFunc(&keypress);

	//////////////////////////////////////////设置按键控制模型///////////////////////////////
	if (key_up==1)
	{
		glTranslated(0,0,50);
	}
	if (key_down==1)
	{
		glTranslated(0,0,-50);
		//cvWaitKey(0);
	}
	if (key_left==1)
	{
		glTranslated(-50,0,0);
	}
	if (key_right==1)
	{
		glTranslated(50,0,0);
	}
	if (key_lift==1)
	{
		glTranslated(0,-50,0);
	}
	if (key_drop==1)
	{
		glTranslated(0,50,0);
	}
	key_up=0;
	key_down=0;
	key_left=0;
	key_right=0;
	key_lift=0;
	key_drop=0;

	if (key_q==1)
	{
		glRotated(10,1,0,0);
	}
	if (key_a==1)
	{
		glRotated(-10,1,0,0);
	}
	if (key_w==1)
	{
		glRotated(10,0,1,0);
	}
	if (key_s==1)
	{
		glRotated(-10,0,1,0);
	}
	if (key_e==1)
	{
		glRotated(10,0,0,1);
	}
	if (key_d==1)
	{
		glRotated(-10,0,0,1);
	}
	key_q=0;
	key_a=0;
	key_w=0;
	key_s=0;
	key_e=0;
	key_d=0;
	//key_num=0;


///////////////////////////////////////////////////////////////


	MM[0]=1;			MM[1]=0;					MM[2]=0;						MM[3]=0;
	MM[4]=0;			MM[5]=1;					MM[6]=0;						MM[7]=0;
	MM[8]=0;			MM[9]=0;					MM[10]=1;						MM[11]=0;
	MM[12]=0;			MM[13]=0;					MM[14]=0;						MM[15]=1;

	MMM[0]=1;			MMM[1]=0;					MMM[2]=0;						MMM[3]=0;
	MMM[4]=0;			MMM[5]=1;					MMM[6]=0;						MMM[7]=0;
	MMM[8]=0;			MMM[9]=0;					MMM[10]=1;						MMM[11]=0;
	MMM[12]=0;			MMM[13]=0;					MMM[14]=0;						MMM[15]=1;

	float tim[4];
	for (int i=0;i<15;i++)//15次icp循环只完成了一次相邻两帧的矩阵计算，算得的结果保存在MM[16]中，M[16]中保存的是每次icp计算的结果，MM由15个M累乘而得
		                  //MMM是一个相对我们宏观真实世界静止的坐标系下的矩阵（称为全局矩阵）
	{

		//err=clEnqueueWriteBuffer(queue,cl_testnum,CL_TRUE,0,sizeof(float)*64*6,testnum,0,0,0);

		err = clEnqueueNDRangeKernel(queue, icp1, 2, 0, global_worksize, 0, 0, 0, 0);

		err = clEnqueueNDRangeKernel(queue, sum_mat, 2, 0, sum_mat_worksize, 0, 0, 0, 0);
		err = clEnqueueNDRangeKernel(queue, sum_vec, 2, 0, sum_vec_worksize, 0, 0, 0, 0);

		err = clEnqueueNDRangeKernel(queue, sum_sum_mat, 1, 0, sum_sum_mat_worksize, 0, 0, 0, 0);
		err = clEnqueueNDRangeKernel(queue, sum_sum_vec, 1, 0, sum_sum_vec_worksize, 0, 0, 0, 0);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		err	=clEnqueueReadBuffer(queue, cl_ATA_sum_sum, CL_TRUE, 0, sizeof(float)*36 ,ATA_sum_sum, 0, 0, 0);
		err=clEnqueueReadBuffer(queue, cl_ATb_sum_sum, CL_TRUE, 0, sizeof(float)*6 ,ATb_sum_sum, 0, 0, 0);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		clFinish(queue);
		//cvWaitKey(250);
		////////////////////////////////////////////////////

		complete_cholesky_decompose(ATA_sum_sum,L,LT);//将矩阵ATA_sum_sum，通过乔列斯基分解法，分解成L和LT的乘积，L和LT都是下三角矩阵

		//分别求解两个下三角矩阵的矩阵方程，最终求得6维向量x[6]
		y[0]=ATb_sum_sum[0]/LT[0];
		y[1]=(ATb_sum_sum[1]-LT[6+0]*y[0])/LT[6+1];
		y[2]=(ATb_sum_sum[2]-LT[6*2+0]*y[0]-LT[6*2+1]*y[1])/LT[6*2+2];
		y[3]=(ATb_sum_sum[3]-LT[6*3+0]*y[0]-LT[6*3+1]*y[1]-LT[6*3+2]*y[2])/LT[6*3+3];
		y[4]=(ATb_sum_sum[4]-LT[6*4+0]*y[0]-LT[6*4+1]*y[1]-LT[6*4+2]*y[2]-LT[6*4+3]*y[3])/LT[6*4+4];
		y[5]=(ATb_sum_sum[5]-LT[6*5+0]*y[0]-LT[6*5+1]*y[1]-LT[6*5+2]*y[2]-LT[6*5+3]*y[3]-LT[6*5+4]*y[4])/LT[6*5+5];

		x[5]=y[5]/L[6*5+5];
		x[4]=(y[4]-L[6*4+5]*x[5])/L[6*4+4];
		x[3]=(y[3]-L[6*3+5]*x[5]-L[6*3+4]*x[4])/L[6*3+3];
		x[2]=(y[2]-L[6*2+5]*x[5]-L[6*2+4]*x[4]-L[6*2+3]*x[3])/L[6*2+2];
		x[1]=(y[1]-L[6*1+5]*x[5]-L[6*1+4]*x[4]-L[6*1+3]*x[3]-L[6*1+2]*x[2])/L[6*1+1];
		x[0]=(y[0]-L[6*0+5]*x[5]-L[6*0+4]*x[4]-L[6*0+3]*x[3]-L[6*0+2]*x[2]-L[6*0+1]*x[1])/L[6*0+0];

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		clFinish(queue);

		//将6维向量x[6]转换为三个位移分量和三个旋转分量，并去除明显错误的分量
			if ((x[3]<100)&(x[3]>-100))
			{
				trans_icp[0]=x[3];
			}
			else
			{
				trans_icp[0]=0;
			}
			if ((x[4]<100)&(x[4]>-100))
			{
				trans_icp[1]=x[4];
			}
			else
			{
				trans_icp[1]=0;
			}
			if ((x[5]<100)&(x[5]>-100))
			{
				trans_icp[2]=x[5];
			}
			else
			{
				trans_icp[2]=0;
			}
			if ((x[2]<1)&(x[2]>-1))
			{
				rot_icp[0]=x[2];
			}
			else
			{
				rot_icp[0]=0;
			}
			if ((x[0]<1)&(x[0]>-1))
			{
				rot_icp[1]=x[0];
			}
			else
			{
				rot_icp[1]=0;
			}
			if ((x[1]<1)&(x[1]>-1))
			{
				rot_icp[2]=x[1];
			}
			else
			{
				rot_icp[2]=0;
			}

			////////////////////////////////////////////////////////////////////////
			/*trans_icp[0]=0;
			trans_icp[1]=0;
			trans_icp[2]=1000;
			rot_icp[0]=0;
			rot_icp[1]=0;
			rot_icp[2]=1.5;*/
			

			//将各个分量整合成一个完整的4*4矩阵
			M[0]=1;				M[4]=rot_icp[0];			M[8]=-1*rot_icp[2];				M[12]=trans_icp[0];
			M[1]=-1*rot_icp[0]; M[5]=1;						M[9]=rot_icp[1];				M[13]=trans_icp[1];
			M[2]=rot_icp[2];	M[6]=-1*rot_icp[1];			M[10]=1;						M[14]=trans_icp[2];
			M[3]=0;				M[7]=0;						M[11]=0;						M[15]=1;


			glPushMatrix();
			glLoadMatrixf(MM);
			glMultMatrixf(M);//MM直接乘以M
			glGetFloatv(GL_MODELVIEW_MATRIX,MM);
			glPopMatrix();
			//glMultMatrixf(M);
			//glLoadMatrixf(M);
			
			//cout<<trans_icp[2]<<"\n";
			////////////////////////////////////////////////////////////////////////////////////////////
			err=clEnqueueWriteBuffer(queue,cl_transvec1,CL_TRUE,0,sizeof(float),&x[3],0,0,0);
			err=clEnqueueWriteBuffer(queue,cl_transvec2,CL_TRUE,0,sizeof(float),&x[4],0,0,0);
			err=clEnqueueWriteBuffer(queue,cl_transvec3,CL_TRUE,0,sizeof(float),&x[5],0,0,0);

			err=clEnqueueWriteBuffer(queue,cl_rotvec1,CL_TRUE,0,sizeof(float),&x[2],0,0,0);
			err=clEnqueueWriteBuffer(queue,cl_rotvec2,CL_TRUE,0,sizeof(float),&x[0],0,0,0);
			err=clEnqueueWriteBuffer(queue,cl_rotvec3,CL_TRUE,0,sizeof(float),&x[1],0,0,0);

			err=clEnqueueWriteBuffer(queue,cl_mm,CL_TRUE,0,sizeof(float)*16,MM,0,0,0);

			err = clEnqueueNDRangeKernel(queue, icp2, 2, 0, global_worksize, 0, 0, 0, 0);
			err = clEnqueueNDRangeKernel(queue, icp3, 2, 0, global_worksize, 0, 0, 0, 0);
			err = clEnqueueNDRangeKernel(queue, icp4, 2, 0, global_worksize, 0, 0, 0, 0);
			
			//////////////////////////////////////////////////////////////////////////////////////////////
		//}
	}//一帧的15次icp计算结束，求得一个MM

	glGetFloatv(GL_MODELVIEW_MATRIX,MMM);
	glLoadIdentity();
	glMultMatrixf(MM);
	glMultMatrixf(MMM);//这里不是用MMM乘以MM，而是先保存MMM，再用MM乘以MMM，即改变相乘顺序
	cout<<"\n";
	////////cout<<"\n"<<result_valid[320+240*640]<<"\n";
	////////cout<<point_cloud[320*3+240*640*3+2]<<"\n";

	glPushMatrix();
	glLoadIdentity();
	GLfloat sun_light_position[] = {1000.0f, -1000.0f, 0.0f, 0.9f};  
	GLfloat sun_light_ambient[]  = {0.0f, 0.0f, 0.0f, 0.1f};  
	GLfloat sun_light_diffuse[]  = {1.0f, 1.0f, 1.0f, 0.1f};  
	GLfloat sun_light_specular[] = {1.0f, 1.0f, 1.0f, 0.9f};  

	glLightfv(GL_LIGHT0, GL_POSITION, sun_light_position);//第0号光源的位置   
	glLightfv(GL_LIGHT0, GL_AMBIENT,  sun_light_ambient); //GL_AMBIENT表示各种光线照射到该材质上,经过很多次反射后最终遗留在环境中的光线强度（颜色）  
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  sun_light_diffuse); //漫反射后
	glLightfv(GL_LIGHT0, GL_SPECULAR, sun_light_specular);//镜面反射后  

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glPopMatrix();

	/*trans_global[0]+=trans_icp[0];
	trans_global[1]+=trans_icp[1];
	trans_global[2]+=trans_icp[2];
	rot_global[0]+=rot_icp[0];
	rot_global[1]+=rot_icp[1];
	rot_global[2]+=rot_icp[2];*/

	//glPopMatrix();
	//glLoadMatrixf(MM);
	
	//glPushMatrix();

	err = clEnqueueNDRangeKernel(queue, add_depth, 2, 0, global_worksize, 0, 0, 0, 0);
	err = clEnqueueNDRangeKernel(queue, update, 2, 0, global_worksize, 0, 0, 0, 0);
	err = clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(unsigned short)*160*120 , result, 0, 0, 0);
	err=clEnqueueReadBuffer(queue, cl_point_num, CL_TRUE, 0, sizeof(unsigned int) ,&point_num, 0, 0, 0);	
	err=clEnqueueReadBuffer(queue,cl_point,CL_TRUE,0,sizeof(float)*160*120*3,point_cloud,0,0,0);
	
	//valid
	
	err=clEnqueueReadBuffer(queue,cl_normalization,CL_TRUE,0,sizeof(float)*160*120*3,normalization,0,0,0);	
	err=clEnqueueReadBuffer(queue,cl_test_point,CL_TRUE,0,sizeof(float)*160*120*6,test_point,0,0,0);	
	err=clEnqueueReadBuffer(queue,cl_energy,CL_TRUE,0,sizeof(float)*160*120,energy,0,0,0);
	err=clEnqueueReadBuffer(queue,cl_depth_icp,CL_TRUE,0,sizeof(float)*160*120,depth_icp,0,0,0);
	err=clEnqueueReadBuffer(queue,cl_doub_filter,CL_TRUE,0,sizeof(float)*160*120,filter,0,0,0);
	err=clEnqueueReadBuffer(queue,cl_doub_normal,CL_TRUE,0,sizeof(float)*160*120*3,doub_normal,0,0,0);
	
	clFinish(queue);

	/////////////////////////////////kernal ends//////////////////////////////////

	//depth_icp[320+640*240]=1000000;
	//result[320+640*240]=1000000;
	/*for (int i=3;i<4;i++)
	{
		for (int j=0;j<7;j++)
		{
			cout<<area[(317+i)*4+(237+j)*640*4+3]<<","<<area[(317+i)*4+(237+j)*640*4+4]<<",,";
		}
		//cout<<"\n";
	}
	cout<<"\n";*/

	//cout<<area[320*4+240*640*4]<<","<<area[320*4+240*640*4+1]<<","<<area[320*4+240*640*4+2]<<","<<area[320*4+240*640*4+3]<<","<<point_cloud[320*3+240*640*3+2]<<"\n\n";
	//cout<<trans_icp[0]<<"\n";
	
	//cvShowImage("win",&img);
	//cvShowImage("win_valid",&img_valid);
	//cvShowImage("win_normalization",&img_normalization);
	//cvShowImage("win_energy",&img_energy);
	//cvShowImage("win_depth_icp",&img_depth_icp);
	//cvShowImage("win_doub_filter",&img_doub_filter);
	//cvShowImage("win_doub_normal",&img_doub_normal);
	key=cvWaitKey(1);

	myDisplay();

	}

}

int main(int argc,char* argv[])
{


	///////////////////////////////////////////////////////////OpenGL starts/////////////////////////////////////////////////////////////

	//初始化opengl
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE);
	glutInitWindowPosition(800,100);
	glutInitWindowSize(640,480);
	glutCreateWindow("Augmented Reality");


	//载入模型
	loader.Init("bmw.3DS",0);
	loader.Init("ford.3DS",1);
	loader.Init("bus.3DS",2);
	loader.Init("youna.3DS",3);
	loader.Init("bmw.3DS",4);

	init_texture();//初始化背景纹理

//	glutDisplayFunc(&myDisplay);//每当屏幕刷新时，调用mydisplay。
//	glutIdleFunc(&myIdle);//每当空闲时，调用括号内的函数。
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();


	///////////////////////////////////////////////////////////part of OpenGL ends/////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////OpenNI starts//////////////////////////////////////////////////////////

	// initial context
	
	eResult = mContext.Init();//初始化openni上下文对象
	CheckOpenNIError( eResult, "initialize context" );

	// set map mode
	
	mapMode.nXRes = 640;
	mapMode.nYRes = 480;
	mapMode.nFPS = 30;

	// create depth generator
	
	eResult = mDepthGenerator.Create( mContext );
	CheckOpenNIError( eResult, "Create depth generator" );
	eResult = mDepthGenerator.SetMapOutputMode( mapMode );
	eResult = mImageGenerator.Create( mContext );
	CheckOpenNIError( eResult, "Create image generator" );
	eResult = mImageGenerator.SetMapOutputMode( mapMode );

	// start generate data
	eResult = mContext.StartGeneratingAll();

///////////////////////////////////////////openNi ends///////////////////////////////////////////////////////
	
	rotvec[0]=0;
	rotvec[1]=0;
	rotvec[2]=0;
	transvec[0]=0;
	transvec[1]=0;
	transvec[2]=0;

	//debugs窗口
	//cvNamedWindow("win",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win",400,100);
	//cvNamedWindow("win_valid",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win_valid",500,100);
	//cvNamedWindow("win_normalization",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win_normalization",600,100);
	//cvNamedWindow("win_energy",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win_energy",700,100);
	//cvNamedWindow("win_depth_icp",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win_depth_icp",800,200);
	//cvNamedWindow("win_doub_filter",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win_doub_filter",100,200);
	//cvNamedWindow("win_doub_normal",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("win_doub_normal",200,200);
	
	//cvInitMatHeader(&mat,480,640,CV_16UC1,result2);
	//cvInitMatHeader(&mat_valid,120,160,CV_16UC1,result_valid);
	//cvInitMatHeader(&mat_normalization,120,160,CV_32FC3,normalization);
	//cvInitMatHeader(&mat_energy,120,160,CV_32FC1,energy);
	//cvInitMatHeader(&mat_depth_icp,120,160,CV_32FC1,depth_icp);
	//cvInitMatHeader(&mat_doub_filter,120,160,CV_32FC1,filter);
	//cvInitMatHeader(&mat_doub_normal,120,160,CV_32FC3,doub_normal);
	//cvInitMatHeader(&mat_img_rgb,480,640,CV_16UC3,m_ImageShow);

	ma=cv::Mat(&mat);
	ma_valid=cv::Mat(&mat_valid);
	ma_normalzation=cv::Mat(&mat_normalization);
	ma_energy=cv::Mat(&mat_energy);
	ma_depth_icp=cv::Mat(&mat_depth_icp);
	ma_doub_filter=cv::Mat(&mat_doub_filter);
	ma_doub_normal=cv::Mat(&mat_doub_normal);
	//ma_img_rgb=cv::Mat(&mat_img_rgb);

	img=IplImage(ma);
	img_valid=IplImage(ma_valid);
	img_normalization=IplImage(ma_normalzation);
	img_energy=IplImage(ma_energy);
	img_depth_icp=IplImage(ma_depth_icp);
	img_doub_filter=IplImage(ma_doub_filter);
	img_rgb=IplImage(m_ImageShow);
	img_doub_normal=IplImage(ma_doub_normal);

	////////////////////////////////////////////////////////////////opencl init starts////////////////////////////////////////////////////////////////

	err = clGetPlatformIDs(0, 0, &num);
	if(err != CL_SUCCESS) {
		cerr << "Unable to get platforms\n";
		return 0;
	}

	std::vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], &num);
	if(err != CL_SUCCESS) {
		cerr << "Unable to get platform ID\n";
		return 0;
	}

	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
	cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if(context == 0) {
		cerr << "Can't create OpenCL context\n";
		return 0;
	}

	size_t cb;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
	std::string devname;
	devname.resize(cb);
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
	std::cout << "Device: " << devname.c_str() << "\n";

	////////////////////////////////////////////////////////////////////opencl init ends////////////////////////////////////////////////////////////////////
	
	queue = clCreateCommandQueue(context, devices[0], 0, 0);
	if(queue == 0) {
		cerr << "Can't create command queue\n";
		clReleaseContext(context);
		return 0;
	}

	mDepthGenerator.GetMetaData(dmdata);

	//在显存中开辟各种空间
	cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned short)*640*480 , (unsigned short*)dmdata.Data(), NULL);
	//	cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cv::Mat) , &ma, NULL);
	cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned short)*160*120 , NULL, NULL);
	cl_point_num=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(unsigned int),NULL,NULL);
	cl_point=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_point_last=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_valid=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(unsigned short)*160*120,NULL,NULL);
	cl_valid_last=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(unsigned short)*160*120,NULL,NULL);
	cl_normalization=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_normalization_last=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_normalization_icp=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_test_point=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*6,NULL,NULL);
	cl_energy=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120,NULL,NULL);
	cl_half_area=clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(unsigned int),NULL,NULL);

	cl_area=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*4,NULL,NULL);
	cl_point_icp=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_valid_icp=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(unsigned short)*160*120,NULL,NULL);
	cl_ATA=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*36,NULL,NULL);
	cl_ATb=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*6,NULL,NULL);
	cl_rotvec1=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_rotvec2=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_rotvec3=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_transvec1=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_transvec2=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_transvec3=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_ATA_sum=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*120*36,NULL,NULL);
	cl_ATb_sum=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*120*6,NULL,NULL);
	cl_ATA_sum_sum=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*36,NULL,NULL);
	cl_ATb_sum_sum=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*6,NULL,NULL);
	cl_depth_icp=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120,NULL,NULL);
	cl_testnum=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*64*6,NULL,NULL);
	cl_matsum1=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*4624*36,NULL,NULL);
	cl_matsum2=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*68*36,NULL,NULL);
	cl_vecsum1=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*4624*6,NULL,NULL);
	cl_vecsum2=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*68*6,NULL,NULL);
	cl_xyzuv=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*5,NULL,NULL);
	cl_doub_filter=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120,NULL,NULL);
	cl_doub_normal=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_buffer1_mat=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*36,NULL,NULL);
	cl_buffer1_vec=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*6,NULL,NULL);
	cl_buffer2_mat=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*36,NULL,NULL);
	cl_buffer2_vec=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*6,NULL,NULL);
	cl_limit=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
	cl_ire_sum_mat=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*2048*36,NULL,NULL);
	cl_ire_sum_vec=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*2048*6,NULL,NULL);
	cl_ire_sum_sum_mat=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*256*36,NULL,NULL);
	cl_ire_sum_sum_vec=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*256*6,NULL,NULL);
	cl_ire_sum_sum_sum_mat=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*36,NULL,NULL);
	cl_ire_sum_sum_sum_vec=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*6,NULL,NULL);
	cl_mm=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*16,NULL,NULL);
	cl_errr=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120,NULL,NULL);
	cl_con=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(int)*160*120,NULL,NULL);
	cl_sub_depth=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*320*240,NULL,NULL);
	cl_sub_point=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*320*240*3,NULL,NULL);
	cl_sub_sub_depth=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120,NULL,NULL);
	cl_sub_sub_normal=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);
	cl_sub_sub_point=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*160*120*3,NULL,NULL);

	if(cl_a == 0  || cl_res == 0) {
		cerr << "Can't create OpenCL buffer\n";
		clReleaseMemObject(cl_a);
		//		clReleaseMemObject(cl_b);
		clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}


	//从.cl文件中读取program程序集
	program = load_program(context, "kernal1.cl");
	if(program == 0) {
		cerr << "Can't load or build program\n";
		clReleaseMemObject(cl_a);
		//		clReleaseMemObject(cl_b);
		clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	//从program程序集中提取各个kernel程序,异常处理
	//addone = clCreateKernel(program,"addone",0);
	/*
	if(addone == 0) {
		cerr << "Can't load kernel addone\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	*/
	range_left = clCreateKernel(program,"range_left",0);
	if(range_left == 0) {
		cerr << "Can't load kernel range_left\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	range_right = clCreateKernel(program,"range_right",0);
	if(range_right == 0) {
		cerr << "Can't load kernel range_right\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	range_up = clCreateKernel(program,"range_up",0);
	if(range_up == 0) {
		cerr << "Can't load kernel range_up\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	range_down = clCreateKernel(program,"range_down",0);
	if(range_down == 0) {
		cerr << "Can't load kernel range_down\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	
	icp1 = clCreateKernel(program,"icp1",0);
	if(icp1 == 0) {
		cerr << "Can't load kernel icp1\n";
		clReleaseProgram(program);
		clReleaseMemObject(cl_a);
		//		clReleaseMemObject(cl_b);
		clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	icp2 = clCreateKernel(program,"icp2",0);
	if(icp2 == 0) {
		cerr << "Can't load kernel icp2\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	icp3 = clCreateKernel(program,"icp3",0);
	if(icp3 == 0) {
		cerr << "Can't load kernel icp3\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	icp4 = clCreateKernel(program,"icp4",0);
	if(icp4 == 0) {
		cerr << "Can't load kernel icp4\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	sum_mat = clCreateKernel(program,"sum_mat",0);
	if(sum_mat == 0) {
		cerr << "Can't load kernel sum_mat\n";
		clReleaseProgram(program);
		//clReleaseMemObject(cl_a);
		//		clReleaseMemObject(cl_b);
		//clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	sum_vec = clCreateKernel(program,"sum_vec",0);
	if(sum_vec == 0) {
		cerr << "Can't load kernel sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	sum_sum_mat = clCreateKernel(program,"sum_sum_mat",0);
	if(sum_sum_mat == 0) {
		cerr << "Can't load kernel sum_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	sum_sum_vec = clCreateKernel(program,"sum_sum_vec",0);
	if(sum_sum_vec == 0) {
		cerr << "Can't load kernel sum_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	update = clCreateKernel(program,"update",0);
	if(update == 0) {
		cerr << "Can't load kernel update\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_sum_mat = clCreateKernel(program,"doub_sum_mat",0);
	if(doub_sum_mat == 0) {
		cerr << "Can't load kernel doub_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_sum_vec = clCreateKernel(program,"doub_sum_vec",0);
	if(doub_sum_vec == 0) {
		cerr << "Can't load kernel doub_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_sum_sum_mat = clCreateKernel(program,"doub_sum_sum_mat",0);
	if(doub_sum_sum_mat == 0) {
		cerr << "Can't load kernel doub_sum_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_sum_sum_vec = clCreateKernel(program,"doub_sum_sum_vec",0);
	if(doub_sum_sum_vec == 0) {
		cerr << "Can't load kernel doub_sum_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_sum_sum_sum_mat = clCreateKernel(program,"doub_sum_sum_sum_mat",0);
	if(doub_sum_sum_sum_mat == 0) {
		cerr << "Can't load kernel doub_sum_sum_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_sum_sum_sum_vec = clCreateKernel(program,"doub_sum_sum_sum_vec",0);
	if(doub_sum_sum_sum_vec == 0) {
		cerr << "Can't load kernel doub_sum_sum_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	doub_filter=clCreateKernel(program,"doub_filter",0);
	if(doub_filter == 0) {
		cerr << "Can't load kernel doub_filter\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	filter_normal=clCreateKernel(program,"filter_normal",0);
	if(filter_normal == 0) {
		cerr << "Can't load kernel filter_normal\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	bit_sum_mat1=clCreateKernel(program,"bit_sum_mat1",0);
	if(bit_sum_mat1 == 0) {
		cerr << "Can't load kernel bit_sum_mat1\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	bit_sum_vec1=clCreateKernel(program,"bit_sum_vec1",0);
	if(bit_sum_vec1 == 0) {
		cerr << "Can't load kernel bit_sum_vec1\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	bit_sum_mat2=clCreateKernel(program,"bit_sum_mat2",0);
	if(bit_sum_mat2 == 0) {
		cerr << "Can't load kernel bit_sum_mat2\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	bit_sum_vec2=clCreateKernel(program,"bit_sum_vec2",0);
	if(bit_sum_vec2 == 0) {
		cerr << "Can't load kernel bit_sum_vec2\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	pre_bit_sum_mat=clCreateKernel(program,"pre_bit_sum_mat",0);
	if(pre_bit_sum_mat == 0) {
		cerr << "Can't load kernel pre_bit_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	pre_bit_sum_vec=clCreateKernel(program,"pre_bit_sum_vec",0);
	if(pre_bit_sum_vec == 0) {
		cerr << "Can't load kernel pre_bit_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	ire_sum_mat=clCreateKernel(program,"ire_sum_mat",0);
	if(ire_sum_mat == 0) {
		cerr << "Can't load kernel ire_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	ire_sum_vec=clCreateKernel(program,"ire_sum_vec",0);
	if(ire_sum_vec == 0) {
		cerr << "Can't load kernel ire_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	ire_sum_sum_mat=clCreateKernel(program,"ire_sum_sum_mat",0);
	if(ire_sum_sum_mat == 0) {
		cerr << "Can't load kernel ire_sum_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	ire_sum_sum_vec=clCreateKernel(program,"ire_sum_sum_vec",0);
	if(ire_sum_sum_vec == 0) {
		cerr << "Can't load kernel ire_sum_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	ire_sum_sum_sum_mat=clCreateKernel(program,"ire_sum_sum_sum_mat",0);
	if(ire_sum_sum_sum_mat == 0) {
		cerr << "Can't load kernel ire_sum_sum_sum_mat\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	ire_sum_sum_sum_vec=clCreateKernel(program,"ire_sum_sum_sum_vec",0);
	if(ire_sum_sum_sum_vec == 0) {
		cerr << "Can't load kernel ire_sum_sum_sum_vec\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	add_depth=clCreateKernel(program,"add_depth",0);
	if(add_depth == 0) {
		cerr << "Can't load kernel add_depth\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	sub_depth_kernel=clCreateKernel(program,"sub_depth",0);
	if(sub_depth_kernel == 0) {
		cerr << "Can't load kernel sub_depth_kernel\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	sub_sub_depth_kernel=clCreateKernel(program,"sub_sub_depth",0);
	if(sub_sub_depth_kernel == 0) {
		cerr << "Can't load kernel sub_sub_depth_kernel\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	mult_kernel=clCreateKernel(program,"mult",0);
	if(mult_kernel == 0) {
		cerr << "Can't load kernel mult_kernel\n";
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}


	//给各个kernel程序设置参数
	/*
	clSetKernelArg(addone, 0, sizeof(cl_mem), &cl_a);
	clSetKernelArg(addone, 1, sizeof(cl_mem), &cl_res);
	clSetKernelArg(addone, 2, sizeof(cl_mem), &cl_point_num);
	clSetKernelArg(addone, 3, sizeof(cl_mem), &cl_point);
	clSetKernelArg(addone, 4, sizeof(cl_mem), &cl_point_last);
	clSetKernelArg(addone, 5, sizeof(cl_mem), &cl_valid);
	clSetKernelArg(addone, 6, sizeof(cl_mem), &cl_valid_last);
	clSetKernelArg(addone, 7, sizeof(cl_mem), &cl_normalization);
	clSetKernelArg(addone, 8, sizeof(cl_mem), &cl_normalization_last);
	clSetKernelArg(addone, 9, sizeof(cl_mem), &cl_test_point);
	clSetKernelArg(addone, 10, sizeof(cl_mem), &cl_energy);
	clSetKernelArg(addone, 11, sizeof(cl_mem), &cl_half_area);
	clSetKernelArg(addone, 12, sizeof(cl_mem), &cl_area);
	*/
	//clSetKernelArg(addone, 13, sizeof(cl_mem), &cl_point_icp);
	//clSetKernelArg(addone, 14, sizeof(cl_mem), &cl_valid_icp);
	//clSetKernelArg(addone, 13, sizeof(cl_mem), &cl_doub_normal);
	//clSetKernelArg(addone, 13, sizeof(cl_mem), &cl_doub_filter);

	clSetKernelArg(range_right, 0, sizeof(cl_mem), &cl_doub_filter);
	clSetKernelArg(range_right, 1, sizeof(cl_mem), &cl_area);
	clSetKernelArg(range_right, 2, sizeof(cl_mem), &cl_half_area);

	clSetKernelArg(range_left, 0, sizeof(cl_mem), &cl_doub_filter);
	clSetKernelArg(range_left, 1, sizeof(cl_mem), &cl_area);
	clSetKernelArg(range_left, 2, sizeof(cl_mem), &cl_half_area);

	clSetKernelArg(range_up, 0, sizeof(cl_mem), &cl_doub_filter);
	clSetKernelArg(range_up, 1, sizeof(cl_mem), &cl_area);
	clSetKernelArg(range_up, 2, sizeof(cl_mem), &cl_half_area);

	clSetKernelArg(range_down, 0, sizeof(cl_mem), &cl_doub_filter);
	clSetKernelArg(range_down, 1, sizeof(cl_mem), &cl_area);
	clSetKernelArg(range_down, 2, sizeof(cl_mem), &cl_half_area);

	clSetKernelArg(filter_normal, 0, sizeof(cl_mem), &cl_normalization);
	clSetKernelArg(filter_normal, 1, sizeof(cl_mem), &cl_doub_normal);
	clSetKernelArg(filter_normal, 2, sizeof(cl_mem), &cl_half_area);
	clSetKernelArg(filter_normal, 3, sizeof(cl_mem), &cl_valid);
	clSetKernelArg(filter_normal, 4, sizeof(cl_mem), &cl_energy);

	clSetKernelArg(icp1, 0, sizeof(cl_mem), &cl_point_icp);
	clSetKernelArg(icp1, 1, sizeof(cl_mem), &cl_point);
	clSetKernelArg(icp1, 2, sizeof(cl_mem), &cl_normalization);
	clSetKernelArg(icp1, 3, sizeof(cl_mem), &cl_valid_icp);
	clSetKernelArg(icp1, 4, sizeof(cl_mem), &cl_valid);
	clSetKernelArg(icp1, 5, sizeof(cl_mem), &cl_ATA);
	clSetKernelArg(icp1, 6, sizeof(cl_mem), &cl_ATb);
	clSetKernelArg(icp1, 7, sizeof(cl_mem), &cl_normalization_icp);
	clSetKernelArg(icp1, 8, sizeof(cl_mem), &cl_errr);
	clSetKernelArg(icp1, 9, sizeof(cl_mem), &cl_con);
	clSetKernelArg(icp1, 10, sizeof(cl_mem), &cl_energy);
	//clSetKernelArg(icp1, 8, sizeof(cl_mem), &cl_depth_icp);

	clSetKernelArg(icp2, 0, sizeof(cl_mem), &cl_point_last);
	clSetKernelArg(icp2, 1, sizeof(cl_mem), &cl_valid_last);
	clSetKernelArg(icp2, 2, sizeof(cl_mem), &cl_transvec1);
	clSetKernelArg(icp2, 3, sizeof(cl_mem), &cl_transvec2);
	clSetKernelArg(icp2, 4, sizeof(cl_mem), &cl_transvec3);
	clSetKernelArg(icp2, 5, sizeof(cl_mem), &cl_rotvec1);
	clSetKernelArg(icp2, 6, sizeof(cl_mem), &cl_rotvec2);
	clSetKernelArg(icp2, 7, sizeof(cl_mem), &cl_rotvec3);
	clSetKernelArg(icp2, 8, sizeof(cl_mem), &cl_valid);
	clSetKernelArg(icp2, 9, sizeof(cl_mem), &cl_xyzuv);
	clSetKernelArg(icp2, 10,sizeof(cl_mem), &cl_mm);

	clSetKernelArg(icp3, 0, sizeof(cl_mem), &cl_point_icp);
	clSetKernelArg(icp3, 1, sizeof(cl_mem), &cl_valid_icp);
	clSetKernelArg(icp3, 2, sizeof(cl_mem), &cl_depth_icp);

	clSetKernelArg(icp4, 0, sizeof(cl_mem), &cl_point_icp);
	clSetKernelArg(icp4, 1, sizeof(cl_mem), &cl_valid_icp);
	clSetKernelArg(icp4, 2, sizeof(cl_mem), &cl_depth_icp);
	clSetKernelArg(icp4, 3, sizeof(cl_mem), &cl_xyzuv);

	clSetKernelArg(sum_mat, 0, sizeof(cl_mem), &cl_ATA);
	clSetKernelArg(sum_mat, 1, sizeof(cl_mem), &cl_ATA_sum);

	clSetKernelArg(sum_vec, 0, sizeof(cl_mem), &cl_ATb);
	clSetKernelArg(sum_vec, 1, sizeof(cl_mem), &cl_ATb_sum);

	clSetKernelArg(sum_sum_mat, 0, sizeof(cl_mem), &cl_ATA_sum);
	clSetKernelArg(sum_sum_mat, 1, sizeof(cl_mem), &cl_ATA_sum_sum);

	clSetKernelArg(sum_sum_vec, 0, sizeof(cl_mem), &cl_ATb_sum);
	clSetKernelArg(sum_sum_vec, 1, sizeof(cl_mem), &cl_ATb_sum_sum);

	clSetKernelArg(update, 0, sizeof(cl_mem), &cl_point);
	clSetKernelArg(update, 1, sizeof(cl_mem), &cl_point_icp);
	clSetKernelArg(update, 2, sizeof(cl_mem), &cl_point_last);
	clSetKernelArg(update, 3, sizeof(cl_mem), &cl_valid);
	clSetKernelArg(update, 4, sizeof(cl_mem), &cl_valid_icp);
	clSetKernelArg(update, 5, sizeof(cl_mem), &cl_valid_last);
	clSetKernelArg(update, 6, sizeof(cl_mem), &cl_normalization);
	clSetKernelArg(update, 7, sizeof(cl_mem), &cl_normalization_icp);
	clSetKernelArg(update, 8, sizeof(cl_mem), &cl_normalization_last);

	clSetKernelArg(doub_sum_mat, 0, sizeof(cl_mem), &cl_ATA);
	clSetKernelArg(doub_sum_mat, 1, sizeof(cl_mem), &cl_matsum1);

	clSetKernelArg(doub_sum_vec, 0, sizeof(cl_mem), &cl_ATb);
	clSetKernelArg(doub_sum_vec, 1, sizeof(cl_mem), &cl_vecsum1);

	clSetKernelArg(doub_sum_sum_mat, 0, sizeof(cl_mem), &cl_matsum1);
	clSetKernelArg(doub_sum_sum_mat, 1, sizeof(cl_mem), &cl_matsum2);

	clSetKernelArg(doub_sum_sum_vec, 0, sizeof(cl_mem), &cl_vecsum1);
	clSetKernelArg(doub_sum_sum_vec, 1, sizeof(cl_mem), &cl_vecsum2);

	clSetKernelArg(doub_sum_sum_sum_mat, 0, sizeof(cl_mem), &cl_ATA_sum_sum);
	clSetKernelArg(doub_sum_sum_sum_mat, 1, sizeof(cl_mem), &cl_matsum2);

	clSetKernelArg(doub_sum_sum_sum_vec, 0, sizeof(cl_mem), &cl_ATb_sum_sum);
	clSetKernelArg(doub_sum_sum_sum_vec, 1, sizeof(cl_mem), &cl_vecsum2);

	clSetKernelArg(doub_filter,0,sizeof(cl_mem),&cl_a);
	clSetKernelArg(doub_filter,1,sizeof(cl_mem),&cl_doub_filter);
	clSetKernelArg(doub_filter,2,sizeof(cl_mem),&cl_half_area);
	//clSetKernelArg(doub_filter,3,sizeof(cl_mem),&cl_valid);
	
	clSetKernelArg(bit_sum_mat1,0,sizeof(cl_mem),&cl_buffer1_mat);
	clSetKernelArg(bit_sum_mat1,1,sizeof(cl_mem),&cl_buffer2_mat);
	clSetKernelArg(bit_sum_mat1,2,sizeof(cl_mem),&cl_limit);

	clSetKernelArg(bit_sum_vec1,0,sizeof(cl_mem),&cl_buffer1_vec);
	clSetKernelArg(bit_sum_vec1,1,sizeof(cl_mem),&cl_buffer2_vec);
	clSetKernelArg(bit_sum_vec1,2,sizeof(cl_mem),&cl_limit);

	clSetKernelArg(bit_sum_mat2,0,sizeof(cl_mem),&cl_buffer2_mat);
	clSetKernelArg(bit_sum_mat2,1,sizeof(cl_mem),&cl_buffer1_mat);
	clSetKernelArg(bit_sum_mat2,2,sizeof(cl_mem),&cl_limit);

	clSetKernelArg(bit_sum_vec2,0,sizeof(cl_mem),&cl_buffer2_vec);
	clSetKernelArg(bit_sum_vec2,1,sizeof(cl_mem),&cl_buffer1_vec);
	clSetKernelArg(bit_sum_vec2,2,sizeof(cl_mem),&cl_limit);

	clSetKernelArg(pre_bit_sum_mat,0,sizeof(cl_mem),&cl_ATA);
	clSetKernelArg(pre_bit_sum_mat,1,sizeof(cl_mem),&cl_buffer2_mat);

	clSetKernelArg(pre_bit_sum_vec,0,sizeof(cl_mem),&cl_ATb);
	clSetKernelArg(pre_bit_sum_vec,1,sizeof(cl_mem),&cl_buffer2_vec);

	clSetKernelArg(ire_sum_mat,0,sizeof(cl_mem),&cl_ATA);
	clSetKernelArg(ire_sum_mat,1,sizeof(cl_mem),&cl_ire_sum_mat);

	clSetKernelArg(ire_sum_vec,0,sizeof(cl_mem),&cl_ATb);
	clSetKernelArg(ire_sum_vec,1,sizeof(cl_mem),&cl_ire_sum_vec);

	clSetKernelArg(ire_sum_sum_mat,0,sizeof(cl_mem),&cl_ire_sum_mat);
	clSetKernelArg(ire_sum_sum_mat,1,sizeof(cl_mem),&cl_ire_sum_sum_mat);

	clSetKernelArg(ire_sum_sum_vec,0,sizeof(cl_mem),&cl_ire_sum_vec);
	clSetKernelArg(ire_sum_sum_vec,1,sizeof(cl_mem),&cl_ire_sum_sum_vec);

	clSetKernelArg(ire_sum_sum_sum_mat,0,sizeof(cl_mem),&cl_ire_sum_sum_mat);
	clSetKernelArg(ire_sum_sum_sum_mat,1,sizeof(cl_mem),&cl_ire_sum_sum_sum_mat);

	clSetKernelArg(ire_sum_sum_sum_vec,0,sizeof(cl_mem),&cl_ire_sum_sum_vec);
	clSetKernelArg(ire_sum_sum_sum_vec,1,sizeof(cl_mem),&cl_ire_sum_sum_sum_vec);

	clSetKernelArg(add_depth,0,sizeof(cl_mem),&cl_depth_icp);
	clSetKernelArg(add_depth,1,sizeof(cl_mem),&cl_valid_icp);
	clSetKernelArg(add_depth,2,sizeof(cl_mem),&cl_sub_sub_depth);
	clSetKernelArg(add_depth,3,sizeof(cl_mem),&cl_valid);
	clSetKernelArg(add_depth,4,sizeof(cl_mem),&cl_point);

	clSetKernelArg(sub_depth_kernel,0,sizeof(cl_mem),&cl_a);
	clSetKernelArg(sub_depth_kernel,1,sizeof(cl_mem),&cl_sub_depth);
	clSetKernelArg(sub_depth_kernel,2,sizeof(cl_mem),&cl_sub_point);

	clSetKernelArg(sub_sub_depth_kernel,0,sizeof(cl_mem),&cl_sub_depth);
	clSetKernelArg(sub_sub_depth_kernel,1,sizeof(cl_mem),&cl_sub_point);
	clSetKernelArg(sub_sub_depth_kernel,2,sizeof(cl_mem),&cl_sub_sub_depth);
	clSetKernelArg(sub_sub_depth_kernel,3,sizeof(cl_mem),&cl_point);
	clSetKernelArg(sub_sub_depth_kernel,4,sizeof(cl_mem),&cl_normalization);
	clSetKernelArg(sub_sub_depth_kernel,5,sizeof(cl_mem),&cl_valid);
	clSetKernelArg(sub_sub_depth_kernel,6,sizeof(cl_mem),&cl_res);
	clSetKernelArg(sub_sub_depth_kernel,7,sizeof(cl_mem),&cl_energy);

	clSetKernelArg(mult_kernel,0,sizeof(cl_mem),&cl_a);

	
	glutDisplayFunc(&myDisplay);//每当屏幕刷新时，调用mydisplay。
	glutIdleFunc(&myIdle);//每当空闲时，调用括号内的函数。
	glMatrixMode(GL_MODELVIEW);
  	glLoadIdentity();

	glutKeyboardFunc(&normal_key);//普通按键响应，委托的方式
	glutSpecialFunc(&keypress);//方向按键响应，委托的方式

	glutMainLoop();
	

	//mainloop ends


  //clReleaseKernel(addone);//
  clReleaseProgram(program);
  clReleaseMemObject(cl_a);
  //clReleaseMemObject(cl_b);
  clReleaseMemObject(cl_res);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
////////////////////////////////////////////////////////////////

  mContext.StopGeneratingAll();
  mContext.Shutdown();

//////////////////////////////////////////////////////

cout<<"done!\n";
cin.get();

return 0;
}


