__kernel void adder(__global const float* a, __global const float* b, __global float* result)
{
int idx = get_global_id(0);
result[idx] = a[idx] + b[idx];
}


//calculate A、b
__kernel void icp1(__global float* point_last,
				   __global float* point,
				   __global float* normalization,
				   __global unsigned short* valid_last,
				   __global unsigned short* valid,
				   __global float* ATA,
				   __global float* ATb,
				   __global float* normalization_last,
				   __global float* err,
				   __global int* con,
				   __global float* energy
				   //__global float* depth_icp
				   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//get normals
	float4 normal=(float4)(0);
	float4 normal_last=(float4)(0);
	
	
	normal.x=normalization[idx*3+idy*gsizex*3];
	normal.y=normalization[idx*3+idy*gsizex*3+1];
	normal.z=normalization[idx*3+idy*gsizex*3+2];
	
	normal_last.x=normalization_last[idx*3+idy*gsizex*3];
	normal_last.y=normalization_last[idx*3+idy*gsizex*3+1];
	normal_last.z=normalization_last[idx*3+idy*gsizex*3+2];
	
	//get coordinates
	float x_last=point_last[idx*3+idy*gsizex*3];
	float y_last=point_last[idx*3+idy*gsizex*3+1];
	float z_last=point_last[idx*3+idy*gsizex*3+2];
	
	float x=point[idx*3+idy*gsizex*3];
	float y=point[idx*3+idy*gsizex*3+1];
	float z=point[idx*3+idy*gsizex*3+2];
	
	//calculate energy 
	float4 distance=(float4)(0);
	float b;
	float c;
	if(valid[idx+idy*gsizex]>0&valid_last[idx+idy*gsizex]>0)
	{
		distance.x=x-x_last;
		distance.y=y-y_last;
		distance.z=z-z_last;
		
		if(fabs(distance.z)<100&fabs(dot(fast_normalize(distance),normal))>0.73)
		{
			b=dot(distance,normal);
			//c=energy[idx+idy*gsizex];
			err[idx+idy*gsizex]=fabs(b);
			con[idx+idy*gsizex]=1;

			
		//AT
		float AT[6];
		AT[0]=0*normal.x+z_last*normal.y-y_last*normal.z;
		AT[1]=-1*z_last*normal.x+0*normal.y+x_last*normal.z;
		AT[2]=y_last*normal.x-x_last*normal.y+0*normal.z;
		AT[3]=1*normal.x;
		AT[4]=1*normal.y;
		AT[5]=1*normal.z;
		
		//k
		//float k=dot(normal,normal_last);
		
		//ATb/k
		ATb[idx*6+idy*gsizex*6]=AT[0]*b;//*c;
		ATb[idx*6+idy*gsizex*6+1]=AT[1]*b;//*c;
		ATb[idx*6+idy*gsizex*6+2]=AT[2]*b;//*c;
		ATb[idx*6+idy*gsizex*6+3]=AT[3]*b;//*c;
		ATb[idx*6+idy*gsizex*6+4]=AT[4]*b;//*c;
		ATb[idx*6+idy*gsizex*6+5]=AT[5]*b;//*c;
		
		//ATA/k
		for(int i=0;i<6;i++)
			for(int j=0;j<6;j++)
			ATA[idx*36+idy*gsizex*36+j+i*6]=AT[i]*AT[j];//*c;
		}
		else
		{
		b=0;
		err[idx+idy*gsizex]=-1;
		con[idx+idy*gsizex]=0;
		ATb[idx*6+idy*gsizex*6]=0;
		ATb[idx*6+idy*gsizex*6+1]=0;
		ATb[idx*6+idy*gsizex*6+2]=0;
		ATb[idx*6+idy*gsizex*6+3]=0;
		ATb[idx*6+idy*gsizex*6+4]=0;
		ATb[idx*6+idy*gsizex*6+5]=0;
		for(int i=0;i<6;i++)
			for(int j=0;j<6;j++)
			ATA[idx*36+idy*gsizex*36+j+i*6]=0;
		}
	}
	else
	{
		err[idx+idy*gsizex]=-1;
		con[idx+idy*gsizex]=0;
		b=0;
		ATb[idx*6+idy*gsizex*6]=0;
		ATb[idx*6+idy*gsizex*6+1]=0;
		ATb[idx*6+idy*gsizex*6+2]=0;
		ATb[idx*6+idy*gsizex*6+3]=0;
		ATb[idx*6+idy*gsizex*6+4]=0;
		ATb[idx*6+idy*gsizex*6+5]=0;
		for(int i=0;i<6;i++)
			for(int j=0;j<6;j++)
			ATA[idx*36+idy*gsizex*36+j+i*6]=0;
	}
	
}
				  
//rotate the points cloud and recompute and update
__kernel void icp2(__global float* point_icp,
				   __global unsigned short* valid_icp,
				   __global float* ta,
				   __global float* tb,
				   __global float* tc,
				   __global float* a,
				   __global float* b,
				   __global float* c,
				   //__global float* normalization,
				   __global unsigned short* valid,
				   //__global float* depth_icp
				   __global float* xyzuv,
				   __global float* trans
				   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//get coordinates
	float x_icp=point_icp[idx*3+idy*gsizex*3];
	float y_icp=point_icp[idx*3+idy*gsizex*3+1];
	float z_icp=point_icp[idx*3+idy*gsizex*3+2];
	
	int u;
	int v;
	
	
	float x;
	float y;
	float z;
	
	float aa;
	float bb;
	float cc;
	float ttaa;
	float ttbb;
	float ttcc;
	
	if(*a>-1&*a<1)
		aa=*a;
	else
		aa=0;
		
	if(*b>-1&*b<1)
		bb=*b;
	else
		bb=0;
		
	if(*c>-1&*c<1)
		cc=*c;
	else
		cc=0;
		
	if(*ta>-100&*ta<100)
		ttaa=*ta;
	else
		ttaa=0;
		
	if(*tb>-100&*tb<100)
		ttbb=*tb;
	else
		ttbb=0;
		
	if(*tc>-100&*tc<100)
		ttcc=*tc;
	else
		ttcc=0;
	
	
	if(valid_icp[idx+idy*gsizex]>0)
	{

		
		x=x_icp*trans[0]+y_icp*trans[4]+z_icp*trans[8]+trans[12];
		y=x_icp*trans[1]+y_icp*trans[5]+z_icp*trans[9]+trans[13];
		z=x_icp*trans[2]+y_icp*trans[6]+z_icp*trans[10]+trans[14];
		
		
		u=(int)(x*525.0f/z+320.0f+2.0f)/4;
		v=(int)(y*525.0f/z+240.0f+2.0f)/4;
		
		
	}
	else
	{
		x=0;
		y=0;
		z=0;
		
		u=-10000;
		v=-10000;
	}
	
	
	xyzuv[idx*5+idy*gsizex*5]=x;
	xyzuv[idx*5+idy*gsizex*5+1]=y;
	xyzuv[idx*5+idy*gsizex*5+2]=z;
	xyzuv[idx*5+idy*gsizex*5+3]=u;
	xyzuv[idx*5+idy*gsizex*5+4]=v;
	
	
	
	///////////////////////////////////////////////////////////////////////
	
}



//init related matrix in icp
__kernel void icp3(__global float* point_icp,
				   __global unsigned short* valid_icp,
				   __global float* depth_icp
				   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	valid_icp[idx+idy*gsizex]=0;
	point_icp[idx*3+idy*gsizex*3]=0;
	point_icp[idx*3+idy*gsizex*3+1]=0;
	point_icp[idx*3+idy*gsizex*3+2]=0;
	depth_icp[idx+idy*gsizex]=0;
}




__kernel void icp4(__global float* point_icp,
				   __global unsigned short* valid_icp,
				   __global float* depth_icp,
				   __global float* xyzuv
				   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);


	float x=xyzuv[idx*5+idy*gsizex*5];
	float y=xyzuv[idx*5+idy*gsizex*5+1];
	float z=xyzuv[idx*5+idy*gsizex*5+2];
			
	int u=(int)xyzuv[idx*5+idy*gsizex*5+3];
	int v=(int)xyzuv[idx*5+idy*gsizex*5+4];
	


	if(u>-1&v>-1&u<160&v<120)
	{
		valid_icp[u+v*gsizex]=1;
		point_icp[u*3+v*gsizex*3]=x;
		point_icp[u*3+v*gsizex*3+1]=y;
		point_icp[u*3+v*gsizex*3+2]=z;
		depth_icp[u+v*gsizex]=z;
	}
	
}


// add depth using some compensation weight stratigy
__kernel void add_depth(__global float* depth_icp,
						__global unsigned short* valid_icp,
						__global float* doub_depth,
						__global unsigned short* valid,
						__global float* point
						)
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	if(valid[idx+idy*gsizex]>0&valid_icp[idx+idy*gsizex]>0)
		doub_depth[idx+idy*gsizex]=doub_depth[idx+idy*gsizex]*0.3+depth_icp[idx+idy*gsizex]*0.7;
	else if(valid[idx+idy*gsizex]==0&valid_icp[idx+idy*gsizex]>0)
	{
		doub_depth[idx+idy*gsizex]=depth_icp[idx+idy*gsizex];
		valid[idx+idy*gsizex]=1;
	}
	
	if(valid[idx+idy*gsizex]>0)
	{
		int u=(int)idx*4;
		int v=(int)idy*4;
		float z=doub_depth[idx+idy*gsizex];
		float x=doub_depth[idx+idy*gsizex]*(u-320)/525;
		float y=doub_depth[idx+idy*gsizex]*(v-240)/525;
	
		point[idx*3+idy*gsizex*3]=x;
		point[idx*3+idy*gsizex*3+1]=y;
		point[idx*3+idy*gsizex*3+2]=z;
	}
	else
	{
		point[idx*3+idy*gsizex*3]=0;
		point[idx*3+idy*gsizex*3+1]=0;
		point[idx*3+idy*gsizex*3+2]=0;
	}
}

					
__kernel void range_right(__global float* mat1,
						  __global float* mat2,
						  __global unsigned int* half_area
						  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	mat2[idx*4+idy*gsizex*4]=mat1[(idx+*half_area)+idy*gsizex];
	//mat2[idx*4+idy*gsizex*4+1]=mat1[(idx-*half_area)+idy*gsizex];
	//mat2[idx*4+idy*gsizex*4+2]=mat1[idx+(idy-*half_area)*gsizex];
	//mat2[idx*4+idy*gsizex*4+3]=mat1[idx+(idy+*half_area)*gsizex];
}

__kernel void range_left(__global float* mat1,
						 __global float* mat2,
						 __global unsigned int* half_area
						 )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//mat2[idx*4+idy*gsizex*4]=mat1[(idx+*half_area)+idy*gsizex];
	mat2[idx*4+idy*gsizex*4+1]=mat1[(idx-*half_area)+idy*gsizex];
	//mat2[idx*4+idy*gsizex*4+2]=mat1[idx+(idy-*half_area)*gsizex];
	//mat2[idx*4+idy*gsizex*4+3]=mat1[idx+(idy+*half_area)*gsizex];
}

__kernel void range_up(__global float* mat1,
					   __global float* mat2,
					   __global unsigned int* half_area
					   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//mat2[idx*4+idy*gsizex*4]=mat1[(idx+*half_area)+idy*gsizex];
	//mat2[idx*4+idy*gsizex*4+1]=mat1[(idx-*half_area)+idy*gsizex];
	mat2[idx*4+idy*gsizex*4+2]=mat1[idx+(idy-*half_area)*gsizex];
	//mat2[idx*4+idy*gsizex*4+3]=mat1[idx+(idy+*half_area)*gsizex];
}

__kernel void range_down(__global float* mat1,
						 __global float* mat2,
						 __global unsigned int* half_area
						 )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//mat2[idx*4+idy*gsizex*4]=mat1[(idx+*half_area)+idy*gsizex];
	//mat2[idx*4+idy*gsizex*4+1]=mat1[(idx-*half_area)+idy*gsizex];
	//mat2[idx*4+idy*gsizex*4+2]=mat1[idx+(idy-*half_area)*gsizex];
	mat2[idx*4+idy*gsizex*4+3]=mat1[idx+(idy+*half_area)*gsizex];
}



__kernel void filter_normal(__global float* normal,
							__global float* doub_normal,
							__global unsigned int* half_area,
							__global unsigned short* valid,
							__global float* energy
							)
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	float4 k=(float4)(0);
	float l=0;//影响因子
	float4 m=(float4)(0);
			m.x=normal[idx*3+idy*3*gsizex];
			m.y=normal[idx*3+idy*3*gsizex+1];
			m.z=normal[idx*3+idy*3*gsizex+2];
	float4 n=(float4)(0);
	float p=0;
	float o=0;
	int cont=0;
	float ene=0;
	for(uint i=0;i<*half_area*2+1;i++)
	{
		for(uint j=0;j<*half_area*2+1;j++)
		{
			if(valid[idx+idy*gsizex]>0)
			{
				n.x=normal[(idx-*half_area+i)*3+(idy-*half_area+j)*3*gsizex];
				n.y=normal[(idx-*half_area+i)*3+(idy-*half_area+j)*3*gsizex+1];
				n.z=normal[(idx-*half_area+i)*3+(idy-*half_area+j)*3*gsizex+2];
				p=dot(m,n);
				o=exp(((i-(float)*half_area)*(i-(float)*half_area)+(j-(float)*half_area)*(j-(float)*half_area))/(-1.0f*(float)*half_area*(float)*half_area));
				l+=o*p;
				k.x+=n.x*o*p;
				k.y+=n.y*o*p;
				k.z+=n.z*o*p;
				ene+=p;
				cont++;
			}
		}
	}
	energy[idx+idy*gsizex]=ene/cont;
	if(l>0.0f)
	{
		doub_normal[idx*3+idy*3*gsizex]=k.x/l;
		doub_normal[idx*3+idy*3*gsizex+1]=k.y/l;
		doub_normal[idx*3+idy*3*gsizex+2]=k.z/l;
		//valid[idx+idy*gsizex]=valid[idx+idy*gsizex]*(exp(l/2)-1)/2000;
	}
	else
	{
		doub_normal[idx*3+idy*3*gsizex]=0;
		doub_normal[idx*3+idy*3*gsizex+1]=0;
		doub_normal[idx*3+idy*3*gsizex+2]=0;
		//valid[idx+idy*gsizex]=100000;
	}
	//valid[idx+idy*gsizex]=valid[idx+idy*gsizex]*l*1000;//100000/(log(121.0f)-log(l))+1;

}

__kernel void sum_mat(__global float* mat,
					  __global float* res
					  )
{
	size_t idx = get_global_id(0);//图像空间行数
	size_t idy = get_global_id(1);//矩阵元素数
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//res[idx*36+idy]=0;
	float sum=0;
	for(int i=0;i<160;i++)
	{
		sum+=mat[idx*160*36+i*36+idy];
	}
	res[idx*36+idy]=sum;
}

__kernel void sum_vec(__global float* vec,
					  __global float* res
					  )
{
	size_t idx = get_global_id(0);//图像空间行数
	size_t idy = get_global_id(1);//向量元素数
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	//res[idx*6+idy]=0;
	float sum=0;
	for(int i=0;i<160;i++)
	{
		sum+=vec[idx*160*6+i*6+idy];
	}
	res[idx*6+idy]=sum;
}


__kernel void sum_sum_mat(__global float* mat,
						  __global float* res
						  )
{
	size_t idx = get_global_id(0);//矩阵元素数
	//size_t idy = get_global_id(1);//矩阵元素数
	size_t gsizex = get_global_size(0);
	//size_t gsizey = get_global_size(1);
	
	//res[idx]=0;
	float sum=0;
	for(int i=0;i<120;i++)
	{
		sum+=mat[i*36+idx];
	}
	res[idx]=sum;
}

__kernel void sum_sum_vec(__global float* vec,
						  __global float* res
						  )
{
	size_t idx = get_global_id(0);//矩阵元素数
	//size_t idy = get_global_id(1);//矩阵元素数
	size_t gsizex = get_global_size(0);
	//size_t gsizey = get_global_size(1);
	
	//res[idx]=0;
	float sum=0;
	for(int i=0;i<120;i++)
	{
		sum+=vec[i*6+idx];
	}
	res[idx]=sum;
}

__kernel void bit_sum_mat1(__global float* buffer_mat1,
						   __global float* buffer_mat2,
						   __global float* limit
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	if(idx<exp2(*limit))
	{
		buffer_mat1[idy+gsizey*idx]=buffer_mat2[idy+gsizey*idx]+buffer_mat2[idy+gsizey*(idx+(int)exp2(*limit))];
	}
}

__kernel void bit_sum_vec1(__global float* buffer_vec1,
						   __global float* buffer_vec2,
						   __global float* limit
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	if(idx<exp2(*limit))
	{
		buffer_vec1[idy+gsizey*idx]=buffer_vec2[idy+gsizey*idx]+buffer_vec2[idy+gsizey*(idx+(int)exp2(*limit))];
	}
}


__kernel void bit_sum_mat2(__global float* buffer_mat2,
						   __global float* buffer_mat1,
						   __global float* limit
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	if(idx<exp2(*limit))
	{
		buffer_mat2[idy+gsizey*idx]=buffer_mat1[idy+gsizey*idx]+buffer_mat1[idy+gsizey*(idx+(int)exp2(*limit))];
	}
}

__kernel void bit_sum_vec2(__global float* buffer_vec2,
						   __global float* buffer_vec1,
						   __global float* limit
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	if(idx<exp2(*limit))
	{
		buffer_vec2[idy+gsizey*idx]=buffer_vec1[idy+gsizey*idx]+buffer_vec1[idy+gsizey*(idx+(int)exp2(*limit))];
	}
}

__kernel void pre_bit_sum_mat(__global float* source_mat,
							  __global float* buffer_mat2
							  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);

	if(idx<45056)
	{
		buffer_mat2[idy+gsizey*idx]=source_mat[idy+gsizey*idx]+source_mat[idy+gsizey*(idx+262144)];
	}
	else
		buffer_mat2[idy+gsizey*idx]=source_mat[idy+gsizey*idx];
}

__kernel void pre_bit_sum_vec(__global float* source_vec,
							  __global float* buffer_vec2
							  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);

	if(idx<45056)
	{
		buffer_vec2[idy+gsizey*idx]=source_vec[idy+gsizey*idx]+source_vec[idy+gsizey*(idx+262144)];
	}
	else
		buffer_vec2[idy+gsizey*idx]=source_vec[idy+gsizey*idx];
}


__kernel void update(__global float* point,
					 __global float* point_icp,
					 __global float* point_last,
					 __global unsigned short* valid,
				     __global unsigned short* valid_icp,
				     __global unsigned short* valid_last,
				     __global float* normalization,
				     __global float* normalization_icp,
				     __global float* normalization_last
				     )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	//////////////////////////////////////////////////////////////////更新上一帧的相关信息//////////////////////////////////////////////////////////////
	if(valid[idx+idy*gsizex]>0)
		valid_last[idx+idy*gsizex]=1;
	else
		valid_last[idx+idy*gsizex]=0;
		
	float x=point[idx*3+idy*gsizex*3];
	float y=point[idx*3+idy*gsizex*3+1];
	float z=point[idx*3+idy*gsizex*3+2];
	
	float n0=normalization[idx*3+idy*gsizex*3];
	float n1=normalization[idx*3+idy*gsizex*3+1];
	float n2=normalization[idx*3+idy*gsizex*3+2];
	
	
	point_last[idx*3+idy*gsizex*3]=x;
	point_last[idx*3+idy*gsizex*3+1]=y;
	point_last[idx*3+idy*gsizex*3+2]=z;
	
	normalization_last[idx*3+idy*gsizex*3]=n0;
	normalization_last[idx*3+idy*gsizex*3+1]=n1;
	normalization_last[idx*3+idy*gsizex*3+2]=n2;
	
	
	
	point_icp[idx*3+idy*gsizex*3]=x;
	point_icp[idx*3+idy*gsizex*3+1]=y;
	point_icp[idx*3+idy*gsizex*3+2]=z;
	
	valid_icp[idx+idy*gsizex]=valid_last[idx+idy*gsizex];
	
	normalization_icp[idx*3+idy*gsizex*3]=n0;
	normalization_icp[idx*3+idy*gsizex*3+1]=n1;
	normalization_icp[idx*3+idy*gsizex*3+2]=n2;
}


__kernel void doub_sum_mat(__global float* mat,
						   //__global float* res,
						   __global float* matsum1
						   //__global float* matsum2
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	////////////////////////////////////////////////////////////
	
	
	float f;
	int g=640*480;
	float sum=0;
	
	
	
	//1
	//f=pown(68.0f,2);
		for(int j=0;j<68;j++)
		{
			if((int)idx*68+j<g)
				sum+=mat[(idx*68+j)*gsizey+idy];
		}
		matsum1[idx*gsizey+idy]=sum;
}

__kernel void doub_sum_vec(__global float* vec,
						   //__global float* res,
						   __global float* vecsum1
						   //__global float* vecsum2
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	////////////////////////////////////////////////////////////
	float f;
	int g=640*480;
	float sum=0;
	///////////////////////////////////////////

	
	//1
	//f=pown(68.0f,2);

		for(int j=0;j<68;j++)
		{
			if((int)idx*68+j<g)
				sum+=vec[(idx*68+j)*gsizey+idy];
		}
		vecsum1[idx*gsizey+idy]=sum;
}


__kernel void doub_sum_sum_mat(//__global float* mat,
						   //__global float* res,
						   __global float* matsum1,
						   __global float* matsum2
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	////////////////////////////////////////////////////////////
	
	
	float f;
	int g=640*480;
	float sum=0;
	

	//2

		for(int j=0;j<68;j++)
		{
			sum+=matsum1[(idx*68+j)*gsizey+idy];
		}
		matsum2[idx*gsizey+idy]=sum;

}

__kernel void doub_sum_sum_vec(//__global float* vec,
						   //__global float* res,
						   __global float* vecsum1,
						   __global float* vecsum2
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	////////////////////////////////////////////////////////////
	float f;
	int g=640*480;
	float sum=0;


	
	//2

		for(int j=0;j<68;j++)
		{
			sum+=vecsum1[(idx*68+j)*gsizey+idy];
		}
		vecsum2[idx*gsizey+idy]=sum;


}




__kernel void doub_sum_sum_sum_mat(//__global float* mat,
						   __global float* res,
						   //__global float* matsum1,
						   __global float* matsum2
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	////////////////////////////////////////////////////////////
	
	
	float f;
	int g=640*480;
	float sum=0;
	

	//3
	//f=pown(68.0f,0);

		for(int j=0;j<68;j++)
		{
			sum+=matsum2[(idx*68+j)*gsizey+idy];
		}
		res[idy]=sum;
}

__kernel void doub_sum_sum_sum_vec(//__global float* vec,
						   __global float* res,
						   //__global float* vecsum1,
						   __global float* vecsum2
						   )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	////////////////////////////////////////////////////////////
	float f;
	int g=640*480;
	float sum=0;
	//3
		for(int j=0;j<68;j++)
		{
			sum+=vecsum2[(idx*68+j)*gsizey+idy];
		}
		res[idy]=sum;
}


__kernel void doub_filter(__global unsigned short* mat1,
						  __global float* mat2,
						  __global unsigned int* half_area
						  //__global unsigned short* valid
						  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
		
	float k=0;
	float l=0;
	float m=0;
	if(mat1[idx+idy*gsizex]>0)
	{
		for(uint i=0;i<*half_area*2+1;i++)
		{
			for(uint j=0;j<*half_area*2+1;j++)
			{
				if(mat1[(idx-*half_area+i)+(idy-*half_area+j)*gsizex]>0)
				{
					m=exp(((i-(float)*half_area)*(i-(float)*half_area)+(j-(float)*half_area)*(j-(float)*half_area))/(-1.0f*(float)*half_area*(float)*half_area))*exp((mat1[(idx-*half_area+i)+(idy-*half_area+j)*gsizex]-mat1[idx+idy*gsizex])*(mat1[(idx-*half_area+i)+(idy-*half_area+j)*gsizex]-mat1[idx+idy*gsizex])/(-30.0f));
					//m=1;
					l+=m;
					k+=mat1[(idx-*half_area+i)+(idy-*half_area+j)*gsizex]*m;
				}
			}
		}
		mat2[idx+idy*gsizex]=k/l;
	}
	else
		mat2[idx+idy*gsizex]=0;
	/*float flag=k/l;
	if(flag>2)
		mat2[idx+idy*gsizex]=flag;
	else
		mat2[idx+idy*gsizex]=0;*/
}


__kernel void ire_sum_mat(__global float* mat,
						  __global float* mat_sum
						  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	__local float buffer[256*9];
	size_t item_id = get_local_id(1);
	//size_t item_idy = get_local_id(1);
	//size_t item_gsizex = get_local_size(0);
	//size_t item_gsizey = get_local_size(1);
	
	if(idy<640*480)
		buffer[item_id]=mat[idx+gsizex*idy];
	else
		buffer[item_id]=0;
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int i=0;i<8;i++)
	{
		if(item_id<exp2(7.0f-i))
			buffer[item_id+(i+1)*256]=buffer[item_id+i*256]+buffer[item_id+(int)exp2(7.0f-i)+i*256];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	//if((int)idy%256==0)
	//	mat_sum[idx+gsizex*((int)idy/256)]=buffer[item_id+8*256];
	if(item_id==0)
		mat_sum[idx+gsizex*((int)idy/256)]=buffer[8*256];
}


__kernel void ire_sum_vec(__global float* vec,
						  __global float* vec_sum
						  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	__local float buffer[256*9];
	size_t item_id = get_local_id(1);
	
	if(idy<640*480)
		buffer[item_id]=vec[idx+gsizex*idy];
	else
		buffer[item_id]=0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int i=0;i<8;i++)
	{
		if(item_id<exp2(7.0f-i))
			buffer[item_id+(i+1)*256]=buffer[item_id+i*256]+buffer[item_id+(int)exp2(7.0f-i)+i*256];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(item_id==0)
		vec_sum[idx+gsizex*((int)idy/256)]=buffer[8*256];
}


__kernel void ire_sum_sum_mat(__global float* mat,
							  __global float* mat_sum
							  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	__local float buffer[256*9];
	size_t item_id = get_local_id(1);
	
	//if(idy<exp2(11.0f))
		buffer[item_id]=mat[idx+gsizex*idy];
	//else
	//	buffer[item_id]=0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int i=0;i<8;i++)
	{
		if(item_id<exp2(7.0f-i))
			buffer[item_id+(i+1)*256]=buffer[item_id+i*256]+buffer[item_id+(int)exp2(7.0f-i)+i*256];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(item_id==0)
		mat_sum[idx+gsizex*((int)idy/256)]=buffer[8*256];
}

__kernel void ire_sum_sum_vec(__global float* vec,
							  __global float* vec_sum
							  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	__local float buffer[256*9];
	size_t item_id = get_local_id(1);
	
	//if(idy<exp2(11.0f))
		buffer[item_id]=vec[idx+gsizex*idy];
	//else
	//	buffer[item_id]=0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int i=0;i<8;i++)
	{
		if(item_id<exp2(7.0f-i))
			buffer[item_id+(i+1)*256]=buffer[item_id+i*256]+buffer[item_id+(int)exp2(7.0f-i)+i*256];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(item_id==0)
		vec_sum[idx+gsizex*((int)idy/256)]=buffer[8*256];
}


__kernel void ire_sum_sum_sum_mat(__global float* mat,
								  __global float* mat_sum
								  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	__local float buffer[256*9];
	size_t item_id = get_local_id(1);
	
	if(idy<exp2(3.0f))
		buffer[item_id]=mat[idx+gsizex*idy];
	else
		buffer[item_id]=0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int i=0;i<8;i++)
	{
		if(item_id<exp2(7.0f-i))
			buffer[item_id+(i+1)*256]=buffer[item_id+i*256]+buffer[item_id+(int)exp2(7.0f-i)+i*256];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(item_id==0)
		mat_sum[idx+gsizex*((int)idy/256)]=buffer[8*256];
}

__kernel void ire_sum_sum_sum_vec(__global float* vec,
								  __global float* vec_sum
								  )
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	__local float buffer[256*9];
	size_t item_id = get_local_id(1);
	
	if(idy<exp2(3.0f))
		buffer[item_id]=vec[idx+gsizex*idy];
	else
		buffer[item_id]=0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int i=0;i<8;i++)
	{
		if(item_id<exp2(7.0f-i))
			buffer[item_id+(i+1)*256]=buffer[item_id+i*256]+buffer[item_id+(int)exp2(7.0f-i)+i*256];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(item_id==0)
		vec_sum[idx+gsizex*((int)idy/256)]=buffer[8*256];
}

__kernel void sub_depth(__global unsigned short* mat,
						__global float* sub_depth,
						__global float* sub_point
						)
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	int n=0;
	float sum=0;
	
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<2;j++)
		{
			if(mat[(idx*2+i)+gsizex*2*(idy*2+j)]>0)
			{
				n++;
				sum+=mat[(idx*2+i)+gsizex*2*(idy*2+j)];
			}
		}
	}
	
	if(n==4)
	{
		sub_depth[idx+gsizex*idy]=sum/4;
		
		int u=(int)idx*2;
		int v=(int)idy*2;
	
		float value=sum/4;
		
		float z=value;
		float x=value*(u-320)/525;
		float y=value*(v-240)/525;
		
		sub_point[idx*3+idy*gsizex*3]=x;
		sub_point[idx*3+idy*gsizex*3+1]=y;
		sub_point[idx*3+idy*gsizex*3+2]=z;
	}
	else
	{
		sub_depth[idx+gsizex*idy]=0;
		sub_point[idx*3+idy*gsizex*3]=0;
		sub_point[idx*3+idy*gsizex*3+1]=0;
		sub_point[idx*3+idy*gsizex*3+2]=0;
	}
	
}

__kernel void sub_sub_depth(__global float* sub_depth,
							__global float* sub_point,
							__global float* sub_sub_depth,
							__global float* sub_sub_point,
							__global float* sub_sub_normal,
							__global unsigned short* valid,
							__global unsigned short* res,
							__global float* energy
							)
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);
	
	int n=0;
	float sum=0;
	
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<2;j++)
		{
			if(sub_depth[(idx*2+i)+gsizex*2*(idy*2+j)]>0)
			{
				n++;
				sum+=sub_depth[(idx*2+i)+gsizex*2*(idy*2+j)];
			}
		}
	}
	
	if(n==4)
	{
		sub_sub_depth[idx+gsizex*idy]=sum/4;
		
		valid[idx+gsizex*idy]=10000;
		res[idx+gsizex*idy]=10000;
		energy[idx+gsizex*idy]=10000;
		
		int u=(int)idx*4;
		int v=(int)idy*4;
	
		float value=sum/4;
		
		float z=value;
		float x=value*(u-320)/525;
		float y=value*(v-240)/525;
		
		sub_sub_point[idx*3+idy*gsizex*3]=x;
		sub_sub_point[idx*3+idy*gsizex*3+1]=y;
		sub_sub_point[idx*3+idy*gsizex*3+2]=z;
		
		float x_right=sub_point[(idx*2+1)*3+idy*2*gsizex*2*3];
		float x_left=sub_point[idx*2*3+idy*2*gsizex*2*3];
		float y_right=sub_point[(idx*2+1)*3+idy*2*gsizex*2*3+1];
		float y_left=sub_point[idx*2*3+idy*2*gsizex*2*3+1];
		float z_right=sub_point[(idx*2+1)*3+idy*2*gsizex*2*3+2];
		float z_left=sub_point[idx*2*3+idy*2*gsizex*2*3+2];
		
		float x_up=sub_point[idx*2*3+idy*2*gsizex*2*3];
		float x_down=sub_point[idx*2*3+(idy*2+1)*gsizex*2*3];
		float y_up=sub_point[idx*2*3+idy*2*gsizex*2*3+1];
		float y_down=sub_point[idx*2*3+(idy*2+1)*gsizex*2*3+1];
		float z_up=sub_point[idx*2*3+idy*2*gsizex*2*3+2];
		float z_down=sub_point[idx*2*3+(idy*2+1)*gsizex*2*3+2];
		
		float4 norm=cross((float4)(x_right-x_left,y_right-y_left,z_right-z_left,0),(float4)(x_up-x_down,y_up-y_down,z_up-z_down,0));
		float4 normal=fast_normalize(norm);
		
		sub_sub_normal[idx*3+gsizex*idy*3]=normal.x*-1;
		sub_sub_normal[idx*3+gsizex*idy*3+1]=normal.y*-1;
		sub_sub_normal[idx*3+gsizex*idy*3+2]=normal.z*-1;
	}
	else
	{
		sub_sub_depth[idx+gsizex*idy]=0;
		sub_sub_point[idx*3+idy*gsizex*3]=0;
		sub_sub_point[idx*3+idy*gsizex*3+1]=0;
		sub_sub_point[idx*3+idy*gsizex*3+2]=0;
		sub_sub_normal[idx*3+gsizex*idy*3]=0;
		sub_sub_normal[idx*3+gsizex*idy*3+1]=0;
		sub_sub_normal[idx*3+gsizex*idy*3+2]=0;
		valid[idx+gsizex*idy]=0;
		res[idx+gsizex*idy]=0;
		energy[idx+gsizex*idy]=0;
	}
}



__kernel void mult(__global unsigned short* mat)
{
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t gsizex = get_global_size(0);
	size_t gsizey = get_global_size(1);

	mat[idx+gsizex*idy]=mat[idx+gsizex*idy]*14;
}