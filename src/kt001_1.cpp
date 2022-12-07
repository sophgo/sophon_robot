#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>
#include <dynamic_reconfigure/server.h>
#include "sophon_robot/pidConfig.h"
#include"IMU_utils.h"

#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#define head1 0xAA
#define head2 0x55
#define sendType_velocity    0x11
#define sendType_pid         0x12
#define sendType_params      0x13

#define rBUFFERSIZE          27

using namespace std;
using namespace boost::asio;
typedef	union{
		float fvalue;
		uint8_t cvalue[4];
}float_union;
io_service iosev;
serial_port sp(iosev);         //Define the serial port for transmission

std::string port_name;
int baud_rate;
bool publish_odom_transform;
int kp;
int ki;
int kd;
double linear_correction;
double angular_correction;
      
ros::Time cmd_time;

double x = 0.0;
double y = 0.0;
double yaw = 0.0;
double imu_yaw = 0.0;

char* imu_file = "/dev/mpu6050";
int imu_fd;
float imu_gz_offset = 0.0;

uint8_t checksum(uint8_t* buf, size_t len)
{
  uint8_t sum = 0x00;
  for(int i=0;i<len;i++)
  {
    sum += *(buf + i);
  }
  
  return sum;
}

//接收数据分析与校验
unsigned char data_analysis(uint8_t* buffer)
{
	unsigned char ret=0,csum;
	// int i;
	// for (int i =0; i < 27; i++){
	//     printf("%x ", buffer[i]);
	// }
	// printf("\n");
	if((buffer[0]==head1) && (buffer[1]==head2)){
		csum = buffer[2]^buffer[3]^buffer[4]^buffer[5]^buffer[6]^buffer[7]^
				buffer[8]^buffer[9]^buffer[10]^buffer[11]^buffer[12]^buffer[13]^
				buffer[14]^buffer[15]^buffer[16]^buffer[17]^buffer[18]^buffer[19]^
				buffer[20]^buffer[21]^buffer[22]^buffer[23]^buffer[24]^buffer[25];
		//ROS_INFO("check sum:0x%02x",csum);
		if(csum == buffer[26]){
			ret = 1;//校验通过，数据包正确
		}
		else 
		  ret =0;//校验失败，丢弃数据包
	}
	/*
	for(i=0;i<rBUFFERSIZE;i++)
	  ROS_INFO("0x%02x",buffer[i]);
	*/
	return ret;
}


/*PID parameter sending function*/
void SetPID(int p,int i, int d)
{
	static uint8_t tmp[11];
	tmp[0] = head1;
	tmp[1] = head2;
	tmp[2] = 0x0b;
	tmp[3] = sendType_pid;
	tmp[4] = (p>>8)&0xff;
	tmp[5] = p&0xff;
	tmp[6] = (i>>8)&0xff;
	tmp[7] = i&0xff;
	tmp[8] = (d>>8)&0xff;
	tmp[9] = d&0xff;
	tmp[10] = checksum(tmp,10);
	write(sp,buffer(tmp,11));
}

/*robot parameter sending function*/
void SetParams(double linear_correction,double angular_correction) {
	static uint8_t tmp[9];
	tmp[0]  = head1;
	tmp[1]  = head2;
	tmp[2]  = 0x09;
	tmp[3]  = sendType_params;
	tmp[4]  = (int16_t)((int16_t)(linear_correction*1000)>>8)&0xff;
	tmp[5]  = (int16_t)(linear_correction*1000)&0xff;
	tmp[6]  = (int16_t)((int16_t)(angular_correction*1000)>>8)&0xff;
	tmp[7]  = (int16_t)(angular_correction*1000)&0xff;
	tmp[8]  = checksum(tmp,8);
	write(sp,buffer(tmp,9));
}

/*robot speed transmission function*/
void SetVelocity(double x, double y, double yaw)
{
  static uint8_t tmp[11];
  tmp[0] = head1;
  tmp[1] = head2;
  tmp[2] = 0x0b;
  tmp[3] = sendType_velocity;
  tmp[4] = ((int16_t)(x*1000)>>8) & 0xff;
  tmp[5] = ((int16_t)(x*1000)) & 0xff;
  tmp[6] = ((int16_t)(y*1000)>>8) & 0xff;
  tmp[7] = ((int16_t)(y*1000)) & 0xff;
  tmp[8] = ((int16_t)(yaw*1000)>>8) & 0xff;
  tmp[9] = ((int16_t)(yaw*1000)) & 0xff;
  tmp[10] = checksum(tmp,10);
  write(sp,buffer(tmp,11));
}

/* cmd_vel Subscriber callback function*/
void cmd_callback(const geometry_msgs::Twist& msg)
{
  x = msg.linear.x;
  y = msg.linear.x;
  yaw = msg.angular.z;
  cmd_time = ros::Time::now();
}

/*pid dynamic_reconfigure callback function*/
void pidConfig_callback(sophon_robot::pidConfig &config)
{
	kp = config.kp;
	ki = config.ki;
	kd = config.kd;
	SetPID(kp,ki,kd);
}

//serial port receiving task
void serial_task()
{
  uint8_t r_buffer[rBUFFERSIZE];
  ros::Time now_time = ros::Time::now();
  ros::Time last_time = now_time;
  ros::NodeHandle n;
  
  //Create Publisher message
  sensor_msgs::Imu imu_msgs;
  geometry_msgs::TransformStamped odom_trans;
  geometry_msgs::Quaternion odom_quat;
  nav_msgs::Odometry odom;

  //Create Publisher
  tf::TransformBroadcaster odom_broadcaster;
  ros::Publisher imu_pub     = n.advertise<sensor_msgs::Imu>("imu",10);
  ros::Publisher odom_pub    = n.advertise<nav_msgs::Odometry>("odom", 10);
  enum frameState{
	State_Head1, State_Head2, State_Data, State_Handle
  };
  ROS_INFO("start receive message");
  frameState state = State_Head1;
  while(true)
  {
    float_union posx,posy,vx,vy,va,yaw;
    signed int imu_buffer[7];
    float gyro_x=0.0, gyro_y=0.0, gyro_z=0.0;
    float accel_x=0.0, accel_y=0.0, accel_z=0.0;
    int ret =0;
   // printf("in loop \n");
    switch (state){
	case State_Head1:
    		read(sp, buffer(&r_buffer[0], 1));
		// printf("buffer 0: %x", r_buffer[0]);
		state = (r_buffer[0] == head1 ? State_Head2 : State_Head1);
		break;
	case State_Head2:
		read(sp, buffer(&r_buffer[1], 1));
		state = (r_buffer[1] == head2 ? State_Data : State_Head1);
		break;
	case State_Data:
		read(sp, buffer(&r_buffer[2], rBUFFERSIZE-2));
		state = data_analysis(r_buffer) == 0 ? State_Head1: State_Handle;
		break;
	case State_Handle:
		// printf("handle msg\n");
		// read IMU msg
                read(imu_fd,imu_buffer, sizeof(imu_buffer));
	        now_time = ros::Time::now();
		if(ret == 0){
			gyro_x = (float)(imu_buffer[0]) / 16.4 / 180 * 3.1415; // rad/second
			gyro_y = (float)(imu_buffer[1]) / 16.4 / 180 * 3.1415;
			gyro_z = ((float)(imu_buffer[2])/ 16.4 - imu_gz_offset) / 180 * 3.1415;
			accel_x= (float)(imu_buffer[3]) / 2048 * 9.8; // m/s^2
			accel_y= (float)(imu_buffer[4]) / 2048 * 9.8; 
			accel_z= (float)(imu_buffer[5]) / 2048 * 9.8;
		}
		imu_yaw += gyro_z * (now_time - last_time).toSec();
		last_time = now_time;
		// printf("now_time: %lf, imu_yaw: %lf \n", now_time.toSec(), imu_yaw);
		imu_msgs.header.stamp = now_time;
		imu_msgs.header.frame_id = "base_imu_link";
		imu_msgs.angular_velocity.x = gyro_x;
		imu_msgs.angular_velocity.y = gyro_y;
		imu_msgs.angular_velocity.z = gyro_z;
		imu_msgs.linear_acceleration.x = accel_x;
		imu_msgs.linear_acceleration.y = accel_y;
		imu_msgs.linear_acceleration.z = accel_y;
		imu_msgs.orientation =tf::createQuaternionMsgFromRollPitchYaw(0, 0, imu_yaw);
		imu_msgs.orientation_covariance = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		imu_msgs.angular_velocity_covariance = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		imu_msgs.linear_acceleration_covariance = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		imu_pub.publish(imu_msgs);

		for(int i=0;i<4;i++){
			posx.cvalue[i] = r_buffer[2+i];//x 坐标
			posy.cvalue[i] = r_buffer[6+i];//y 坐标
			vx.cvalue[i] = r_buffer[10+i];// x方向速度
			vy.cvalue[i] = r_buffer[14+i];//y方向速度
			va.cvalue[i] = r_buffer[18+i];//角速度
			yaw.cvalue[i] = r_buffer[22+i];	//yaw 偏航角
		}
	
		//发布坐标变换父子坐标系
	        odom_trans.header.stamp = now_time;
	        odom_trans.header.frame_id = "odom";
	        odom_trans.child_frame_id = "base_link";
	        odom_trans.transform.translation.x = posx.fvalue;
	        odom_trans.transform.translation.y = posy.fvalue;
	        odom_trans.transform.translation.z = 0.0;
	        odom_quat = tf::createQuaternionMsgFromYaw(yaw.fvalue);//将偏航角转换成四元数才能发布
	        odom_trans.transform.rotation = odom_quat;
	
	        //获取当前时间
		//载入里程计时间戳
		odom.header.stamp = now_time;
		//里程计父子坐标系
		odom.header.frame_id = "odom";
		odom.child_frame_id = "base_link";
		//里程计位置数据
		odom.pose.pose.position.x = posx.fvalue;
		odom.pose.pose.position.y = posy.fvalue;
		odom.pose.pose.position.z = 0;
		odom.pose.pose.orientation = odom_quat;
		//载入线速度和角速度
		odom.twist.twist.linear.x = vx.fvalue;
		odom.twist.twist.linear.y = vy.fvalue;
		odom.twist.twist.angular.z = va.fvalue;
	        odom.twist.covariance = { 1e-9, 0, 0, 0, 0, 0, 
					  0, 1e-3, 1e-9, 0, 0, 0, 
					  0, 0, 1e6, 0, 0, 0,
				          0, 0, 0, 1e6, 0, 0, 
					  0, 0, 0, 0, 1e6, 0, 
					  0, 0, 0, 0, 0, 0.1 };
		odom.pose.covariance = { 1e-9, 0, 0, 0, 0, 0, 
				         0, 1e-3, 1e-9, 0, 0, 0, 
					 0, 0, 1e6, 0, 0, 0,
					 0, 0, 0, 1e6, 0, 0, 
					 0, 0, 0, 0, 1e6, 0, 
					 0, 0, 0, 0, 0, 1e3 };
	
	        if(publish_odom_transform)odom_broadcaster.sendTransform(odom_trans);
	        //publish the odom message
	        odom_pub.publish(odom);
		state = State_Head1;
		break;
	default:
		state = State_Head1;
		break;
    	}

  } 
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "robot");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");
  
  /*Get robot parameters from configuration file*/
  pn.param<std::string>("port_name",port_name,std::string("/dev/ttyUSB0"));
  pn.param<int>("baud_rate",baud_rate,115200);

  pn.param<double>("linear_correction",linear_correction,1.0);
  pn.param<double>("angular_correction",angular_correction,1.0);
  pn.param<bool>("publish_odom_transform",publish_odom_transform,true);
  
  //set serial port
  boost::system::error_code ec;
  sp.open(port_name,ec);
  sp.set_option(serial_port::baud_rate(baud_rate));   
  sp.set_option(serial_port::flow_control(serial_port::flow_control::none));
  sp.set_option(serial_port::parity(serial_port::parity::none));
  sp.set_option(serial_port::stop_bits(serial_port::stop_bits::one));
  sp.set_option(serial_port::character_size(8));
  
  //open imu data
  imu_fd = open(imu_file, O_RDWR);
  if(imu_fd < 0){
  	printf("warning: can not open IMU: %s!!\n", imu_file);
  	return -1;
  }
 
  // cal IMU_offset
  imu_gz_offset = get_gyro_z_offset(imu_fd);
  
  //Create Subscriber
  ros::Subscriber cmd_sub    = n.subscribe("/cmd_vel",10,cmd_callback);
  
  //set pid dynamic_reconfigure
  dynamic_reconfigure::Server<sophon_robot::pidConfig> server;
  dynamic_reconfigure::Server<sophon_robot::pidConfig>::CallbackType f;
  f = boost::bind(&pidConfig_callback, _1);
  server.setCallback(f);

  ros::Duration(0.02).sleep();
  SetParams(linear_correction,angular_correction);
  
  //Create serial port receiving task
  thread serial_thread(boost::bind(serial_task));
  
  ros::Time current_time, last_time;
  current_time = ros::Time::now();
  last_time = ros::Time::now();
  
  while(ros::ok()){
    //send once velocity to robot base every 0.02
    current_time = ros::Time::now();
    if((current_time - last_time).toSec() > 0.02){           
      last_time = current_time;
      
      if((current_time - cmd_time).toSec() > 1)
      {
        x = 0.0;
        y = 0.0;
        yaw = 0.0;
      }
      SetVelocity(x,y,yaw);
      SetPID(kp,ki,kd);
    }
    
    ros::spinOnce();
  }
     
  return 0;
}

