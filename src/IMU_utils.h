#ifndef __IMU_UTILS
#define __IMU_UTILS
#include<stdio.h>

#define IMU_SAMPLE_TIMES 200

// cal mpu6050's data offset
float get_gyro_z_offset(int imu_fd){
    signed int imu_buffer[7];
    int ret = 0;
    float gyro_z_sum = 0.0;
    for (int i =0; i < IMU_SAMPLE_TIMES;i++){
    	ret = read(imu_fd, imu_buffer, sizeof(imu_buffer));
	if(ret == 0){
		float gyro_x = (float)(imu_buffer[0]) / 16.4;
		float gyro_y = (float)(imu_buffer[1]) / 16.4;
		float gyro_z = (float)(imu_buffer[2]) / 16.4;
		gyro_z_sum += gyro_z; 
	}
    }
    return gyro_z_sum / IMU_SAMPLE_TIMES;
}

// filter to denoise
void filter_imu_data(float& gyro_x_o, float& gyro_y_o, float& gyro_z_o,
		     float& acc_x_o,  float& acc_y_o,  float& acc_z_o){
	//TODO	
}


#endif
