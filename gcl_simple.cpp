#include "Galil.h"
#include "ros/ros.h"
#include "servo_class/parameter.h"
#include <dynamic_reconfigure/server.h>
#include <servo_class/servoConfig.h>
#include <iostream>

using namespace std;

double getSpeed[3];
double gethomingSpeed[3];
double getDegree[3];
double getInterval[3];
double getTime[3];
int a=0, b=0, c=0, d=0;

std::string getAxis;
double getrotationalSpeed[4];
double gethomingSpeedNEW[4];
double getrotationalDegree[4];
double getstopInterval[4];
double getstopTime[4];


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													THERMAL CAMERA SERVO 1												  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void RotateThermalServo1()
{	
		Galil servo_1("A", "192.168.1.41", getrotationalSpeed[0], gethomingSpeedNEW[0], getrotationalDegree[0], getstopInterval[0], getstopTime[0]); 
		cout << servo_1.connection() << endl;
		servo_1.readLimitSwitches();

		if(a < 1)
		{
			if(servo_1.LFB == 1)
			{
				servo_1.jogCW(servo_1.inputSpeed);
				servo_1.homingAxis(servo_1.inputhomingSpeed);
			}
			servo_1.locationIni(servo_1.inputDegree, servo_1.inputSpeed);
			servo_1.gotoDegree(servo_1.inputDegree, servo_1.inputSpeed, servo_1.inputInterval, servo_1.inputStopTime);	
			a++;
		}		
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													THERMAL CAMERA SERVO 2												  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void RotateThermalServo2()
{

		Galil servo_2("B", "192.168.1.41", getrotationalSpeed[1], gethomingSpeedNEW[1], getrotationalDegree[1], getstopInterval[1], getstopTime[1]); 
		cout << servo_2.connection() << endl;
		servo_2.readLimitSwitches();

		if(b < 1)
		{
			if(servo_2.LFB == 1)
			{
				servo_2.jogCW(servo_2.inputSpeed);
				servo_2.homingAxis(servo_2.inputhomingSpeed);
			}
			servo_2.locationIni(servo_2.inputDegree, servo_2.inputSpeed);
			servo_2.gotoDegree(servo_2.inputDegree, servo_2.inputSpeed, servo_2.inputInterval, servo_2.inputStopTime);	
			b++;
		}			
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													LASER SCANNER SERVO 3												  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void RotateLaserServo()
{
		Galil servo_3("C", "192.168.1.41", getrotationalSpeed[2], gethomingSpeedNEW[2], getrotationalDegree[2], getstopInterval[2], getstopTime[2]);
		cout << servo_3.connection() << endl;
		servo_3.readLimitSwitches();

		if(c < 1)
		{
			if(servo_3.LFB == 1)
			{
				servo_3.jogCW(servo_3.inputSpeed);
				servo_3.homingAxis(servo_3.inputhomingSpeed);
			}
			servo_3.locationIni(servo_3.inputDegree, servo_3.inputSpeed);
			servo_3.gotoDegree(servo_3.inputDegree, servo_3.inputSpeed, servo_3.inputInterval, servo_3.inputStopTime);	
			c++;
		}		
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													LINEAR ACTUATOR SERVO 4												  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LinearActuatorServo()
{
		Galil servo_4("D", "192.168.1.41", getrotationalSpeed[3], gethomingSpeedNEW[3], getrotationalDegree[3], getstopInterval[3], getstopTime[3]);
		cout << servo_4.connection() << endl;
		servo_4.readLimitSwitches();

		if(servo_4.inputSpeed != 0)
		{

			if(d < 1)
			{
				if(d < 1 && servo_4.LFD == 1)
				{
					servo_4.moveDown(servo_4.inputSpeed);
					servo_4.homingAxis(servo_4.inputhomingSpeed);
				}
				else
				{
					servo_4.homingAxis(servo_4.inputSpeed);	
				}
				d++;
			}
			else if(d==1 && servo_4.HMD == 1)
			{
				servo_4.moveUp(servo_4.inputSpeed);
				d++;
			}
				
			else if(d==2 && servo_4.LRD == 0)
			{
				servo_4.moveDown(servo_4.inputSpeed);
				d++;
			}
			else if(d==3 && servo_4.LFD == 0)
			{
				servo_4.homingAxis(servo_4.inputSpeed);
			}
		}		
}

void callback(servo_class::servoConfig &config, uint32_t level) 
{
    getrotationalSpeed[0] = config.rotationalSpeedA;
	gethomingSpeedNEW[0] = config.homingSpeedA;
	getrotationalDegree[0] = config.rotationalDegreeA;
	getstopInterval[0] = config.stopIntervalA;
	getstopTime[0] = config.stopTimeA;

	getrotationalSpeed[1] = config.rotationalSpeedB;
	gethomingSpeedNEW[1] = config.homingSpeedB;
	getrotationalDegree[1] = config.rotationalDegreeB;
	getstopInterval[1] = config.stopIntervalB;
	getstopTime[1] = config.stopTimeB;

	getrotationalSpeed[2] = config.rotationalSpeedC;
	gethomingSpeedNEW[2] = config.homingSpeedC;
	getrotationalDegree[2] = config.rotationalDegreeC;
	getstopInterval[2] = config.stopIntervalC;
	getstopTime[2] = config.stopTimeC;

	getrotationalSpeed[3] = config.rotationalSpeedD;
	gethomingSpeedNEW[3] = config.homingSpeedD;
	getrotationalDegree[3] = config.rotationalDegreeD;
	getstopInterval[3] = config.stopIntervalD;
	getstopTime[3] = config.stopTimeD;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "receive");

	dynamic_reconfigure::Server<servo_class::servoConfig> gcl_simple;
  	dynamic_reconfigure::Server<servo_class::servoConfig>::CallbackType f;

	f = boost::bind(&callback, _1, _2);
 	gcl_simple.setCallback(f);

	while(ros::ok())
	{	
		try
		{
			ros::spinOnce();	
			RotateThermalServo1();
			RotateThermalServo2();
			RotateLaserServo();
			LinearActuatorServo();
		}
		catch(string s)
		{
			cout << s << endl;
			return 1;
		}

	}
	ROS_INFO("Spinning node");
  	ros::spin();
	
	return 0;
}
