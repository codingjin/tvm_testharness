#include <stdio.h>
#include <stdlib.h>
#include "sys/time.h"
#include "time.h"

struct timeval start, finish;

void get_time(int flag){
	if (flag == 1){
		gettimeofday(&start, NULL);
	} else{
		gettimeofday(&finish, NULL);
		unsigned int timeval = ((unsigned int)(finish.tv_sec - start.tv_sec))*1000000 + (unsigned int)(finish.tv_usec - start.tv_usec);
		printf("Elapsed time(us) is : %u\n", timeval);
	}
}
