#include <sys/time.h>
#include <stdio.h>
const double kMicro = 1.0e-6;
double wall_time()
{
    struct timeval TV;

    const int RC = gettimeofday(&TV, NULL);
    if(RC == -1)
    {
        printf("ERROR: Bad call to gettimeofday\n");
        return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()
