
__device__ static double atomic_cas(double* address, double oldv, double newv)
{
    long long int * address_as_i = (long long int *) address;
    return  (double)(atomicCAS(address_as_i, ((long long int) oldv),((long long int)newv)));
}


