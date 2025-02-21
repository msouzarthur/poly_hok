
__device__ static double atomic_cas(double* address, double oldv, double newv)
{
    unsigned long long int * address_as_i = (unsigned long long int *) address;
    return  (double)(atomicCAS(address_as_i, ((unsigned long long int) oldv),((unsigned long long int)newv)));
}


