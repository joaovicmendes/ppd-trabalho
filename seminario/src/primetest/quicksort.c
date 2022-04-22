void quicksort(long int *primes,int first,int last)
{
    int i, j, pivot;
    long int temp;

    if (first < last)
    {
        pivot = first;
        i = first;
        j = last;
        while (i < j)
        {
            while (primes[i] <= primes[pivot] && i<last)
                i++;
            while(primes[j] > primes[pivot])
                j--;

            if (i < j)
            {
                temp = primes[i];
                primes[i] = primes[j];
                primes[j] = temp;
            }
        }

        temp = primes[pivot];
        primes[pivot] = primes[j];
        primes[j] = temp;

        quicksort(primes,first,j-1);
        quicksort(primes,j+1,last);
    }
}
