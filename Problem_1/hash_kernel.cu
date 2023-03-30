//CITATIONS:
//chat GPT


// To generate hash value
__device__
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions, unsigned int mod);
/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
__global__
void hash_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
 
    // Generate hash values for every item in the array
    if (index < array_size) {
        unsigned int nonce = nonce_array[index];
        hash_array[index] = generate_hash(nonce, index, transactions, n_transactions, mod);
    }

} // End Hash Kernel //

/* Generate Hash ----------------------
*       Generates a hash value.
*/
__device__
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    unsigned int hash =  (nonce + transactions[0] * (index + 1)) % mod;

    for (int i = 1; i < n_transactions; i++) {
        unsigned int factor = index + 1;
        hash = (hash + transactions[i] * factor) % mod;
    }

    return hash;

} // End Generate Hash //