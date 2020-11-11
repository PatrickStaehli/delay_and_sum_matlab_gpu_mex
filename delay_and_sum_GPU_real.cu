 /*
 * delay_and_sum.cu -  MATLAB external interfaces GPU implementation of a delay and sum algorithm for plane-wave acquisitions.
 * Reconstructs a set of plane-wave ultrasound acquisitions with various different acquistition angles. 
 * 
 *
 *	Input:  	raw_data 				- 3D Matlab GPU array (axial, lateral, acquisition angle)
 * 				acquisition_parameter	- Struct 
 *											- F 				Sampling frequency [MHz]			
 *											- pitch 			Element Pitch [mm] 					
 * 											- c0 				Acquisition speed-of-sound [mm/us] 	
 *				recon_parameter			- Struct
 *											- lense.thickness	Thickness of the acoustic transducer lense
 *											- lense.c 			Speed-of-sound of the acoustic transducer lense
 *											- c 				Anticipated speed-of-sound for delay-and-sum
 *											- apodtan			Tan of the apodization angle
 * 				tx_delays 				- 1D Matlab GPU array of transmit delays for each steering angle
 * 				tx_angles				- 1D Matlab GPU array of steering angles [rad]
 * 
 * 	Output:		rekon 					- 3D Matlab GPU array of the delay-and-summed input signal (axial, lateral, acquisition angle)
 *
 *
 *
 * The calling syntax from Matlab is:
 *
 *		outMatrix = delay_and_sum_GPU(raw_data,acquisition_parameter,recon_parameter, tx_delays, tx_angles)
 *
 * This is a MEX file for MATLAB.
 * 
*/



#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math.h>


/* Create parameter Struct */
struct parameters {
    double sampling_frequency;
	double pitch;
	double speed_of_sound;
	double steering_delay;
	double lense_thickness;
	double lense_speed_of_sound;
	double apodtan;
	double tx_angle;
    int time_steps;
    int number_of_elements;
	int num_threads;
};

/* Input error handling */
/*--------------------------------------------------------------------------------------------------*/
void inputErrorHandling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    /* Verify if the inputs have a valid shape and that all requred inputs are given.
     *
     * Expected input from matlab: (complex Signal, acquisition_parameter, recon_parameters)
     *
     * Expected output to matlab: complex image 
     *
    */
  
    if(nrhs != 5)
      mexErrMsgIdAndTxt( "MATLAB:convec:invalidNumInputs",
              "Five inputs required.");
    if(nlhs > 2)
      mexErrMsgIdAndTxt( "MATLAB:convec:maxlhs",
              "Only one output argument of size T*X allowed");
	
	// Check if the input is GPU Array
	if (!(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt("parallel:gpu:mexGPUExample:InvalidInput", "Invalid input to MEX file.");
    }
    // Check that the acquisition_parameter structure contains all requred fields
    if(mxGetPr(mxGetField(prhs[1],0,"pitch")) == NULL || mxGetPr(mxGetField(prhs[1],0,"F")) == NULL || mxGetPr(mxGetField(prhs[1],0,"c0")) == NULL){
        mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
               " Requred input: (complex_signal, acquisition_parameter, recon_parameter, tx_angle, tx_delay), with  \n" 
               " acquisition_parameter.F \t \t -> sampling frequency [MHz] \n"
               " acquisition_parameter.pitch \t -> element pitch  [mm] \n"
               " acquisition_parameter.c0 \t \t -> Speed-of-sound used for the delay calculation [mm/us]");
    }
    // Check that the recon_parameter structure contains all requred fields
    if(mxGetPr(mxGetField(prhs[2],0,"c")) == NULL || mxGetPr(mxGetField(prhs[2],0,"apodtan")) == NULL || mxGetPr(mxGetField(mxGetField(prhs[2],0,"lense"),0,"thickness")) == NULL ||  mxGetPr(mxGetField(mxGetField(prhs[2],0,"lense"),0,"c")) == NULL){
        mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
               " Requred input: (complex_signal, acquisition_parameter, recon_parameter, tx_angle, tx_delay), with  \n" 
               " recon_parameter.lense.thickness \t -> Thickness of the transducer lense [mm] \n"
               " recon_parameter.lense.c \t \t \t -> Speed-of-sound of the transducer lense [mm/us] \n"
               " recon_parameter.c \t \t \t \t \t -> anticipated speed-of-sound [mm/us] \n"
               " recon_parameter.apodtan \t \t \t -> Tan of the apodization angle \n");
    }
    
}


//* Device code */
/*--------------------------------------------------------------------------------------------------*/
void __global__ delay_and_sum(double const * const raw_data, 
                         double * const rekon,
						 double const * const delay_F, double const * const tx_angles,
						 struct parameters recon_parameters,
                         int const N)
{
	// Calculate the global linear index, assuming a 1-d grid.
    int const iter = blockDim.x * blockIdx.x + threadIdx.x;
	
	// variable declaration
	//thrust::complex<double> phasefac (0,1);
	//double const pi = 3.14159265359;
	//thrust::complex<double> j(0,1);
	
	double tx_angle;		// Current Tx angle
	
	double pitch_F;  		// Element pitch in units of sampling_frequency/SoS
	double tx_delay;		// Delay from element to pixel
	
	double tx_lense_cos; 	// cosine of the lense tx path
	double tx_correction;	// Lense delay correction  
	
	double z_T;				// To be reconstructed depth in units of sampling_frequency/SoS

	double roundtrip_tx = 0;	// Total roundtrip time in TX (steering delay + lense delay + transmit delay)
    
	double delata_pos_rx; 	// Difference of the Rx element position from the pixel positoin
	
	double rx_delay;		// Delay form the pixel to the Rx element
	double roundtrip_rx = 0;	// Total roundtrip time in RX
	
	double rx_element_weight; // Weight of the rx element in dependence of the element rx angle
	
	double rx_angle_sin;	// Sinus of the rx element angle
	double rx_lense_sin;	// Sinus of propagation angle inside lense 
	double rx_lense_cos;	// Cosine of the propagation angle inside the lense
	double rx_correction;	// Lense delay correction in Rx
	
	double total_roundtrip_time; // Total roundtrip time consisting of Tx delay and Rx delay
	
	double rekon_signal = 0;  // Reconstructed signal
	
	// To be reconstructed pixel (z,x,angle)
	int z = iter%recon_parameters.time_steps+ 1;
	int x = (iter / recon_parameters.time_steps) % recon_parameters.number_of_elements	+ 1;
	int angle_num = iter/(recon_parameters.time_steps*recon_parameters.number_of_elements); 
	
	// The tx_angle of the kernel 
	tx_angle = tx_angles[angle_num];

	
	if (iter < N) {	// must be a valid index
		
		pitch_F = (recon_parameters.pitch / recon_parameters.speed_of_sound) * recon_parameters.sampling_frequency; // Element pitch [F/c]
		z_T = (z-1.0) / 2.0;	 // current depth in numbers of speed_of_sound/sampling_frequency
		
		// Tx delay
		tx_delay = (-roundf(recon_parameters.number_of_elements / 2) + (double)x) * pitch_F * sinf(tx_angle) ; 	// Calculate the delay [F/c]
		// Tx delay due to acoustic lense of the transducer
		tx_lense_cos = sqrtf(1 - powf(sinf(abs(tx_angle)),2) * powf(recon_parameters.lense_speed_of_sound,2) / powf(recon_parameters.speed_of_sound,2)); 
		tx_correction = recon_parameters.lense_thickness / recon_parameters.lense_speed_of_sound * tx_lense_cos * recon_parameters.sampling_frequency; // Lense delay correction [F/c]
		
		// Total Tx time
		roundtrip_tx = tx_delay + tx_correction + z_T*cosf(tx_angle);

		
		// Rx Delay
		for (int i_Rx_Element=0; i_Rx_Element< recon_parameters.number_of_elements; i_Rx_Element++ ){  // Rx element offset from insonified pixel
			
			delata_pos_rx = abs((x-1)-i_Rx_Element)*pitch_F; // Difference of the element from the to-be reconsturcted pixel in x
			rx_delay = sqrtf(powf(delata_pos_rx,2) + powf(z_T,2));
			
			rx_element_weight = expf(-powf(delata_pos_rx,2)/powf(z_T,2)/powf(recon_parameters.apodtan,2)); // Weight of the receive element in depencence of the Rx angle
			
			if (recon_parameters.apodtan == 0) { // If the weight is 0, equal weighting for all elementes
				rx_element_weight = 1.0;
			}
			
			
			if (rx_element_weight>1e-3) { // We exclude elements with a small weight
				
				//Rx lense correction	
				rx_angle_sin = delata_pos_rx / rx_delay; // sinus of receive angles 
				rx_lense_sin = rx_angle_sin * recon_parameters.lense_speed_of_sound / recon_parameters.speed_of_sound; // sinus of propagation angle inside lense 
				rx_lense_cos = sqrtf(1-powf(rx_lense_sin,2)); // cosine of propagation angle inside lense 
				rx_correction = recon_parameters.lense_thickness * rx_lense_cos * recon_parameters.sampling_frequency / recon_parameters.lense_speed_of_sound; // delay correction in numbers of lense_speed_of_sound/sampling_frequency
				roundtrip_rx = rx_delay + rx_correction;	
				
				total_roundtrip_time =  roundtrip_rx - delay_F[angle_num] + roundtrip_tx;
				
				
				if (total_roundtrip_time >= 1 && total_roundtrip_time <= recon_parameters.time_steps){ // only valid indices
					
					// Determine the 1d Index of the signal position
					long idx_angle = recon_parameters.time_steps*recon_parameters.number_of_elements*(angle_num);
					long idx_sig = (round(total_roundtrip_time-1)) + (i_Rx_Element)*(recon_parameters.time_steps) + idx_angle;
					
					// Add the signal to the reconstructed signal
					rekon_signal +=  raw_data[idx_sig] * rx_element_weight;
				}
			}
		}
		rekon[iter] = rekon_signal;
		
    }
}




/* Host code */
/*--------------------------------------------------------------------------------------------------*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	
	/* Variable declarations */
	/*--------------------------------------------------------------------------------------------------*/
	
    // GPU Variables
	mxGPUArray const *raw_data, *delay, *tx_angle;
    mxGPUArray *rekon;
    double const *raw_dataP, *delayP, *tx_angleP;
    double *rekonP;
    int N;

	// Choose a reasonably sized number of threads for the block. 
    int const threadsPerBlock = 256*2;
    int blocksPerGrid;
	
	
	/* Initialize the MathWorks GPU API. */
	/*--------------------------------------------------------------------------------------------------*/
	
    mxInitGPU();
    // Input Errro handling
    inputErrorHandling(nlhs, plhs, nrhs, prhs);
  
    
	/* Read the inputs from Matlab */
	/*--------------------------------------------------------------------------------------------------*/
	raw_data = 	mxGPUCreateFromMxArray(prhs[0]);
    raw_dataP = 		(double const *)(mxGPUGetDataReadOnly(raw_data)); // Pointer to raw_data
	
	delay = 	mxGPUCreateFromMxArray(prhs[3]);
	tx_angle = 	mxGPUCreateFromMxArray(prhs[4]);
	delayP =	(double const *)(mxGPUGetDataReadOnly(delay)); // Pointer to delay
	tx_angleP =	(double const *)(mxGPUGetDataReadOnly(tx_angle)); // Pointer to tx_angle
	
	
	// Create a GPUArray to hold the result and get its underlying pointer.
    rekon = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(raw_data),
                            mxGPUGetDimensions(raw_data),
                            mxGPUGetClassID(raw_data),
                            mxGPUGetComplexity(raw_data),
                            MX_GPU_DO_NOT_INITIALIZE);
    rekonP = (double *)(mxGPUGetData(rekon)); // Pointer to rekon
	
	
	// Get the dimension of the input array
	const mwSize *dim_array;
    dim_array = mxGPUGetDimensions(raw_data);
    
	// Read the parameters and store them in a struct
	struct parameters recon_parameters;
 
    recon_parameters.time_steps = dim_array[0];
    recon_parameters.number_of_elements = dim_array[1];
    recon_parameters.sampling_frequency = *mxGetPr(mxGetField(prhs[1],0,"F"));
	recon_parameters.pitch = *mxGetPr(mxGetField(prhs[1],0,"pitch"));
    recon_parameters.speed_of_sound = *mxGetPr(mxGetField(prhs[2],0,"c"));
    recon_parameters.steering_delay = *mxGetPr(prhs[3]);
    recon_parameters.lense_thickness = *mxGetPr(mxGetField(mxGetField(prhs[2],0,"lense"),0,"thickness"));
    recon_parameters.lense_speed_of_sound = *mxGetPr(mxGetField(mxGetField(prhs[2],0,"lense"),0,"c"));
    recon_parameters.apodtan = *mxGetPr(mxGetField(prhs[2],0,"apodtan"));
    recon_parameters.tx_angle = *mxGetPr(prhs[4]);
    
	
	/* Execute device code */
	/*--------------------------------------------------------------------------------------------------*/
	
	N = (int)(mxGPUGetNumberOfElements(raw_data));
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	delay_and_sum<<<blocksPerGrid, threadsPerBlock>>>(raw_dataP, rekonP, delayP, tx_angleP, recon_parameters, N);	
	
	
    /* Create Output to Matlab */
	/*--------------------------------------------------------------------------------------------------*/
	
	//Wrap the result up as a MATLAB gpuArray for return.
    plhs[0] = mxGPUCreateMxArrayOnGPU(rekon);
	
	
	/* Clearing GPU memory cache */
	/*--------------------------------------------------------------------------------------------------*/
	mxGPUDestroyGPUArray(raw_data);
	mxGPUDestroyGPUArray(rekon);
	mxGPUDestroyGPUArray(delay);
    mxGPUDestroyGPUArray(tx_angle);
	
}
