# BrainWear_gait_analysis

Identify the walking interval and then extract and compare the gait features. The basic workflow is shown below. 

## Raw data processing 

Example data could be found in the [AX3 website](https://axivity.com/downloads/ax3).

The raw data collected from the AX3 accelerometer need to be pre-processed using the biobank RF-HMM code. Please follow the instruction [Biobank RF-HMM code](https://github.com/Tim-Yu/BrainWear_gait_analysis/blob/master/Biobank_data_processing.txt). 

After the initial processing, the raw gz file was converted to the csv file, meanwhile, the activity classification results were stored into the -timeSeries.csv.gz file

## The splicer 

The inputs for the splicer are the converted csv files, which contains the raw tri-axial acceleration data, and the -timeSeries.csv.gz files, which has the activity diary.

The splicer will 
- firstly, pick out the walking periods from the whole timeline based on the classification results, 
- secondly, calculate the Euclidean norm of the acceleration value of the three arises for each time point in the walking period (the AX3 default recording sampling rate is 100Hz which means 100 time point in 1s), 
- finally, store each individual walking gait data into separate csv files and plot line charts of the calculated norm. 

The outputs from the splicer are the gait data which are the calculated Euclidean norm series of each walking periods. Each file represents the wave pattern of the acceleration change in the corresponding walking time period.

## The pipeline 

Before feeding data into the pipeline, the gait data under comparison need to be stored into two individual folders with the same parent directory. We are normally comparing the gaits before and after the surgery.

The inputs for the pipeline are the directory containing the gait data folder for comparison.

The pipeline will 
- firstly, filter out the ‘noisy’ data based on the wave shape of the gait data,
- secondly, extract the gait pattern features by calculating the continuous wavelet transform coefficients of the selected ‘pure’ walking gait wave,
- thirdly, train the SVM model to learn one gait pattern,
- finally, apply the trained model to recognise the other gait pattern.

The label of the sample for the SVM training is the before or after the surgery. The gait changing before and after treatments was determined by if the SVM which was trained on the before-treatments dataset could recognise the after-treatments dataset. The performance of the model represents the gait changing degree.

The outputs are the SVM recognition accuracy i.e. the gait difference.

## The Visualization 

Some tools to visualize the data. (for presentation)
