# BrainWear_gait_analysis

Identify the walking interval and then extract and compare the gait features. The basic workflow is shown below. 

## Raw data processing 

Example data could be found in the [AX3 website](https://axivity.com/downloads/ax3).

The raw data collected from the AX3 accelerometer need to be pre-processed using the biobank RF-HMM code. Please follow the instruction [Running biobank RF-HMM](https://gitlab.com/computational.oncology/brainwear/blob/Daisy/Running%20biobank%20RF-HMM). 

After the initial processing, the raw gz file was converted to the csv file, meanwhile, the activity classification results were stored into the -timeSeries.csv.gz file

## The splicer 

The inputs for the splicer are the converted csv files and the -timeSeries.csv.gz files.

The splicer will 
- firstly, pick out the walking periods from the whole timeline based on the classification results, 
- secondly, calculate the Euclidean norm of the acceleration value of the three arises for each time point in the walking period (the AX3 default recording sampling rate is 100Hz which means 100 time point in 1s), 
- finally, store each individual walking gait data into separate csv files and plot line charts of the calculated norm. 

The outputs from the splicer are the gait data which are the calculated Euclidean norm series of each walking periods.

## The pipeline 

Before feeding data into the pipeline, the gait data under comparison need to be stored into two individual folders with the same parent directory. 

The inputs for the pipeline are the directory containing the gait data folder for comparison.

The pipeline will 
- firstly, filter out the ‘noisy’ data based on the wave shape of the gait data,
- secondly, train the SVM model to learn one gait pattern,
- thirdly, apply the trained model to recognise the other gait pattern.

The gait changing before and after treatments was determined by if the SVM which was trained on the before-treatments dataset could recognise the after-treatments dataset. The performance of the model represents the gait changing degree.

The outputs are the SVM performance i.e. the gait difference.

## The Visualization 

Some tools to visualize the data. (for presentation)
