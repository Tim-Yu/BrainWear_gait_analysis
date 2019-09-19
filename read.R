
rm(list = ls())
graphics.off()


library(GGIR)
g.shell.GGIR(#=======================================
             # INPUT NEEDED:
             mode=c(1,2,3,4,5), # which part is applied 
             datadir="C:/Users/simpl/Data_folder/BrainWear_KLC/100",
             outputdir="C:/Users/simpl/Data_folder/BrainWear_KLC/out",
             f0=1, f1=2,# no worry, just for detecting if the dir is a directory or a list of file
             #-------------------------------
             # Part 1:
             #-------------------------------
             # Key functions: reading file, auto-calibration, and extracting features
             windowsize = c(5,900,3600), # Epoch length, non-wear detection res, non wear detection evaluation window
             do.cal = TRUE, # allpy autocalibration
             
             do.enmo = TRUE,             do.anglez=TRUE,
             chunksize=1,    # low donw if causing memory problem             
             printsummary=TRUE,
             
             
             #-------------------------------
             # Part 2:
             #-------------------------------
             strategy = 2, #strategy = 1: Exclude ¡®hrs.del.start¡¯ number of hours at the beginning and ¡®hrs.del.end¡¯ 
                           #number of hours at the end of the measurement and never allow for more than ¡®maxdur¡¯ number 
                           #of hours. These three parameters are set by their respective function arguments.
                           #strategy = 2 makes that only the data between the first midnight and the last midnight is used 
                           #for imputation. - strategy = 3 only selects the most active X days in the files. X is specified 
                           #by argument ¡®ndayswindow¡¯              
             ndayswindow=7,
             hrs.del.start = 0,          hrs.del.end = 0,
             maxdur = 9,                 includedaycrit = 16,
             winhr = c(5,10),
             qlevels = c(c(1380/1440),c(1410/1440)),
             qwindow=c(0,24),
             ilevels = c(seq(0,400,by=50),8000),
             mvpathreshold =c(100,120),
             bout.metric = 4,
             closedbout=FALSE,
             #-------------------------------
             # Part 3:
             #-------------------------------
             # Key functions: Sleep detection
             timethreshold= c(5),        anglethreshold=5,
             ignorenonwear = TRUE,
             #-------------------------------
             # Part 4:
             #-------------------------------
             # Key functions: Integrating sleep log (if available) with sleep detection
             # storing day and person specific summaries of sleep
             excludefirstlast = TRUE,
             includenightcrit = 16,
             def.noc.sleep = c(),
             #loglocation= "C:/Users/simpl/Documents/Python_trial/Ax3_Data/KLC/KLC_sleeplog.csv.xlsx",
             outliers.only = TRUE,
             criterror = 4,
             relyonsleeplog = FALSE,
             sleeplogidnum = TRUE,
             colid=1,
             coln1=2,
             do.visual = TRUE,
             nnights = 9,
             #-------------------------------
             # Part 5:
             # Key functions: Merging physical activity with sleep analyses
             #-------------------------------
             threshold.lig = c(30), threshold.mod = c(100),  threshold.vig = c(400),
             boutcriter = 0.8,      boutcriter.in = 0.9,     boutcriter.lig = 0.8,
             boutcriter.mvpa = 0.8, boutdur.in = c(1,10,30), boutdur.lig = c(1,10),
             boutdur.mvpa = c(1),   timewindow = c("WW"),
             #-----------------------------------
             # Report generation
             #-------------------------------
             # Key functions: Generating reports based on meta-data
             do.report=c(2,4,5),
             visualreport=TRUE,     dofirstpage = TRUE,
             viewingwindow=1)

