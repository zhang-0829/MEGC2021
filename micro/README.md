===========================

## Micro-expression spotting
This is our implementation for micro-expression spotting in MEGC 2021.


## Requirements

tensorflow-gpu >= 1.13.1
keras >= 2.4.2
numpy >= 1.16.0
pandas >= 1.1.5
dlib >= 19.21.0
opencv-python >= 3.4.2.17
h5py == 2.10.0

## procedure

1. Data_preprocess

2. Train

3. Evaluation


## Documents Structure
├── Readme.md                   
├── datasets                         
│   └── CAS
│       └── S15
│             └── 15_0101disgustingteeth
│                  └── img_1.jpg
│                    ...
│                  └── img_2052.jpg
		    ...
│       └── S40		  
│   └── SAMM
│       └── 006_1
│       		└── 006_1_0001.jpg
│       		  ...
│       		└── 006_1_8748.jpg
          ...
│       └── 037_7
├── models                         
│   └── CAS
│       └── S15
│             └── 15.h5
 		    ...
│       └── S40
│             └── 15.h5		  
│   └── SAMM
│       └── 006
│       		└── 006.h5
│         ...
│       └── 037
│       		└── 037.h5
├── data_preprocess                      
│   ├── CAS_Apex_move.py          
│   ├── SAMM_Apex_move.py      
│   ├── CAS_loso_move.py           
│   ├── SAMM_loso_move.py       
├── train                     				    
│   ├── CAS_train.py                     
│   ├── SAMM_train.py                 
├── eval                     				    
│   ├── CAS_eval.py                     
│   ├── SAMM_eval.py                 
└── tools

## Notes

The document 'shape_predictor_68_face_landmarks.dat' can be downloaded on the Internet.
